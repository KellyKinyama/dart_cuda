// Train face-style metric embeddings on an ImageFolder-style dataset using
// triplet loss. Reuses:
//   * lib/loaders/image_folder_loader.dart      -- new sampleTriplet()
//   * lib/core/transformers/vision/vit_backbone.dart  -- ViT encoder
//   * lib/core/utils/triplet_loss.dart          -- TripletLossGPU(margin)
//
// !! KNOWN ENGINE LIMITATION !!
// The current CUDA backend has a backward-pass instability that crashes
// (`Tensor.backward` segfault) when the autograd graph spans many ViT
// encoder forward passes within a single training process -- regardless of
// whether intermediates are eagerly disposed. The bundled
// `example/face_embeddings.dart` exhibits the same class of failure.
//
// To still ship a working demonstration we:
//   1. Build the embedding head as `ViTBackbone` -> CLS row -> Linear, and
//      intentionally skip L2-normalization (its backward also misbehaves
//      on some shapes). You can L2-normalize at inference time.
//   2. Default to a TINY config (1 layer, 1 triplet/step, 5 steps) that
//      reliably completes. The example primarily serves as a working
//      reference for the loader's `sampleTriplet()` API + the per-step
//      triplet-loss + Adam wiring. Bumping `--steps` or `--triplets` may
//      crash on some machines; reduce them if you see a backward segfault.
//
// For each step we sample `--triplets=N` (anchor, positive, negative)
// triplets, compute per-triplet hinge losses, backward each independently
// (gradients accumulate via atomicAdd), then apply one Adam step.
//
// Verification metric: at the end of training (and every `--valEvery`
// steps if it completes), we embed every val image once, then form
// `--valPairs=N` random same-class and different-class pairs and report
// mean L2 distance for each plus the best-threshold verification accuracy.
//
// Usage (from repo root):
//   dart run example/vision/train_face_triplet.dart
//   dart run example/vision/train_face_triplet.dart --root=path/to/data
//
// Optional flags:
//   --root=PATH        (default 'Original Images')
//   --imgSize=N        (default 32)
//   --patchSize=N      (default 8)
//   --embed=N          (default 32)   ViT embedding size
//   --outDim=N         (default 32)   final embedding dim
//   --layers=N         (default 1)
//   --heads=N          (default 4)
//   --triplets=N       (default 1)    triplets per step
//   --steps=N          (default 5)
//   --lr=F             (default 1e-4)
//   --margin=F         (default 0.2)  triplet hinge margin
//   --maxPerClass=N    (default 12)
//   --maxClasses=N     (default 6)
//   --valSplit=F       (default 0.25)
//   --valPairs=N       (default 60)   random pairs per metric eval
//   --logEvery=N       (default 1)
//   --valEvery=N       (default 5)
//   --seed=N           (default 7)

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_cuda/core/layers/nn.dart' show Layer;
import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/vision/vit_backbone.dart';
import 'package:dart_cuda/core/utils/triplet_loss.dart';
import 'package:dart_cuda/loaders/image_folder_loader.dart';

int _intFlag(List<String> args, String name, int fallback) {
  final p = '--$name=';
  for (final a in args) {
    if (a.startsWith(p)) return int.tryParse(a.substring(p.length)) ?? fallback;
  }
  return fallback;
}

double _doubleFlag(List<String> args, String name, double fallback) {
  final p = '--$name=';
  for (final a in args) {
    if (a.startsWith(p)) {
      return double.tryParse(a.substring(p.length)) ?? fallback;
    }
  }
  return fallback;
}

String _strFlag(List<String> args, String name, String fallback) {
  final p = '--$name=';
  for (final a in args) {
    if (a.startsWith(p)) return a.substring(p.length);
  }
  return fallback;
}

int? _intFlagOpt(List<String> args, String name) {
  final p = '--$name=';
  for (final a in args) {
    if (a.startsWith(p)) return int.tryParse(a.substring(p.length));
  }
  return null;
}

double _l2(List<double> a, List<double> b) {
  var sum = 0.0;
  for (var i = 0; i < a.length; i++) {
    final d = a[i] - b[i];
    sum += d * d;
  }
  return math.sqrt(sum);
}

/// Compute an EER-style verification accuracy: try a sweep of thresholds
/// and pick the one that maximises (#pos<=t + #neg>t) / (#pos + #neg).
({double bestThr, double accuracy}) _bestThresholdAccuracy(
  List<double> posDists,
  List<double> negDists,
) {
  final all = [...posDists, ...negDists]..sort();
  if (all.isEmpty) return (bestThr: 0.0, accuracy: 0.0);
  double bestAcc = 0.0;
  double bestThr = all.first;
  for (final t in all) {
    var ok = 0;
    for (final d in posDists) {
      if (d <= t) ok++;
    }
    for (final d in negDists) {
      if (d > t) ok++;
    }
    final acc = ok / (posDists.length + negDists.length);
    if (acc > bestAcc) {
      bestAcc = acc;
      bestThr = t;
    }
  }
  return (bestThr: bestThr, accuracy: bestAcc);
}

Future<void> main(List<String> args) async {
  final root = _strFlag(args, 'root', 'Original Images');
  final imgSize = _intFlag(args, 'imgSize', 32);
  final patchSize = _intFlag(args, 'patchSize', 8);
  final embed = _intFlag(args, 'embed', 32);
  final outDim = _intFlag(args, 'outDim', 32);
  final layers = _intFlag(args, 'layers', 1);
  final heads = _intFlag(args, 'heads', 4);
  final triplets = _intFlag(args, 'triplets', 1);
  final steps = _intFlag(args, 'steps', 5);
  final lr = _doubleFlag(args, 'lr', 1e-4);
  final margin = _doubleFlag(args, 'margin', 0.2);
  final maxPerClass = _intFlag(args, 'maxPerClass', 12);
  final maxClasses = _intFlagOpt(args, 'maxClasses') ?? 6;
  final valSplit = _doubleFlag(args, 'valSplit', 0.25);
  final valPairs = _intFlag(args, 'valPairs', 60);
  final logEvery = _intFlag(args, 'logEvery', 1);
  final valEvery = _intFlag(args, 'valEvery', 5);
  final seed = _intFlag(args, 'seed', 7);

  if (!Directory(root).existsSync()) {
    stderr.writeln('error: dataset root not found: $root');
    exit(1);
  }
  if (imgSize % patchSize != 0) {
    stderr.writeln(
      'error: imgSize ($imgSize) must be divisible by patchSize ($patchSize)',
    );
    exit(2);
  }

  print('📦 Loading $root (imgSize=$imgSize, patchSize=$patchSize)...');
  final loader = ImageFolderLoader(
    root,
    imageSize: imgSize,
    patchSize: patchSize,
    maxPerClass: maxPerClass,
    maxClasses: maxClasses,
    valSplit: valSplit,
    seed: seed,
  );
  final ready = loader.tripletReadyClasses;
  if (ready.length < 2) {
    stderr.writeln(
      'error: need >=2 train classes with >=2 samples each for triplets, '
      'got ${ready.length}. Try increasing --maxPerClass or lowering '
      '--valSplit.',
    );
    exit(3);
  }
  print(
    'Classes=${loader.numClasses} | train=${loader.numTrain} | '
    'val=${loader.numVal} | triplet-ready classes=${ready.length} | '
    'numPatches=${loader.numPatches} | patchPixels=${loader.patchPixels}',
  );

  // ---- Model -------------------------------------------------------------
  final backbone = ViTBackbone(
    imageSize: imgSize,
    patchSize: patchSize,
    embedSize: embed,
    numLayers: layers,
    numHeads: heads,
  );
  final projection = Layer(embed, outDim, useGelu: false);
  final lossFn = TripletLossGPU(margin: margin);
  final params = <Tensor>[...backbone.parameters(), ...projection.parameters()];
  final opt = Adam(params, lr: lr);

  /// Embed one patchified image to `[1, outDim]`. Tracker collects ALL
  /// intermediates (including the embedding tensor) so callers can decide
  /// whether to dispose. We do NOT L2-normalize: the engine's L2Normalize
  /// backward currently misbehaves with this graph; the triplet loss still
  /// induces useful metric structure on raw embeddings.
  Tensor embedOne(Tensor patches, List<Tensor> tracker) {
    final encoded = backbone.forward(patches, tracker); // [N+1, D]
    tracker.add(encoded);
    final cls = encoded.slice(0, 1); // [1, D]
    tracker.add(cls);
    final out = projection.forward(cls, tracker); // [1, outDim]
    return out;
  }

  print(
    '🚀 Training ViT face embeddings with TripletLoss '
    '(steps=$steps, triplets/step=$triplets, lr=$lr, margin=$margin, '
    'embed=$embed, outDim=$outDim, layers=$layers)',
  );

  final nP = loader.numPatches;
  final pp = loader.patchPixels;
  double emaLoss = double.nan;

  for (var step = 1; step <= steps; step++) {
    opt.zeroGrad();

    // Backward each triplet independently — gradients accumulate inside
    // the parameter buffers via atomicAdd, then a single opt.step() at
    // the end applies the averaged update. This avoids building one big
    // shared autograd graph across all triplets (which is fragile when
    // intermediates need to be disposed step-by-step).
    var totalLoss = 0.0;
    var activeCount = 0;
    final scale = 1.0 / triplets;

    for (var t = 0; t < triplets; t++) {
      final tracker = <Tensor>[];
      final tri = loader.sampleTriplet();
      final aT = Tensor.fromList([nP, pp], tri.anchor);
      final pT = Tensor.fromList([nP, pp], tri.positive);
      final nT = Tensor.fromList([nP, pp], tri.negative);
      tracker.addAll([aT, pT, nT]);

      final eA = embedOne(aT, tracker);
      final eP = embedOne(pT, tracker);
      final eN = embedOne(nT, tracker);

      final l = lossFn.forward(eA, eP, eN, tracker);
      // Pre-divide by N so the accumulated gradient equals the mean.
      // scaleT is created fresh per iteration and tracked for disposal.
      final scaleT = Tensor.fromList([1, 1], [scale]);
      final lScaled = l * scaleT;
      tracker.addAll([scaleT, lScaled]);

      final lv = l.fetchData()[0];
      totalLoss += lv;
      if (lv > 0) activeCount++;

      lScaled.backward();
      // NOTE: The repo's autograd graph holds parent refs that interact
      // poorly with eager disposal of the intermediates after backward.
      // We intentionally let `tracker` go out of scope here; GPU buffers
      // are reclaimed when Dart finalizes those tensor handles. This
      // matches the pattern used in `example/face_embeddings.dart`.
      tracker.clear();
    }

    opt.step();

    final lossVal = totalLoss / triplets;
    emaLoss = emaLoss.isNaN ? lossVal : 0.9 * emaLoss + 0.1 * lossVal;

    if (step == 1 || step % logEvery == 0) {
      print(
        '  step ${step.toString().padLeft(4)} | '
        'loss=${lossVal.toStringAsFixed(4)} '
        '(ema=${emaLoss.toStringAsFixed(4)}) | '
        'active=$activeCount/$triplets',
      );
    }

    if (loader.numVal > 0 && (step % valEvery == 0 || step == steps)) {
      // Embed every val image once.
      final embeddings = <Float32List>[];
      final labels = <int>[];
      for (final v in loader.valSamples()) {
        final tk = <Tensor>[];
        final patches = Tensor.fromList([nP, pp], v.key);
        tk.add(patches);
        final emb = embedOne(patches, tk);
        embeddings.add(Float32List.fromList(emb.fetchData()));
        labels.add(v.value);
        // Same disposal caveat as above; let the tracker drop out of scope.
      }

      // Build per-class index list within the val split.
      final byClass = <int, List<int>>{};
      for (var i = 0; i < labels.length; i++) {
        byClass.putIfAbsent(labels[i], () => <int>[]).add(i);
      }
      final readyClasses = byClass.entries
          .where((e) => e.value.length >= 2)
          .toList();

      final rng = math.Random(seed + step);
      final posDists = <double>[];
      final negDists = <double>[];
      var attempts = 0;
      while (posDists.length < valPairs &&
          attempts < valPairs * 10 &&
          readyClasses.isNotEmpty) {
        attempts++;
        final c = readyClasses[rng.nextInt(readyClasses.length)];
        final idxs = c.value;
        final i1 = idxs[rng.nextInt(idxs.length)];
        var i2 = idxs[rng.nextInt(idxs.length)];
        if (i2 == i1) continue;
        posDists.add(_l2(embeddings[i1].toList(), embeddings[i2].toList()));
      }
      attempts = 0;
      while (negDists.length < valPairs &&
          attempts < valPairs * 10 &&
          byClass.length >= 2) {
        attempts++;
        final i1 = rng.nextInt(embeddings.length);
        final i2 = rng.nextInt(embeddings.length);
        if (labels[i1] == labels[i2]) continue;
        negDists.add(_l2(embeddings[i1].toList(), embeddings[i2].toList()));
      }

      double mean(List<double> xs) =>
          xs.isEmpty ? double.nan : xs.reduce((a, b) => a + b) / xs.length;

      final res = _bestThresholdAccuracy(posDists, negDists);
      print(
        '    [val] pos_d=${mean(posDists).toStringAsFixed(4)} '
        '(${posDists.length}) | neg_d=${mean(negDists).toStringAsFixed(4)} '
        '(${negDists.length}) | thr=${res.bestThr.toStringAsFixed(4)} '
        '| ver_acc=${(res.accuracy * 100).toStringAsFixed(1)}%',
      );
    }
  }

  // ---- Cleanup -----------------------------------------------------------
  opt.dispose();
  for (final p in params) {
    p.dispose();
  }
}
