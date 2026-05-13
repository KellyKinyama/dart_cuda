// Train a tiny ViT image classifier on a folder-organised image dataset.
//
// Default dataset is the bundled `Original Images/` celebrity folder
// (31 classes, ~2.5k images). Pass `--root=path/to/dataset` to point at
// any other ImageFolder-style directory:
//
//   <root>/
//     <class_a>/  *.jpg|*.png
//     <class_b>/  *.jpg|*.png
//     ...
//
// Pipeline per training step:
//   1. Sample a mini-batch of images from the train split.
//   2. For each image, patchify -> ViTBackbone forward -> CLS row ->
//      Linear classifier head -> [1, numClasses] logits.
//   3. Sum cross-entropy across the mini-batch (one CE per image), call
//      backward, Adam step.
//   4. Periodically evaluate on the held-out val split.
//
// Usage (from repo root):
//   dart run example/vision/train_image_classifier.dart
//   dart run example/vision/train_image_classifier.dart --root=path/to/data
//
// Optional flags:
//   --root=PATH        (default 'Original Images')
//   --imgSize=N        (default 32)    image is resized to NxN
//   --patchSize=N      (default 8)     ViT patch side; imgSize % patchSize=0
//   --embed=N          (default 64)    ViT embedding size
//   --layers=N         (default 2)     number of transformer encoder layers
//   --heads=N          (default 4)     attention heads
//   --batch=N          (default 8)     mini-batch size
//   --steps=N          (default 400)   total optimizer steps
//   --lr=F             (default 5e-4)  Adam learning rate
//   --maxPerClass=N    (default 60)    cap images per class
//   --maxClasses=N     (none)          cap number of classes
//   --valSplit=F       (default 0.2)   fraction of images held out for val
//   --logEvery=N       (default 20)    log train loss/acc every N steps
//   --valEvery=N       (default 100)   evaluate val loss/acc every N steps
//   --seed=N           (default 7)     RNG seed

import 'dart:io';

import 'package:dart_cuda/core/layers/nn.dart' show Layer;
import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/vision/vit_backbone.dart';
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

int _argmax(List<double> xs) {
  var best = 0;
  var bestV = xs[0];
  for (var i = 1; i < xs.length; i++) {
    if (xs[i] > bestV) {
      bestV = xs[i];
      best = i;
    }
  }
  return best;
}

/// Forward one patchified image through (ViT backbone -> CLS row -> head)
/// returning the `[1, numClasses]` logits tensor.
Tensor _classifyOne(
  ViTBackbone backbone,
  Layer head,
  Tensor patchifiedImage,
  List<Tensor> tracker,
) {
  final encoded = backbone.forward(patchifiedImage, tracker); // [N+1, D]
  tracker.add(encoded);
  final cls = encoded.slice(0, 1); // [1, D]
  tracker.add(cls);
  final logits = head.forward(cls, tracker); // [1, numClasses]
  return logits;
}

Future<void> main(List<String> args) async {
  final root = _strFlag(args, 'root', 'Original Images');
  final imgSize = _intFlag(args, 'imgSize', 32);
  final patchSize = _intFlag(args, 'patchSize', 8);
  final embed = _intFlag(args, 'embed', 64);
  final layers = _intFlag(args, 'layers', 2);
  final heads = _intFlag(args, 'heads', 4);
  final batch = _intFlag(args, 'batch', 8);
  final steps = _intFlag(args, 'steps', 400);
  final lr = _doubleFlag(args, 'lr', 5e-4);
  final maxPerClass = _intFlag(args, 'maxPerClass', 60);
  final maxClasses = _intFlagOpt(args, 'maxClasses');
  final valSplit = _doubleFlag(args, 'valSplit', 0.2);
  final logEvery = _intFlag(args, 'logEvery', 20);
  final valEvery = _intFlag(args, 'valEvery', 100);
  final seed = _intFlag(args, 'seed', 7);

  if (!Directory(root).existsSync()) {
    stderr.writeln('error: dataset root not found: $root');
    exit(1);
  }
  if (imgSize % patchSize != 0) {
    stderr.writeln('error: imgSize ($imgSize) must be divisible by patchSize ($patchSize)');
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
  print(
    'Classes=${loader.numClasses} | train=${loader.numTrain} | '
    'val=${loader.numVal} | numPatches=${loader.numPatches} | '
    'patchPixels=${loader.patchPixels}',
  );
  if (loader.numTrain == 0) {
    stderr.writeln('error: no training images');
    exit(3);
  }

  // ---- Model ---------------------------------------------------------------
  final backbone = ViTBackbone(
    imageSize: imgSize,
    patchSize: patchSize,
    embedSize: embed,
    numLayers: layers,
    numHeads: heads,
  );
  final head = Layer(embed, loader.numClasses, useGelu: false);

  final params = <Tensor>[...backbone.parameters(), ...head.parameters()];
  final opt = Adam(params, lr: lr);

  print(
    '🚀 Training ViT classifier '
    '(steps=$steps, batch=$batch, lr=$lr, embed=$embed, layers=$layers, heads=$heads)',
  );

  double emaLoss = double.nan;

  for (var step = 1; step <= steps; step++) {
    final samples = loader.sampleTrainBatch(batch);

    opt.zeroGrad();
    final tracker = <Tensor>[];

    // Per-image forward + accumulated cross-entropy.
    Tensor? lossSum;
    var correct = 0;
    final nP = loader.numPatches;
    final pp = loader.patchPixels;
    for (final entry in samples) {
      final patches = Tensor.fromList([nP, pp], entry.key);
      tracker.add(patches);
      final logits = _classifyOne(backbone, head, patches, tracker);
      final ce = logits.crossEntropy([entry.value]);
      tracker.add(ce);
      if (lossSum == null) {
        lossSum = ce;
      } else {
        lossSum = lossSum + ce;
        tracker.add(lossSum);
      }
      // Track top-1 correctness from logits (no extra forward).
      if (_argmax(logits.fetchData()) == entry.value) correct++;
    }
    if (lossSum == null) continue; // empty batch shouldn't happen
    // Mean over batch (CE is already mean over T=1 per image, so divide by B).
    final invB = Tensor.fromList([1, 1], [1.0 / batch]);
    final meanLoss = lossSum * invB;
    tracker.addAll([invB, meanLoss]);

    meanLoss.backward();
    opt.step();

    final lossVal = meanLoss.fetchData()[0];
    final acc = correct / batch;
    emaLoss = emaLoss.isNaN ? lossVal : 0.9 * emaLoss + 0.1 * lossVal;

    // Free the per-step tracker (params are excluded by virtue of not being added).
    final paramAddrs = params.map((p) => p.handle.address).toSet();
    final freed = <int>{};
    for (final t in tracker) {
      final a = t.handle.address;
      if (a == 0 || freed.contains(a) || paramAddrs.contains(a)) continue;
      t.dispose();
      freed.add(a);
    }

    if (step == 1 || step % logEvery == 0) {
      print(
        '  step ${step.toString().padLeft(4)} | '
        'loss=${lossVal.toStringAsFixed(4)} '
        '(ema=${emaLoss.toStringAsFixed(4)}) | '
        'train_acc=${(acc * 100).toStringAsFixed(1)}%',
      );
    }

    if (loader.numVal > 0 && (step % valEvery == 0 || step == steps)) {
      var vCorrect = 0;
      var vTotal = 0;
      var vLossSum = 0.0;
      for (final v in loader.valSamples()) {
        final tk = <Tensor>[];
        final patches = Tensor.fromList([nP, pp], v.key);
        tk.add(patches);
        final logits = _classifyOne(backbone, head, patches, tk);
        final ce = logits.crossEntropy([v.value]);
        tk.add(ce);
        vLossSum += ce.fetchData()[0];
        if (_argmax(logits.fetchData()) == v.value) vCorrect++;
        vTotal++;
        for (final t in tk) {
          final a = t.handle.address;
          if (a != 0 && !paramAddrs.contains(a)) t.dispose();
        }
      }
      final vLoss = vLossSum / vTotal;
      print(
        '    [val] loss=${vLoss.toStringAsFixed(4)} | '
        'acc=${(vCorrect / vTotal * 100).toStringAsFixed(1)}% | '
        'images=$vTotal',
      );
    }
  }

  // ---- Cleanup ------------------------------------------------------------
  opt.dispose();
  for (final p in params) {
    p.dispose();
  }
}
