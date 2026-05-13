// Image-folder classification loader.
//
// Mirrors the directory layout used by `Original Images/`:
//
//   <root>/
//     <class_0>/  *.jpg|*.png
//     <class_1>/  *.jpg|*.png
//     ...
//
// On construction every image is decoded once with `package:image`, resized
// to `imageSize x imageSize`, normalized to [0, 1], and cached in RAM as a
// flat `Float32List` of length `imageSize * imageSize * 3` (channels-last
// RGB, layout (y, x, c)).
//
// Designed to feed a ViT encoder: `nextBatchPatchified` returns one
// `[numPatches, patchPixels]` tensor per image plus its int class label,
// where each patch is a `patchSize x patchSize x 3` block flattened.
//
// Train/val splitting is done deterministically per-class given a seed.

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

class _Sample {
  final Float32List flat; // [H*W*3], normalized [0,1], channels-last
  final int label;
  _Sample(this.flat, this.label);
}

class ImageFolderLoader {
  final String rootPath;
  final int imageSize;
  final int patchSize;
  late final List<String> classes;
  final List<_Sample> _train = [];
  final List<_Sample> _val = [];
  final math.Random _rng;

  int get numClasses => classes.length;
  int get numTrain => _train.length;
  int get numVal => _val.length;
  int get patchPixels => patchSize * patchSize * 3;
  int get numPatches =>
      (imageSize ~/ patchSize) * (imageSize ~/ patchSize);

  ImageFolderLoader(
    this.rootPath, {
    required this.imageSize,
    required this.patchSize,
    int maxPerClass = 1 << 30,
    int? maxClasses,
    double valSplit = 0.2,
    int seed = 7,
  }) : _rng = math.Random(seed) {
    if (imageSize % patchSize != 0) {
      throw ArgumentError(
        'imageSize ($imageSize) must be divisible by patchSize ($patchSize)',
      );
    }
    final root = Directory(rootPath);
    if (!root.existsSync()) {
      throw ArgumentError('rootPath does not exist: $rootPath');
    }

    // Collect class directories (sorted for determinism).
    final classDirs = root
        .listSync()
        .whereType<Directory>()
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    if (maxClasses != null && classDirs.length > maxClasses) {
      classDirs.removeRange(maxClasses, classDirs.length);
    }
    final mutableClasses = <String>[];
    final splitRng = math.Random(seed);

    for (var classIdx = 0; classIdx < classDirs.length; classIdx++) {
      final dir = classDirs[classIdx];
      final name = dir.path.split(Platform.pathSeparator).last;
      final files = dir
          .listSync()
          .whereType<File>()
          .where((f) {
            final p = f.path.toLowerCase();
            return p.endsWith('.jpg') ||
                p.endsWith('.jpeg') ||
                p.endsWith('.png');
          })
          .toList()
        ..sort((a, b) => a.path.compareTo(b.path));
      if (files.isEmpty) continue;
      final cap = files.length > maxPerClass ? maxPerClass : files.length;
      mutableClasses.add(name);
      // Per-class shuffle so val split isn't all the same filename suffix.
      final indices = List<int>.generate(cap, (i) => i)..shuffle(splitRng);
      final nVal = (cap * valSplit).round();
      for (var k = 0; k < cap; k++) {
        final f = files[indices[k]];
        final flat = _decode(f);
        if (flat == null) continue;
        final s = _Sample(flat, mutableClasses.length - 1);
        if (k < nVal) {
          _val.add(s);
        } else {
          _train.add(s);
        }
      }
    }

    // Replace the const empty `classes` list with the real one.
    classes = mutableClasses;
  }

  Float32List? _decode(File file) {
    try {
      final raw = img.decodeImage(file.readAsBytesSync());
      if (raw == null) return null;
      final resized = img.copyResize(raw,
          width: imageSize, height: imageSize, interpolation: img.Interpolation.linear);
      final flat = Float32List(imageSize * imageSize * 3);
      var i = 0;
      for (final p in resized) {
        flat[i++] = p.r / 255.0;
        flat[i++] = p.g / 255.0;
        flat[i++] = p.b / 255.0;
      }
      return flat;
    } catch (_) {
      return null;
    }
  }

  /// Patchify a flat HxWx3 image (channels-last) into `[numPatches * P*P*3]`,
  /// row-major over patches: patch row `p` contains its `P*P*3` pixel values
  /// laid out as (py_in_patch, px_in_patch, c).
  Float32List patchify(Float32List flat) {
    final H = imageSize, W = imageSize, P = patchSize;
    final perPatch = P * P * 3;
    final nP = numPatches;
    final out = Float32List(nP * perPatch);
    final patchesPerRow = W ~/ P;
    for (var py = 0; py < H ~/ P; py++) {
      for (var px = 0; px < patchesPerRow; px++) {
        final pIdx = py * patchesPerRow + px;
        final outBase = pIdx * perPatch;
        var w = 0;
        for (var dy = 0; dy < P; dy++) {
          final y = py * P + dy;
          for (var dx = 0; dx < P; dx++) {
            final x = px * P + dx;
            final inBase = (y * W + x) * 3;
            out[outBase + w++] = flat[inBase];
            out[outBase + w++] = flat[inBase + 1];
            out[outBase + w++] = flat[inBase + 2];
          }
        }
      }
    }
    return out;
  }

  /// Sample a uniform random batch from the training split.
  /// Returns a list of `(patchifiedFloats[numPatches*patchPixels], label)`.
  List<MapEntry<Float32List, int>> sampleTrainBatch(int batchSize) {
    final out = <MapEntry<Float32List, int>>[];
    for (var i = 0; i < batchSize; i++) {
      final s = _train[_rng.nextInt(_train.length)];
      out.add(MapEntry(patchify(s.flat), s.label));
    }
    return out;
  }

  /// Iterate the full validation split (in cached order). Use for evaluation.
  Iterable<MapEntry<Float32List, int>> valSamples() sync* {
    for (final s in _val) {
      yield MapEntry(patchify(s.flat), s.label);
    }
  }
}
