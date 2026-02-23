import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:dart_cuda/gpu_tensor.dart';

class TripletLoader {
  final String rootPath;
  final int imageSize;
  final Map<String, List<File>> _identityMap = {};
  final Random _random = Random();

  TripletLoader(this.rootPath, this.imageSize) {
    _scanDataset();
  }

  void _scanDataset() {
    final rootDir = Directory(rootPath);
    for (var entity in rootDir.listSync()) {
      if (entity is Directory) {
        final personName = entity.path.split(Platform.pathSeparator).last;
        _identityMap[personName] = entity
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.jpg') || f.path.endsWith('.png'))
            .toList();
      }
    }
  }

  /// Loads a batch of [batchSize] triplets directly into GPU memory
  Map<String, Tensor> nextBatch(int batchSize, List<Tensor> tracker) {
    List<double> anchors = [];
    List<double> positives = [];
    List<double> negatives = [];

    for (int i = 0; i < batchSize; i++) {
      // 1. Pick a random person for Anchor & Positive
      final people = _identityMap.keys.toList();
      final personA = people[_random.nextInt(people.length)];
      final imagesA = _identityMap[personA]!;

      final imgIdx1 = _random.nextInt(imagesA.length);
      int imgIdx2;
      do {
        imgIdx2 = _random.nextInt(imagesA.length);
      } while (imgIdx1 == imgIdx2 && imagesA.length > 1);

      // 2. Pick a different person for Negative
      String personB;
      do {
        personB = people[_random.nextInt(people.length)];
      } while (personA == personB);
      final imagesB = _identityMap[personB]!;

      // 3. Process and Flatten
      anchors.addAll(_processImage(imagesA[imgIdx1]));
      positives.addAll(_processImage(imagesA[imgIdx2]));
      negatives.addAll(_processImage(imagesB[_random.nextInt(imagesB.length)]));
    }

    final anchorTensor = Tensor.fromList([
      batchSize,
      imageSize * imageSize * 3,
    ], anchors);
    final positiveTensor = Tensor.fromList([
      batchSize,
      imageSize * imageSize * 3,
    ], positives);
    final negativeTensor = Tensor.fromList([
      batchSize,
      imageSize * imageSize * 3,
    ], negatives);

    tracker.addAll([anchorTensor, positiveTensor, negativeTensor]);

    return {
      'anchor': anchorTensor,
      'positive': positiveTensor,
      'negative': negativeTensor,
    };
  }

  List<double> _processImage(File file) {
    final raw = img.decodeImage(file.readAsBytesSync());
    final resized = img.copyResize(raw!, width: imageSize, height: imageSize);
    // Convert to normalized Float32 values (0.0 to 1.0)
    return resized
        .map((p) => [p.r / 255.0, p.g / 255.0, p.b / 255.0])
        .expand((e) => e)
        .toList();
  }
}
