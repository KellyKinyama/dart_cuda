import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class TripletLoader {
  final String rootPath;
  final int imageSize;

  // 🔥 FIX 1: Change Map type to store the processed pixels, not the File pointers
  final Map<String, List<Float32List>> _identityMap = {};
  final Random _random = Random();

  TripletLoader(this.rootPath, this.imageSize, int numOfFiles) {
    _scanDataset(numOfFiles);
  }

  void _scanDataset(int numOfFiles) {
    int scannedFiles = 0;
    final rootDir = Directory(rootPath);
    if (!rootDir.existsSync()) return;

    print("Caching images to RAM as Float32List...");

    for (var entity in rootDir.listSync()) {
      if (entity is Directory) {
        final personName = entity.path.split(Platform.pathSeparator).last;

        final List<Float32List> processedImages = [];

        final files = entity.listSync().whereType<File>().where(
          (f) => f.path.endsWith('.jpg') || f.path.endsWith('.png'),
        );

        for (var file in files) {
          // 🔥 FIX 2: Process the file into pixels immediately
          processedImages.add(_processImage(file));
        }

        if (processedImages.isNotEmpty) {
          _identityMap[personName] = processedImages;
        }
      }
      if (scannedFiles > numOfFiles) break;
      scannedFiles++;
    }
    print("Loaded ${_identityMap.length} identities.");
  }

  // 🔥 FIX 3: Change return type to Float32List for high-performance memory packing
  Float32List _processImage(File file) {
    final bytes = file.readAsBytesSync();
    final raw = img.decodeImage(bytes);
    if (raw == null) throw Exception("Could not decode ${file.path}");

    final resized = img.copyResize(raw, width: imageSize, height: imageSize);

    // We use a flat Float32List to avoid the overhead of nested lists/Iterables
    final floatData = Float32List(imageSize * imageSize * 3);
    int idx = 0;

    for (var pixel in resized) {
      floatData[idx++] = pixel.r / 255.0;
      floatData[idx++] = pixel.g / 255.0;
      floatData[idx++] = pixel.b / 255.0;
    }
    return floatData;
  }

  Map<String, Float32List> nextBatch(int batchSize) {
    final int features = imageSize * imageSize * 3;
    final anchorBatch = Float32List(batchSize * features);
    final positiveBatch = Float32List(batchSize * features);
    final negativeBatch = Float32List(batchSize * features);

    final people = _identityMap.keys.toList();

    for (int i = 0; i < batchSize; i++) {
      // 1. Pick Anchor/Positive
      final personA = people[_random.nextInt(people.length)];
      final imagesA = _identityMap[personA]!;

      final idx1 = _random.nextInt(imagesA.length);
      int idx2 = _random.nextInt(imagesA.length);
      if (imagesA.length > 1) {
        while (idx1 == idx2) {
          idx2 = _random.nextInt(imagesA.length);
        }
      }

      // 2. Pick Negative
      String personB;
      do {
        personB = people[_random.nextInt(people.length)];
      } while (personA == personB);
      final imagesB = _identityMap[personB]!;
      final idx3 = _random.nextInt(imagesB.length);

      // 3. Set pixel data into the batch lists (Types now match: Float32List to Float32List)
      anchorBatch.setAll(i * features, imagesA[idx1]);
      positiveBatch.setAll(i * features, imagesA[idx2]);
      negativeBatch.setAll(i * features, imagesB[idx3]);
    }

    return {
      'anchor': anchorBatch,
      'positive': positiveBatch,
      'negative': negativeBatch,
    };
  }
}
