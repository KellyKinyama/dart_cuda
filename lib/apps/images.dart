import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:dart_cuda/gpu_tensor.dart';

// void main() async {
//   // Create a 256x256 8-bit (default) rgb (default) image.
//   final image = img.Image(width: 256, height: 256);
//   // Iterate over its pixels
//   for (var pixel in image) {
//     // Set the pixels red value to its x position value, creating a gradient.
//     pixel
//       ..r = pixel.x
//       // Set the pixels green value to its y position value.
//       ..g = pixel.y;
//   }
//   // Encode the resulting image to the PNG image format.
//   final png = img.encodePng(image);
//   // Write the PNG formatted data to a file.
//   await File('image.png').writeAsBytes(png);
// }

Tensor imageToTensor(String path, int targetSize) {
  // 1. Load and Resize
  final bytes = File(path).readAsBytesSync();
  final rawImage = img.decodeImage(bytes);
  if (rawImage == null) throw "Could not decode image";

  final resized = img.copyResize(
    rawImage,
    width: targetSize,
    height: targetSize,
  );

  // 2. Convert to Float32 List (Normalized 0.0 to 1.0)
  final floatData = Float32List(targetSize * targetSize * 3);
  int i = 0;
  for (var pixel in resized) {
    floatData[i++] = pixel.r / 255.0;
    floatData[i++] = pixel.g / 255.0;
    floatData[i++] = pixel.b / 255.0;
  }

  // 3. Upload to GPU
  // Shape: [1, pixels] or [numPatches, pixels] depending on your ViT setup
  return Tensor.fromList([1, floatData.length], floatData);
}

void main() {
  print(imageToTensor('image.png', 32).printMatrix());
}
