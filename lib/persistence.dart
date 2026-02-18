import 'dart:io';
import 'dart:typed_data';

import 'gpu_tensor.dart';
import 'nn.dart';

/// Saves GPU weights to a flat binary file.
Future<void> saveModuleBinary(Module module, String filePath) async {
  final List<Tensor> parameters = module.parameters();
  final BytesBuilder builder = BytesBuilder();

  print('📦 Syncing GPU weights for saving...');

  for (var p in parameters) {
    // p.data pulls from GPU via engine.getTensorData
    final floatList = Float32List.fromList(p.data);
    builder.add(floatList.buffer.asUint8List());
  }

  final File file = File(filePath);
  await file.writeAsBytes(builder.toBytes());
  print('✅ Saved: $filePath (${file.lengthSync()} bytes)');
}

/// Loads weights from a binary file and pushes them back to the GPU.
Future<void> loadModuleBinary(Module module, String filePath) async {
  final file = File(filePath);
  if (!await file.exists()) {
    throw FileSystemException('Binary weight file not found', filePath);
  }

  // 1. Read all bytes from disk
  final Uint8List allBytes = await file.readAsBytes();

  // 2. Interpret the bytes as 32-bit floats
  final Float32List allFloats = allBytes.buffer.asFloat32List();

  // 3. Get the list of tensors we need to fill
  final List<Tensor> params = module.parameters();

  // --- START OF SAFETY CHECK ---
  final int totalExpectedElements = params.fold(0, (sum, p) => sum + p.length);

  if (allFloats.length != totalExpectedElements) {
    throw Exception(
      'Weight file size mismatch!\n'
      'Model expects: $totalExpectedElements floats\n'
      'File contains: ${allFloats.length} floats\n'
      'Check if your model architecture (hiddenSize, layers) matches the saved file.',
    );
  }
  // --- END OF SAFETY CHECK ---

  print('🚀 Injecting weights into GPU...');

  int offset = 0;
  for (var p in params) {
    final int len = p.length;

    // Extract the exact slice for this tensor
    final List<double> weightSlice = allFloats.sublist(offset, offset + len);

    // p.data = values pushes to GPU via engine.setTensorData
    p.data = weightSlice;

    offset += len;
  }

  print('✨ Model weights successfully restored to GPU memory.');
}
