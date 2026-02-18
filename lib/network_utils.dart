import 'dart:io';
import 'dart:typed_data';

import 'gpu_tensor.dart';
import 'nn.dart';

/// Saves GPU weights to a flat binary file.
Future<void> saveModuleBinary(Module module, String filePath) async {
  final List<Tensor> parameters = module.parameters();
  final BytesBuilder builder = BytesBuilder();

  for (var p in parameters) {
    // 1. Get List<double> from GPU (via your data getter)
    // 2. Convert to Float32List
    // 3. Get the raw byte buffer
    final floatList = Float32List.fromList(p.data);
    builder.add(floatList.buffer.asUint8List());
  }

  await File(filePath).writeAsBytes(builder.toBytes());
  print(
    '💾 GPU weights saved as Binary: $filePath (${File(filePath).lengthSync()} bytes)',
  );
}

/// Loads weights from a flat binary file and pushes them to the GPU.
Future<void> loadModuleBinary(Module module, String filePath) async {
  final file = File(filePath);
  if (!await file.exists()) return print('Error: Binary file not found');

  final Uint8List allBytes = await file.readAsBytes();
  final Float32List allFloats = allBytes.buffer.asFloat32List();
  final List<Tensor> params = module.parameters();

  int offset = 0;
  for (var p in params) {
    // Extract the slice of floats belonging to this specific tensor
    final int len = p.length;
    final List<double> weightSlice = allFloats.sublist(offset, offset + len);

    // Push to GPU using your 'set data' setter
    p.data = weightSlice;

    offset += len;
  }
  print('🚀 Binary weights injected into GPU memory.');
}
