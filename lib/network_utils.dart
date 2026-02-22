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

/// Reads a binary file and pushes the weights into GPU VRAM
Future<bool> loadModuleBinary(Module model, String filePath) async {
  final file = File(filePath);
  if (!await file.exists()) return false;

  final Uint8List allBytes = await file.readAsBytes();
  final Float32List allFloats = allBytes.buffer.asFloat32List();
  final List<Tensor> params = model.parameters();

  // Safety Check
  final int totalExpected = params.fold(0, (sum, p) => sum + p.length);
  if (allFloats.length != totalExpected) {
    print(
      '⚠️ Mismatch! Model needs $totalExpected floats, file has ${allFloats.length}',
    );
    return false;
  }

  print('🚀 Injecting weights into GPU VRAM...');
  int offset = 0;
  for (var p in params) {
    final int len = p.length;
    // p.data (setter) triggers engine.setTensorData -> cudaMemcpyHostToDevice
    p.data = allFloats.sublist(offset, offset + len).toList();
    offset += len;
  }
  return true;
}
