// Pure-Dart reader for the `safetensors` file format
// (https://github.com/huggingface/safetensors).
//
// File layout:
//   [8 bytes ] little-endian u64 = JSON header length N
//   [N bytes ] UTF-8 JSON of the form
//              { "tensor_name": { "dtype": "F32",
//                                 "shape": [d0, d1, ...],
//                                 "data_offsets": [start, end] },
//                ...,
//                "__metadata__": { ... }              // optional
//              }
//   [rest    ] raw little-endian tensor data; `data_offsets` are byte
//              offsets into this trailing region.
//
// We deliberately do not depend on any package; safetensors is simple
// enough that a 100-line reader is the right level of leverage.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

/// A single tensor entry in a safetensors file.
class SafeTensorRecord {
  final String name;
  final String dtype;
  final List<int> shape;
  final Uint8List bytes; // raw little-endian bytes for this tensor

  SafeTensorRecord({
    required this.name,
    required this.dtype,
    required this.shape,
    required this.bytes,
  });

  int get numel => shape.fold<int>(1, (a, b) => a * b);

  /// Materialize this record as float32 values. Supports F32 directly and
  /// converts F16 / BF16 on the fly. Other dtypes throw — convert them in
  /// Python before exporting (e.g. `tensor.float()`).
  Float32List asFloat32() {
    switch (dtype) {
      case 'F32':
        // The view must be aligned. Copy to be safe against unaligned
        // offsets in the host buffer.
        final out = Float32List(numel);
        final bd = ByteData.sublistView(bytes);
        for (var i = 0; i < numel; i++) {
          out[i] = bd.getFloat32(i * 4, Endian.little);
        }
        return out;
      case 'F16':
        final out = Float32List(numel);
        final bd = ByteData.sublistView(bytes);
        for (var i = 0; i < numel; i++) {
          out[i] = _f16ToF32(bd.getUint16(i * 2, Endian.little));
        }
        return out;
      case 'BF16':
        final out = Float32List(numel);
        final bd = ByteData.sublistView(bytes);
        // bfloat16 = upper 16 bits of float32 (round-toward-zero).
        // Reconstruct float32 by left-shifting into the high half.
        final tmp = ByteData(4);
        for (var i = 0; i < numel; i++) {
          final u16 = bd.getUint16(i * 2, Endian.little);
          tmp.setUint16(2, u16, Endian.little);
          tmp.setUint16(0, 0, Endian.little);
          out[i] = tmp.getFloat32(0, Endian.little);
        }
        return out;
      default:
        throw UnsupportedError(
          'safetensors: dtype "$dtype" for tensor "$name" is not supported. '
          'Cast to float32 in Python before exporting '
          '(e.g. `state_dict[k] = v.float()`).',
        );
    }
  }
}

/// IEEE-754 half (sign 1 / exp 5 / mant 10) → float32.
double _f16ToF32(int h) {
  final sign = (h >> 15) & 0x1;
  final exp = (h >> 10) & 0x1f;
  final mant = h & 0x3ff;
  int f32Bits;
  if (exp == 0) {
    if (mant == 0) {
      f32Bits = sign << 31;
    } else {
      // subnormal half → normalize
      var e = -1;
      var m = mant;
      do {
        e++;
        m <<= 1;
      } while ((m & 0x400) == 0);
      final fExp = 127 - 15 - e;
      f32Bits = (sign << 31) | (fExp << 23) | ((m & 0x3ff) << 13);
    }
  } else if (exp == 0x1f) {
    f32Bits = (sign << 31) | (0xff << 23) | (mant << 13);
  } else {
    f32Bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  final bd = ByteData(4)..setUint32(0, f32Bits, Endian.little);
  return bd.getFloat32(0, Endian.little);
}

/// Parses an in-memory safetensors blob and returns the records.
Map<String, SafeTensorRecord> parseSafetensors(Uint8List blob) {
  if (blob.length < 8) {
    throw const FormatException('safetensors: file shorter than 8-byte header');
  }
  final headerLen = ByteData.sublistView(
    blob,
    0,
    8,
  ).getUint64(0, Endian.little);
  if (8 + headerLen > blob.length) {
    throw FormatException(
      'safetensors: header length $headerLen exceeds file size',
    );
  }
  final headerJson = utf8.decode(blob.sublist(8, 8 + headerLen));
  final Map<String, dynamic> meta =
      jsonDecode(headerJson) as Map<String, dynamic>;
  final dataStart = 8 + headerLen;
  final out = <String, SafeTensorRecord>{};
  for (final entry in meta.entries) {
    if (entry.key == '__metadata__') continue;
    final m = entry.value as Map<String, dynamic>;
    final dtype = m['dtype'] as String;
    final shape = (m['shape'] as List).map((e) => e as int).toList();
    final offs = (m['data_offsets'] as List).map((e) => e as int).toList();
    final start = dataStart + offs[0];
    final end = dataStart + offs[1];
    if (end > blob.length) {
      throw FormatException(
        'safetensors: tensor "${entry.key}" data_offsets out of bounds',
      );
    }
    out[entry.key] = SafeTensorRecord(
      name: entry.key,
      dtype: dtype,
      shape: shape,
      bytes: Uint8List.sublistView(blob, start, end),
    );
  }
  return out;
}

/// Convenience wrapper: read a `.safetensors` file from disk.
Future<Map<String, SafeTensorRecord>> loadSafetensorsFile(String path) async {
  final bytes = await File(path).readAsBytes();
  return parseSafetensors(bytes);
}
