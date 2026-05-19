// Tests for the safetensors reader + PyTorch port helper.
//
// We don't depend on a Python toolchain in CI, so the test builds the
// safetensors blob by hand from raw bytes — this is also the best
// possible regression test for the parser itself.

import 'dart:convert';
import 'dart:typed_data';

import 'package:dart_cuda/core/loaders/pytorch_port.dart';
import 'package:dart_cuda/core/loaders/safetensors.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

/// Helper: build a safetensors blob for a single F32 tensor named [name].
Uint8List _makeSimpleBlob(String name, List<int> shape, List<double> values) {
  final data = Float32List.fromList(values);
  final dataBytes = data.buffer.asUint8List();
  final header = jsonEncode({
    name: {
      'dtype': 'F32',
      'shape': shape,
      'data_offsets': [0, dataBytes.length],
    },
  });
  final headerBytes = utf8.encode(header);
  // 8-byte little-endian length + header + raw data
  final out = BytesBuilder();
  final lenBuf = ByteData(8)..setUint64(0, headerBytes.length, Endian.little);
  out.add(lenBuf.buffer.asUint8List());
  out.add(headerBytes);
  out.add(dataBytes);
  return out.toBytes();
}

/// Builds a blob with two F32 tensors: a [2,3] matrix and a [3] bias.
Uint8List _makeTwoTensorBlob() {
  final w = Float32List.fromList([1, 2, 3, 4, 5, 6]); // 2x3, row-major
  final b = Float32List.fromList([10, 20, 30]);
  final wBytes = w.buffer.asUint8List();
  final bBytes = b.buffer.asUint8List();
  final header = jsonEncode({
    'w': {
      'dtype': 'F32',
      'shape': [2, 3],
      'data_offsets': [0, wBytes.length],
    },
    'b': {
      'dtype': 'F32',
      'shape': [3],
      'data_offsets': [wBytes.length, wBytes.length + bBytes.length],
    },
  });
  final headerBytes = utf8.encode(header);
  final out = BytesBuilder();
  final lenBuf = ByteData(8)..setUint64(0, headerBytes.length, Endian.little);
  out.add(lenBuf.buffer.asUint8List());
  out.add(headerBytes);
  out.add(wBytes);
  out.add(bBytes);
  return out.toBytes();
}

void main() {
  group('safetensors parser', () {
    test('reads a single F32 tensor with correct shape & values', () {
      final blob = _makeSimpleBlob('foo', [2, 2], [1.5, 2.5, 3.5, 4.5]);
      final recs = parseSafetensors(blob);
      expect(recs.keys, equals(['foo']));
      final r = recs['foo']!;
      expect(r.dtype, equals('F32'));
      expect(r.shape, equals([2, 2]));
      expect(r.numel, equals(4));
      expect(r.asFloat32().toList(), equals([1.5, 2.5, 3.5, 4.5]));
    });

    test('reads multiple tensors with correct slicing of data section', () {
      final blob = _makeTwoTensorBlob();
      final recs = parseSafetensors(blob);
      expect(recs.keys.toSet(), equals({'w', 'b'}));
      expect(recs['w']!.asFloat32().toList(), equals([1, 2, 3, 4, 5, 6]));
      expect(recs['b']!.asFloat32().toList(), equals([10, 20, 30]));
    });

    test('skips __metadata__ sentinel', () {
      final inner = {
        '__metadata__': {'format': 'pt'},
        'x': {
          'dtype': 'F32',
          'shape': [1],
          'data_offsets': [0, 4],
        },
      };
      final headerBytes = utf8.encode(jsonEncode(inner));
      final data = (Float32List.fromList([7.5])).buffer.asUint8List();
      final out = BytesBuilder();
      final lenBuf = ByteData(8)
        ..setUint64(0, headerBytes.length, Endian.little);
      out.add(lenBuf.buffer.asUint8List());
      out.add(headerBytes);
      out.add(data);
      final recs = parseSafetensors(out.toBytes());
      expect(recs.keys, equals(['x']));
      expect(recs['x']!.asFloat32()[0], closeTo(7.5, 1e-6));
    });

    test('rejects unsupported dtype with helpful message', () {
      final headerBytes = utf8.encode(
        jsonEncode({
          'x': {
            'dtype': 'I64',
            'shape': [2],
            'data_offsets': [0, 16],
          },
        }),
      );
      final out = BytesBuilder();
      final lenBuf = ByteData(8)
        ..setUint64(0, headerBytes.length, Endian.little);
      out.add(lenBuf.buffer.asUint8List());
      out.add(headerBytes);
      out.add(Uint8List(16));
      final recs = parseSafetensors(out.toBytes());
      expect(
        () => recs['x']!.asFloat32(),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains('Cast to float32'),
          ),
        ),
      );
    });
  });

  group('PyTorch port', () {
    test('copies plain (non-transposed) tensors into targets', () {
      final blob = _makeTwoTensorBlob();
      final recs = parseSafetensors(blob);
      final w = Tensor.fromList([2, 3], List.filled(6, 0.0));
      final b = Tensor.fromList([1, 3], [0, 0, 0]);
      addTearDown(() {
        w.dispose();
        b.dispose();
      });
      final warnings = loadPytorchInto(recs, {
        'w': PortSpec(w),
        'b': PortSpec(b),
      });
      expect(warnings, isEmpty);
      expect(w.data, equals([1, 2, 3, 4, 5, 6]));
      expect(b.data, equals([10, 20, 30]));
    });

    test('transposes nn.Linear-style weights from [out,in] to [in,out]', () {
      // PyTorch nn.Linear(in=3, out=2).weight has shape [2, 3]:
      //   [[w00, w01, w02],
      //    [w10, w11, w12]]
      // Our matmul uses x @ w with w of shape [in, out] = [3, 2]:
      //   [[w00, w10],
      //    [w01, w11],
      //    [w02, w12]]
      final blob = _makeSimpleBlob(
        'linear.weight',
        [2, 3],
        [
          // row 0: [1, 2, 3]; row 1: [4, 5, 6]
          1, 2, 3, 4, 5, 6,
        ],
      );
      final recs = parseSafetensors(blob);
      final ourW = Tensor.fromList([3, 2], List.filled(6, 0.0));
      addTearDown(ourW.dispose);
      loadPytorchInto(recs, {'linear.weight': PortSpec(ourW, transpose: true)});
      // Expected row-major flat: [1,4, 2,5, 3,6]
      expect(ourW.data, equals([1, 4, 2, 5, 3, 6]));
    });

    test('reports missing keys with all errors in one exception', () {
      final blob = _makeSimpleBlob('exists', [1], [1.0]);
      final recs = parseSafetensors(blob);
      final a = Tensor.fromList([1, 1], [0]);
      final b = Tensor.fromList([1, 1], [0]);
      addTearDown(() {
        a.dispose();
        b.dispose();
      });
      expect(
        () => loadPytorchInto(recs, {
          'missing_a': PortSpec(a),
          'missing_b': PortSpec(b),
        }),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            allOf(contains('missing_a'), contains('missing_b')),
          ),
        ),
      );
    });

    test('flags shape mismatch (with post-transpose shape in message)', () {
      // Source [2, 3]; with transpose → [3, 2]; target expects [4, 2].
      final blob = _makeSimpleBlob('w', [2, 3], List.filled(6, 0.0));
      final recs = parseSafetensors(blob);
      final bad = Tensor.fromList([4, 2], List.filled(8, 0.0));
      addTearDown(bad.dispose);
      expect(
        () => loadPytorchInto(recs, {'w': PortSpec(bad, transpose: true)}),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            allOf(contains('shape mismatch'), contains('transposed→[3, 2]')),
          ),
        ),
      );
    });

    test('auto-broadcasts rank-1 source [N] into rank-2 target [1, N] '
        '(PyTorch nn.Linear.bias convention)', () {
      final blob = _makeSimpleBlob('bias', [3], [10, 20, 30]);
      final recs = parseSafetensors(blob);
      final b = Tensor.fromList([1, 3], [0, 0, 0]);
      addTearDown(b.dispose);
      final warnings = loadPytorchInto(recs, {'bias': PortSpec(b)});
      expect(warnings, isEmpty);
      expect(b.data, equals([10, 20, 30]));
    });

    test('returns warnings for unused keys when non-strict', () {
      final w = Float32List.fromList([1, 2]);
      final wb = w.buffer.asUint8List();
      final headerBytes = utf8.encode(
        jsonEncode({
          'used': {
            'dtype': 'F32',
            'shape': [2],
            'data_offsets': [0, wb.length],
          },
          'leftover': {
            'dtype': 'F32',
            'shape': [2],
            'data_offsets': [wb.length, wb.length * 2],
          },
        }),
      );
      final out = BytesBuilder();
      final lenBuf = ByteData(8)
        ..setUint64(0, headerBytes.length, Endian.little);
      out.add(lenBuf.buffer.asUint8List());
      out.add(headerBytes);
      out.add(wb);
      out.add(wb);
      final recs = parseSafetensors(out.toBytes());

      final t = Tensor.fromList([1, 2], [0, 0]);
      addTearDown(t.dispose);
      final warnings = loadPytorchInto(recs, {'used': PortSpec(t)});
      expect(warnings, equals(['unused safetensors key: "leftover"']));
      // Strict mode should throw.
      expect(
        () => loadPytorchInto(recs, {'used': PortSpec(t)}, strict: true),
        throwsA(isA<StateError>()),
      );
    });
  });
}
