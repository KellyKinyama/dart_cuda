// End-to-end test that proves a PyTorch-trained checkpoint runs in Dart.
//
// Strategy: we hand-craft an in-memory `safetensors` blob using the
// EXACT layout `safetensors.torch.save_file(model.state_dict())` would
// produce for a 2-layer MLP with `nn.Linear` (i.e. weights stored as
// `[out, in]`). Then we:
//
//   1. Parse it with the Dart safetensors reader.
//   2. Build the same architecture using our `Layer` class
//      (weights stored as `[in, out]`).
//   3. Load with `loadPytorchInto`, requesting `transpose: true` on the
//      .weight entries.
//   4. Run forward on a fixed input on the GPU.
//   5. Independently compute the expected output by doing the matmul +
//      bias in plain Dart with the same source weights — that is the
//      ground-truth PyTorch would produce.
//
// If the GPU result matches the CPU/PyTorch math, the entire
// `safetensors → name map → transpose → set_tensor_data → GPU matmul`
// pipeline is verified end-to-end without needing torch installed.

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/loaders/pytorch_port.dart';
import 'package:dart_cuda/core/loaders/safetensors.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:test/test.dart';

/// Build a safetensors blob for the keys / shapes / values supplied,
/// matching the byte layout that `safetensors.torch.save_file` writes.
Uint8List _buildBlob(Map<String, ({List<int> shape, List<double> values})> t) {
  // Serialize header with byte offsets pre-computed.
  final entries = <String, Map<String, Object>>{};
  final dataChunks = <Uint8List>[];
  var cursor = 0;
  for (final e in t.entries) {
    final f32 = Float32List.fromList(e.value.values);
    final bytes = f32.buffer.asUint8List();
    entries[e.key] = {
      'dtype': 'F32',
      'shape': e.value.shape,
      'data_offsets': [cursor, cursor + bytes.length],
    };
    dataChunks.add(bytes);
    cursor += bytes.length;
  }
  final headerBytes = utf8.encode(jsonEncode(entries));
  final out = BytesBuilder();
  final lenBuf = ByteData(8)..setUint64(0, headerBytes.length, Endian.little);
  out.add(lenBuf.buffer.asUint8List());
  out.add(headerBytes);
  for (final c in dataChunks) {
    out.add(c);
  }
  return out.toBytes();
}

/// CPU reference for `x @ W^T + b` (i.e. PyTorch nn.Linear semantics).
/// w has shape [out, in], b has shape [out]. Returns [B, out] row-major.
List<double> _linearRef({
  required List<double> x, // [B, in], row-major
  required int batch,
  required int inDim,
  required int outDim,
  required List<double> w, // [out, in], row-major
  required List<double> b, // [out]
}) {
  final out = List<double>.filled(batch * outDim, 0);
  for (var i = 0; i < batch; i++) {
    for (var o = 0; o < outDim; o++) {
      var sum = b[o];
      for (var k = 0; k < inDim; k++) {
        sum += x[i * inDim + k] * w[o * inDim + k];
      }
      out[i * outDim + o] = sum;
    }
  }
  return out;
}

List<double> _tanhRef(List<double> v) => [for (final x in v) _tanh(x)];

double _tanh(double x) {
  // dart:math has no tanh; use the identity (e^x - e^-x) / (e^x + e^-x).
  final ex = math.exp(x);
  final emx = math.exp(-x);
  return (ex - emx) / (ex + emx);
}

void main() {
  test(
    'end-to-end: PyTorch nn.Linear weights → Dart Layer → matching output',
    () {
      // ---- 1. Define a 2-layer MLP in PyTorch convention. ------------------
      //
      //   input  : [B=2, 3]
      //   fc1    : nn.Linear(in=3, out=4)   → weight [4, 3], bias [4]
      //   tanh
      //   fc2    : nn.Linear(in=4, out=2)   → weight [2, 4], bias [2]
      //   output : [2, 2]
      //
      // The numbers below are arbitrary but deterministic — they stand in
      // for what `torch.save(model.state_dict(), ...)` would emit.

      final fc1W = <double>[
        // 4 rows x 3 cols = [out=4, in=3]
        0.1, 0.2, -0.3,
        -0.4, 0.5, 0.6,
        0.7, -0.8, 0.9,
        -1.0, 0.0, 0.25,
      ];
      final fc1B = <double>[0.01, -0.02, 0.03, -0.04];
      final fc2W = <double>[
        // 2 rows x 4 cols = [out=2, in=4]
        0.5, -0.5, 0.25, -0.25,
        -0.1, 0.1, -0.2, 0.2,
      ];
      final fc2B = <double>[0.0, 0.0];

      final x = <double>[
        // [B=2, in=3] — batch>1 exercises the row-broadcast bias add
        // (`[1, out] + [B, out]`) which is handled by a dedicated CUDA
        // kernel `add_tensor_row_broadcast`.
        1.0, 2.0, 3.0,
        -1.0, 0.5, 0.0,
      ];

      // ---- 2. Build the safetensors blob (PyTorch on-disk format). ---------
      final blob = _buildBlob({
        'fc1.weight': (shape: [4, 3], values: fc1W),
        'fc1.bias': (shape: [4], values: fc1B),
        'fc2.weight': (shape: [2, 4], values: fc2W),
        'fc2.bias': (shape: [2], values: fc2B),
      });
      final records = parseSafetensors(blob);

      // ---- 3. Build the Dart model and port weights in. --------------------
      final l1 = Layer(3, 4, useGelu: false);
      final l2 = Layer(4, 2, useGelu: false);
      addTearDown(() {
        l1.w.dispose();
        l1.b.dispose();
        l2.w.dispose();
        l2.b.dispose();
      });
      final warnings = loadPytorchInto(records, {
        'fc1.weight': PortSpec(l1.w, transpose: true), // [4,3] → [3,4]
        'fc1.bias': PortSpec(l1.b), // [4] auto-broadcasts to our [1,4]
        'fc2.weight': PortSpec(l2.w, transpose: true), // [2,4] → [4,2]
        'fc2.bias': PortSpec(l2.b),
      }, strict: true);
      expect(warnings, isEmpty);

      // Sanity: bias values landed in the [1, out] targets. We compare
      // element-wise with a tiny tolerance because the safetensors blob
      // stores f32 and the literals above are f64.
      expect(l1.b.shape, equals([1, 4]));
      for (var i = 0; i < fc1B.length; i++) {
        expect(l1.b.data[i], closeTo(fc1B[i], 1e-6));
      }
      for (var i = 0; i < fc2B.length; i++) {
        expect(l2.b.data[i], closeTo(fc2B[i], 1e-6));
      }

      // ---- 4. Forward pass on GPU. -----------------------------------------
      final xT = Tensor.fromList([2, 3], x);
      final tracker = <Tensor>[];
      final h1 = l1.forward(xT, tracker);
      final h1Act = h1.tanh();
      final yT = l2.forward(h1Act, tracker);
      addTearDown(() {
        xT.dispose();
        for (final t in tracker) {
          t.dispose();
        }
        h1Act.dispose();
        yT.dispose();
      });

      final gpuOut = yT.data;

      // ---- 5. Reference PyTorch math on CPU. -------------------------------
      final h1Ref = _linearRef(
        x: x,
        batch: 2,
        inDim: 3,
        outDim: 4,
        w: fc1W,
        b: fc1B,
      );
      final h1ActRef = _tanhRef(h1Ref);
      final yRef = _linearRef(
        x: h1ActRef,
        batch: 2,
        inDim: 4,
        outDim: 2,
        w: fc2W,
        b: fc2B,
      );

      // ---- 6. Compare. -----------------------------------------------------
      expect(yT.shape, equals([2, 2]));
      expect(gpuOut.length, equals(yRef.length));
      for (var i = 0; i < gpuOut.length; i++) {
        expect(
          gpuOut[i],
          closeTo(yRef[i], 1e-5),
          reason:
              'index $i: gpu=${gpuOut[i]} ref=${yRef[i]} '
              '(this means the safetensors→port→GPU pipeline diverged)',
        );
      }
    },
  );
}
