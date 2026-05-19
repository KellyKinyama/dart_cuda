// Runs a PyTorch-trained MLP on the GPU in Dart, end to end.
//
// This is the production wiring for the `safetensors -> port -> GPU forward`
// pipeline exercised by `test/core/loaders/pytorch_port_e2e_test.dart`.
//
// Workflow:
//   1. Export your PyTorch checkpoint to safetensors:
//        python3 python/tools/export_to_safetensors.py \
//            -i /path/to/model.pt -o /path/to/model.safetensors --list
//      The `--list` flag prints every tensor's name + shape + dtype, which
//      is what you wire into the `mapping` below.
//
//   2. Define an equivalent `Layer` stack here (sizes and activation must
//      match your PyTorch model exactly).
//
//   3. Run:
//        dart run example/bin/run_pytorch_checkpoint.dart \
//            <model.safetensors> [<input.json>]
//      where `<input.json>` is optionally a JSON list of floats. Without
//      it the example uses a zero-vector input so you can sanity-check
//      shapes and the bias-only output.
//
// The mapping below assumes a 2-layer MLP defined in PyTorch as:
//
//   import torch.nn as nn
//   class Net(nn.Module):
//       def __init__(self):
//           super().__init__()
//           self.fc1 = nn.Linear(IN_DIM,  HIDDEN_DIM)
//           self.fc2 = nn.Linear(HIDDEN_DIM, OUT_DIM)
//       def forward(self, x):
//           return self.fc2(torch.tanh(self.fc1(x)))
//
// Edit the `IN_DIM` / `HIDDEN_DIM` / `OUT_DIM` constants below to match
// your actual model, and rename the safetensors keys in the mapping if
// your module names differ (`encoder.0.weight` etc.).

import 'dart:convert';
import 'dart:io';

import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/loaders/pytorch_port.dart';
import 'package:dart_cuda/core/loaders/safetensors.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

// -------- Edit these to match your PyTorch model. ---------------------------
const int kInDim = 3;
const int kHiddenDim = 4;
const int kOutDim = 2;
// ----------------------------------------------------------------------------

Future<void> main(List<String> argv) async {
  if (argv.isEmpty) {
    stderr.writeln(
      'usage: dart run example/bin/run_pytorch_checkpoint.dart '
      '<weights.safetensors> [<input.json>]',
    );
    exit(64);
  }

  final weightsPath = argv[0];
  final records = await loadSafetensorsFile(weightsPath);
  stdout.writeln('Loaded ${records.length} tensors from $weightsPath:');
  for (final e in records.entries) {
    stdout.writeln(
      '  ${e.key}  dtype=${e.value.dtype}  shape=${e.value.shape}',
    );
  }

  // Build the Dart model.
  final l1 = Layer(kInDim, kHiddenDim, useGelu: false);
  final l2 = Layer(kHiddenDim, kOutDim, useGelu: false);

  // Port PyTorch weights in. PyTorch `nn.Linear` stores its weight as
  // [out, in] and bias as [out]; our `Layer` stores w as [in, out] and
  // b as [1, out]. `transpose: true` handles the weight rearrangement;
  // the bias rank-1 -> rank-2 case is handled automatically.
  final warnings = loadPytorchInto(records, {
    'fc1.weight': PortSpec(l1.w, transpose: true),
    'fc1.bias': PortSpec(l1.b),
    'fc2.weight': PortSpec(l2.w, transpose: true),
    'fc2.bias': PortSpec(l2.b),
  });
  for (final w in warnings) {
    stderr.writeln('warning: $w');
  }

  // Build an input tensor.
  List<double> xData;
  if (argv.length >= 2) {
    final raw = await File(argv[1]).readAsString();
    final parsed = jsonDecode(raw);
    if (parsed is! List) {
      stderr.writeln('input json must be a list of numbers');
      exit(65);
    }
    xData = parsed.cast<num>().map((n) => n.toDouble()).toList();
    if (xData.length != kInDim) {
      stderr.writeln('input length ${xData.length} != expected kInDim=$kInDim');
      exit(65);
    }
  } else {
    xData = List<double>.filled(kInDim, 0.0);
  }
  final xT = Tensor.fromList([1, kInDim], xData);

  // Forward.
  final tracker = <Tensor>[];
  final h1 = l1.forward(xT, tracker);
  final h1Act = h1.tanh();
  final yT = l2.forward(h1Act, tracker);

  stdout.writeln('input:  $xData');
  stdout.writeln('output: ${yT.data}');

  // Clean up GPU resources.
  for (final t in tracker) {
    t.dispose();
  }
  h1Act.dispose();
  yT.dispose();
  xT.dispose();
  l1.w.dispose();
  l1.b.dispose();
  l2.w.dispose();
  l2.b.dispose();
}
