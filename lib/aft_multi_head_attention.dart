import 'package:dart_cuda/adam.dart';

import 'aft.dart';
import 'gpu_tensor.dart';
import 'nn.dart';

class MultiHeadAFT extends Module {
  final List<AFTAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;
  final bool masked;

  MultiHeadAFT(
    this.numHeads,
    int embedSize,
    int maxSeqLen, {
    this.masked = false,
  }) : assert(embedSize % numHeads == 0),
       headSize = embedSize ~/ numHeads,
       heads = List.generate(
         numHeads,
         (i) => AFTAttention(
           embedSize,
           embedSize ~/ numHeads,
           maxSeqLen,
           masked: masked,
         ),
       ),
       proj = Layer(embedSize, embedSize, useGelu: false);

  Tensor forward(Tensor x, List<Tensor> tracker) {
    // 1. Compute each head on GPU
    final List<Tensor> headOutputs = heads
        .map((h) => h.forward(x, tracker))
        .toList();

    // 2. Concatenate results on GPU
    final concatenated = Tensor.concat(headOutputs);
    tracker.add(concatenated);

    // 3. Final Linear Projection
    return proj.forward(concatenated, tracker);
  }

  @override
  List<Tensor> parameters() => [
    ...heads.expand((h) => h.parameters()),
    ...proj.parameters(),
  ];
}

void main() {
  // 1. Hyperparameters
  const int numHeads =
      4; // Note: The logic provided processes embedSize directly
  const int embedSize = 16; // Total embedding dimension
  const int maxSeqLen = 20; // Capacity of the position bias matrix
  const int currentT = 8; // Actual sequence length for this batch
  const double lr = 0.01; // Learning rate

  // 2. Initialize Module (with Causal Masking enabled)
  // Masked: true is typical for Autoregressive tasks (like GPT)
  final aft = MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: true);

  // 3. Create Dummy Data [T, Dim]
  final input = Tensor.random([currentT, embedSize]);

  // Target: We want the model to predict a specific pattern (e.g., all 0.5s)
  final target = Tensor.fill([currentT, embedSize], 0.5);

  List<Tensor> tracker = [];

  final optimizer = Adam(aft.parameters(), lr: lr);

  print('--- MultiHeadAFT Training Step (Causal) ---');
  for (int x = 0; x < 20; x++) {
    optimizer.zeroGrad();
    // 4. Forward Pass
    // The module internally handles the slicing of posBias and the causal loop
    final output = aft.forward(input, tracker);
    print('Output shape: ${output.shape}');

    // 5. Compute Loss
    final diff = output - target;
    final loss = diff.pow(2.0);
    // final loss = output.mseLoss(target);
    print('Initial Loss: ${loss.data[0].toStringAsFixed(6)}');

    // 6. Backward Pass
    // This triggers the custom onBackward blocks for extraction,
    // the AFT logic loop, and the linear projections.
    loss.backward();

    // 7. Optimizer Step (Manual SGD)
    optimizer.step();
    // print('Updated $paramCount parameter tensors.');
    optimizer.zeroGrad();
    // 8. Verify Progress
    final nextOutput = aft.forward(input, tracker);

    final nextDiff = nextOutput - target;
    final nextLoss = nextDiff.pow(2.0);
    // final nextLoss = nextOutput.mseLoss(target);
    print('Loss after 1 step: ${nextLoss.data[0].toStringAsFixed(6)}');

    if (nextLoss.data[0] < loss.data[0]) {
      print('Success: Gradients flowed and loss decreased!');
    }
  }

  for (var track in tracker) {
    track.dispose();
  }
}
