import 'adam.dart';
import 'aft_cross_attention.dart';
import 'gpu_tensor.dart';
import 'nn.dart';

class MultiHeadAFTCross extends Module {
  final List<AFTCrossAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;

  MultiHeadAFTCross(
    this.numHeads,
    int decoderEmbedSize,
    int encoderEmbedSize,
    int maxTDec,
    int maxTEnc,
  ) : assert(decoderEmbedSize % numHeads == 0),
      headSize = decoderEmbedSize ~/ numHeads,
      heads = List.generate(
        numHeads,
        (i) => AFTCrossAttention(
          decoderEmbedSize,
          encoderEmbedSize,
          decoderEmbedSize ~/ numHeads,
          maxTDec,
          maxTEnc,
        ),
      ),
      proj = Layer(decoderEmbedSize, decoderEmbedSize, useGelu: false);

  /// xDec: [T_dec, decoderEmbedSize]
  /// xEnc: [T_enc, encoderEmbedSize]
  Tensor forward(Tensor xDec, Tensor xEnc, List<Tensor> tracker) {
    // 1. Run each Cross-Attention head on GPU
    // Each head returns a [TDec, headSize] tensor handle
    final headOutputs = heads
        .map((h) => h.forward(xDec, xEnc, tracker))
        .toList();

    // 2. Concatenate heads along axis 1 (GPU kernel)
    // Uses the concat_tensors_gpu kernel we added to engine.cu
    final concatenated = Tensor.concat(headOutputs);
    tracker.add(concatenated);

    // 3. Final linear projection
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
  const int numHeads = 4;
  const int decoderEmbed = 32; // The dimension the decoder is working with
  const int encoderEmbed = 64; // The dimension of the encoder's output
  const int maxSeqLen = 50; // Max capacity for position bias
  const double lr = 0.01;

  // 2. Initialize Module
  // Note: encoderEmbedSize can be different from embedSize
  final aftCross = MultiHeadAFTCross(
    numHeads,
    decoderEmbed,
    encoderEmbed,
    maxSeqLen,
    50,
  );

  // 3. Prepare Dummy Tensors
  // Decoder sequence (e.g., words already translated)
  final xDec = Tensor.random([5, decoderEmbed]);

  // Encoder sequence (e.g., the full source sentence being translated)
  final xEnc = Tensor.random([12, encoderEmbed]);

  // Target output for this step [T_dec, decoderEmbed]
  final target = Tensor.fill([5, decoderEmbed], 0.1);

  final List<Tensor> tracker = [];
  final optimizer = Adam(aftCross.parameters(), lr: lr);

  print('--- MultiHeadAFTCross Training Step ---');
  for (int x = 0; x < 20; x++) {
    optimizer.zeroGrad();
    // 4. Forward Pass
    // The decoder queries the encoder
    final output = aftCross.forward(xDec, xEnc, tracker);
    print('Output shape: ${output.shape}'); // Expected: [5, 32]

    // 5. Compute Loss
    final loss = output.mseLoss(target);
    print('Initial Cross-Attention Loss: ${loss.data[0].toStringAsFixed(6)}');

    // 6. Backward Pass
    // This backpropagates through the decoder projections AND encoder projections
    loss.backward();

    // 7. Manual SGD Update
    optimizer.step();
    optimizer.zeroGrad();
    // 8. Verify
    final nextLoss = aftCross.forward(xDec, xEnc, tracker).mseLoss(target);
    print(
      'Loss after Cross-Attention update: ${nextLoss.data[0].toStringAsFixed(6)}',
    );

    if (nextLoss.data[0] < loss.data[0]) {
      print('Success: Gradients flowed and loss decreased!');
    }
  }

  for (var track in tracker) {
    track.dispose();
  }
}


