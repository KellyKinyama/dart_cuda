import 'dart:math' as math;
import 'adam.dart';
import 'aft_multi_head_attention.dart';
import 'aft_multi_head_cross_attention.dart';
import 'feed_forward.dart';
import 'gpu_tensor.dart';
import 'layer_norm.dart';
import 'nn.dart';

class TransformerDecoderBlock extends Module {
  final MultiHeadAFT selfAttention;
  final MultiHeadAFTCross crossAttention;
  final FeedForward ffn;
  final LayerNorm ln1, ln2, ln3;
  final int embedSize;

  TransformerDecoderBlock(
    this.embedSize,
    int numHeads,
    int encoderEmbedSize,
    int maxSeqLen,
  ) : selfAttention = MultiHeadAFT(
        numHeads,
        embedSize,
        maxSeqLen,
        masked: true,
      ),
      crossAttention = MultiHeadAFTCross(
        numHeads,
        embedSize,
        encoderEmbedSize,
        maxSeqLen,
        maxSeqLen,
      ),
      ffn = FeedForward(embedSize),
      ln1 = LayerNorm(embedSize),
      ln2 = LayerNorm(embedSize),
      ln3 = LayerNorm(embedSize) {
    _initializeWeights();
  }

  /// Xavier initialization scaled down for AFT stability
  void _initializeWeights() {
    final rand = math.Random();
    for (var p in parameters()) {
      final data = p.fetchData();
      for (int i = 0; i < data.length; i++) {
        // Using a very small range helps the AFT exponentials start in a safe zone
        data[i] = (rand.nextDouble() * 2 - 1) * 0.02;
      }
      p.data = data;
    }
    // print("🎯 Weights re-initialized to stable range [-0.02, 0.02]");
  }

  Tensor forward(Tensor xDec, Tensor xEnc, List<Tensor> tracker) {
    // --- 1. Masked Self-Attention (Pre-LN) ---
    final xNorm1 = ln1.forward(xDec, tracker);
    final selfAttnOut = selfAttention.forward(xNorm1, tracker);
    final xRes1 = xDec + selfAttnOut;
    tracker.add(xRes1);

    // --- 2. Cross-Attention (Pre-LN) ---
    final xNorm2 = ln2.forward(xRes1, tracker);
    final crossAttnOut = crossAttention.forward(xNorm2, xEnc, tracker);
    final xRes2 = xRes1 + crossAttnOut;
    tracker.add(xRes2);

    // --- 3. Feed-Forward (Expansion/Contraction) ---
    final xNorm3 = ln3.forward(xRes2, tracker);
    final ffnOut = ffn.forward(xNorm3, tracker);
    final out = xRes2 + ffnOut;

    // Final tracker addition for the block output
    tracker.add(out);

    _debugStats(out); // Monitor for explosions
    return out;
  }

  void _debugStats(Tensor t) {
    final data = t.fetchData();
    if (data.isEmpty) return;
    double maxAbs = 0;
    for (var v in data) {
      if (v.abs() > maxAbs) maxAbs = v.abs();
    }
    // if (maxAbs > 5.0) {
    //   print(
    //     '  🚨 [STABILITY ALERT] Block Output MaxAbs: ${maxAbs.toStringAsFixed(4)}',
    //   );
    // }
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttention.parameters(),
    ...crossAttention.parameters(),
    ...ffn.parameters(),
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...ln3.parameters(),
  ];
}

void main() {
  const int embedSize = 32;
  const int numHeads = 4;
  const int encoderEmbedSize = 64;
  const int maxSeqLen = 50;

  // Use a very conservative LR for the first few steps of AFT
  const double lr = 0.0001;
  const double gradClip = 0.1;

  final decoderBlock = TransformerDecoderBlock(
    embedSize,
    numHeads,
    encoderEmbedSize,
    maxSeqLen,
  );

  final xDec = Tensor.random([8, embedSize]);
  final xEnc = Tensor.random([12, encoderEmbedSize]);
  final target = Tensor.fill([8, embedSize], 0.5);

  final List<Tensor> tracker = [];
  final optimizer = Adam(decoderBlock.parameters(), lr: lr, gradClip: gradClip);

  print('--- Training Corrected TransformerDecoderBlock ---');

  for (int epoch = 0; epoch < 50; epoch++) {
    optimizer.zeroGrad();

    final output = decoderBlock.forward(xDec, xEnc, tracker);
    final loss = output.mseLoss(target);

    if (epoch % 5 == 0) {
      print(
        'Epoch ${epoch.toString().padLeft(2)} | Loss: ${loss.data[0].toStringAsFixed(8)}',
      );
    }

    if (loss.data[0].isNaN) {
      print("❌ NaN detected at Epoch $epoch. Shutting down.");
      break;
    }

    loss.backward();
    optimizer.step();

    // Memory Cleanup
    for (var t in tracker) t.dispose();
    tracker.clear();
    loss.dispose();
    output.dispose();
  }
}
