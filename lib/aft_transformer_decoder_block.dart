import 'dart:math' as math;
import 'adam.dart';
import 'aft_multi_head_attention.dart';
import 'aft_multi_head_cross_attention.dart';
import 'gpu_tensor.dart';
import 'layer_norm.dart';
import 'nn.dart';

class TransformerDecoderBlock extends Module {
  final MultiHeadAFT selfAttention;
  final MultiHeadAFTCross crossAttention;
  final Layer ffn;
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
      ffn = Layer(embedSize, embedSize, useGelu: true),
      ln1 = LayerNorm(embedSize),
      ln2 = LayerNorm(embedSize),
      ln3 = LayerNorm(embedSize) {
    _initializeWeights();
  }

  /// Force weights to small range to prevent exponential overflow in AFT
  void _initializeWeights() {
    final rand = math.Random();
    for (var p in parameters()) {
      final data = p.fetchData();
      for (int i = 0; i < data.length; i++) {
        data[i] = (rand.nextDouble() * 2 - 1) * 0.02;
      }
      p.data = data;
    }
    print("🎯 Weights initialized to stable range [-0.02, 0.02]");
  }

  Tensor forward(Tensor xDec, Tensor xEnc, List<Tensor> tracker) {
    // --- 1. Masked Self-Attention ---
    final xNorm1 = ln1.forward(xDec, tracker);
    _debugTensor("Post-LN1", xNorm1);

    final selfAttnOut = selfAttention.forward(xNorm1, tracker);
    _debugTensor("Self-Attn Out", selfAttnOut);

    final xRes1 = xDec + selfAttnOut;
    tracker.add(xRes1);

    // --- 2. Cross-Attention ---
    final xNorm2 = ln2.forward(xRes1, tracker);
    _debugTensor("Post-LN2", xNorm2);

    final crossAttnOut = crossAttention.forward(xNorm2, xEnc, tracker);
    _debugTensor("Cross-Attn Out", crossAttnOut);

    final xRes2 = xRes1 + crossAttnOut;
    tracker.add(xRes2);

    // --- 3. Feed-Forward ---
    final xNorm3 = ln3.forward(xRes2, tracker);
    _debugTensor("Post-LN3", xNorm3);

    final ffnOut = ffn.forward(xNorm3, tracker);
    _debugTensor("FFN Out", ffnOut);

    final out = xRes2 + ffnOut;
    _debugTensor("Block Final Out", out);

    return out;
  }

  void _debugTensor(String name, Tensor t) {
    final data = t.fetchData();
    if (data.isEmpty) return;
    double maxVal = 0;
    bool hasNaN = false;
    for (var v in data) {
      if (v.isNaN || v.isInfinite) hasNaN = true;
      if (v.abs() > maxVal) maxVal = v.abs();
    }
    if (hasNaN || maxVal > 10.0) {
      print(
        '  🚨 [DEBUG $name] MaxAbs: ${maxVal.toStringAsFixed(4)} ${hasNaN ? "!! NAN/INF !!" : "!! HIGH VARIANCE !!"}',
      );
    }
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
  const double lr = 0.0005;
  const double gradClip = 0.1; // Strict clipping for debug phase

  final decoderBlock = TransformerDecoderBlock(
    embedSize,
    numHeads,
    encoderEmbedSize,
    maxSeqLen,
  );

  // Use slightly smaller random inputs to be safe
  final xDec = Tensor.random([8, embedSize]);
  final xEnc = Tensor.random([15, encoderEmbedSize]);
  final target = Tensor.fill([8, embedSize], 0.1);

  final List<Tensor> tracker = [];
  final optimizer = Adam(decoderBlock.parameters(), lr: lr, gradClip: gradClip);

  print('--- TransformerDecoderBlock Full Debug Training ---');

  try {
    for (int epoch = 0; epoch < 20; epoch++) {
      optimizer.zeroGrad();

      final output = decoderBlock.forward(xDec, xEnc, tracker);
      final loss = output.mseLoss(target);
      final double lVal = loss.data[0];

      print(
        'Epoch ${epoch.toString().padLeft(2)} | Loss: ${lVal.toStringAsFixed(6)}',
      );

      if (lVal.isNaN) break;

      loss.backward();

      // Monitor Gradient Magnitudes
      _checkGradients(decoderBlock.parameters());

      optimizer.step();

      for (var t in tracker) t.dispose();
      tracker.clear();
      loss.dispose();
      output.dispose();
    }
  } catch (e) {
    print("❌ Crash: $e");
  } finally {
    optimizer.dispose();
  }
}

void _checkGradients(List<Tensor> params) {
  double maxGrad = 0;
  for (var p in params) {
    if (p.grad != null) {
      for (var g in p.grad!) {
        if (g.abs() > maxGrad) maxGrad = g.abs();
      }
    }
  }
  if (maxGrad > 1.0)
    print('  ⚠️ [GRAD] High Gradient: ${maxGrad.toStringAsFixed(4)}');
}
