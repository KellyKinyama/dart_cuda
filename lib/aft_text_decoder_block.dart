// lib/aft_text_decoder_block.dart (New File)
import 'aft.dart'; // AFTAttention
import 'layer_norm.dart';
import 'nn.dart';
import 'gpu_tensor.dart';

class TextDecoderBlock extends Module {
  final AFTAttention selfAttention; // Masked Self-Attention
  final LayerNorm norm1;
  final AFTAttention crossAttention; // Cross-Attention to multimodal context
  final LayerNorm norm2;
  final Layer ff1, ff2; // Feed-forward network
  final LayerNorm norm3;

  TextDecoderBlock(int embedSize, int numHeads, int blockSize)
    : selfAttention = AFTAttention(
        embedSize,
        numHeads,
        blockSize,
        masked: true,
      ), // <--- CAUSAL
      norm1 = LayerNorm(embedSize),
      crossAttention = AFTAttention(
        embedSize,
        numHeads,
        blockSize,
      ), // <--- STANDARD
      norm2 = LayerNorm(embedSize),
      ff1 = Layer(embedSize, embedSize * 4, useGelu: true),
      ff2 = Layer(embedSize * 4, embedSize, useGelu: false),
      norm3 = LayerNorm(embedSize);

  Tensor forward(Tensor x, Tensor encoderOutput, List<Tensor> tracker) {
    // 1. Masked Self-Attention
    final attentionOutput1 = selfAttention.forward(
      norm1.forward(x, tracker),
      tracker,
    );
    final x1 = x + attentionOutput1; // Residual connection

    // 2. Cross-Attention to Multimodal Context
    // Q from x1, K/V from encoderOutput
    final attentionOutput2 = crossAttention.forward(
      norm2.forward(x1, tracker),
      tracker,
      kv: encoderOutput,
    );
    final x2 = x1 + attentionOutput2; // Residual connection

    // 3. Feed-Forward
    final ffOutput = ff2.forward(
      ff1.forward(norm3.forward(x2, tracker), tracker),
      tracker,
    );
    final x3 = x2 + ffOutput; // Residual connection

    return x3;
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttention.parameters(),
    ...norm1.parameters(),
    ...crossAttention.parameters(),
    ...norm2.parameters(),
    ...ff1.parameters(),
    ...ff2.parameters(),
    ...norm3.parameters(),
  ];
}
