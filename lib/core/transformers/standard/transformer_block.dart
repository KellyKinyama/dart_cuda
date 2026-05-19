/// Decoder-only GPT-style transformer block (pre-norm).
///
///     x_norm1 = LN1(x)
///     attn    = MHA(x_norm1)
///     x_res1  = x + attn
///     x_norm2 = LN2(x_res1)
///     ffn     = FFN(x_norm2)
///     out     = x_res1 + ffn
import 'package:dart_cuda/core/attention/multi_head_attention.dart';
import 'package:dart_cuda/core/layers/feed_forward.dart';
import 'package:dart_cuda/core/layers/layer_norm.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

class TransformerBlock extends Module {
  final int embedSize;
  final MultiHeadAttention attention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;

  TransformerBlock(this.embedSize, int numHeads, {bool masked = false})
      : attention = MultiHeadAttention(numHeads, embedSize, masked: masked),
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  Tensor forward(Tensor x, List<Tensor> tracker) {
    final xNorm1 = ln1.forward(x, tracker);
    final attnOut = attention.forward(xNorm1, tracker);
    final xRes1 = x + attnOut;
    tracker.addAll([attnOut, xRes1]);

    final xNorm2 = ln2.forward(xRes1, tracker);
    final ffnOut = ffn.forward(xNorm2, tracker);
    final out = xRes1 + ffnOut;
    tracker.add(ffnOut);
    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...attention.parameters(),
        ...ffn.parameters(),
        ...ln1.parameters(),
        ...ln2.parameters(),
      ];
}
