// file: lib/text_transformer.dart
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/aft/transformer_encoder.dart';
import 'package:dart_cuda/core/layers/nn.dart';

class TextTransformer extends Module {
  final TransformerEncoder encoder;
  final int embedSize;

  TextTransformer({
    required int vocabSize,
    required int maxSeqLen,
    required this.embedSize,
    int numLayers = 4,
    int numHeads = 4,
  }) : encoder = TransformerEncoder(
         vocabSize: vocabSize,
         embedSize: embedSize,
         blockSize: maxSeqLen,
         numLayers: numLayers,
         numHeads: numHeads,
       );

  Tensor forward(List<int> tokens, List<Tensor> tracker) {
    // Uses your token embedding + position embedding + transformer blocks
    return encoder.forward(tokens, tracker);
  }

  @override
  List<Tensor> parameters() => encoder.parameters();
}
