/// Full encoder-decoder Transformer.
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/standard/transformer_decoder.dart';
import 'package:dart_cuda/core/transformers/standard/transformer_encoder.dart';

class EncoderDecoderTransformer extends Module {
  final TransformerEncoder encoder;
  final TransformerDecoder decoder;

  EncoderDecoderTransformer({
    required int sourceVocabSize,
    required int targetVocabSize,
    required int embedSize,
    required int sourceBlockSize,
    required int targetBlockSize,
    required int numLayers,
    required int numHeads,
  }) : encoder = TransformerEncoder(
         vocabSize: sourceVocabSize,
         embedSize: embedSize,
         blockSize: sourceBlockSize,
         numLayers: numLayers,
         numHeads: numHeads,
       ),
       decoder = TransformerDecoder(
         vocabSize: targetVocabSize,
         embedSize: embedSize,
         blockSize: targetBlockSize,
         numLayers: numLayers,
         numHeads: numHeads,
         encoderEmbedSize: embedSize,
       );

  Tensor forward(
    List<int> sourceIdx,
    List<int> targetIdx,
    List<Tensor> tracker,
  ) {
    final encoderOutput = encoder.forward(sourceIdx, tracker);
    return decoder.forward(targetIdx, encoderOutput, tracker);
  }

  @override
  List<Tensor> parameters() => [
    ...encoder.parameters(),
    ...decoder.parameters(),
  ];
}
