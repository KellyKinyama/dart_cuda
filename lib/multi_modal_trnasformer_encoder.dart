// lib/multi_modal_transformer.dart (Updated)
import 'aft_transformer_encoder.dart';
import 'audio_transformer.dart';
import 'gpu_tensor.dart';
import 'nn.dart';
import 'text_transformer.dart';
import 'video_transformer.dart';

class MultimodalTransformer extends Module {
  final AudioTransformer audioEncoder;
  final VideoTransformer videoEncoder;
  final TextTransformer
  textEncoder; // Use as an encoder for input text if needed

  // You might want a joint transformer here if the modalities need to interact before the decoder
  final TransformerEncoder fusionEncoder;
  final int jointEmbedSize;

  MultimodalTransformer({
    required this.audioEncoder,
    required this.videoEncoder,
    required this.textEncoder, // This will be used as a pre-processor for input text
    required this.jointEmbedSize, // e.g., 128
    int fusionLayers = 2,
    int fusionHeads = 4,
    int maxTotalSeqLen = 200, // Max length of combined modality sequences
  }) : fusionEncoder = TransformerEncoder(
         vocabSize: 0, // No direct token embeddings
         embedSize: jointEmbedSize,
         blockSize: maxTotalSeqLen,
         numLayers: fusionLayers,
         numHeads: fusionHeads,
       ),
       assert(
         audioEncoder.embedSize == jointEmbedSize &&
             videoEncoder.embedSize == jointEmbedSize &&
             textEncoder.embedSize == jointEmbedSize,
         "All encoder outputs must match jointEmbedSize for concatenation",
       );

  // This forward now returns the multimodal *context* for the decoder
  Tensor forward(
    Tensor audio,
    Tensor video,
    List<int> inputTextTokens,
    List<Tensor> tracker,
  ) {
    // 1. Get full sequence embeddings from each modality
    final audioSeqEmbeds = audioEncoder.forward(
      audio,
      tracker,
    ); // [SeqA, Embed]
    final videoSeqEmbeds = videoEncoder.forward(
      video,
      tracker,
    ); // [SeqV, Embed]
    final textSeqEmbeds = textEncoder.forward(
      inputTextTokens,
      tracker,
    ); // [SeqT, Embed]

    // 2. Concatenate all sequences
    final combinedSequence = Tensor.concat([
      audioSeqEmbeds,
      videoSeqEmbeds,
      textSeqEmbeds,
    ]);
    tracker.add(combinedSequence);

    // 3. Optional: Pass through a fusion transformer for deeper interaction
    final fusedOutput = fusionEncoder.forwardEmbeddings(
      combinedSequence,
      tracker,
    );
    return fusedOutput; // This is the encoderOutput for the TextDecoder
  }

  @override
  List<Tensor> parameters() => [
    ...audioEncoder.parameters(),
    ...videoEncoder.parameters(),
    ...textEncoder.parameters(),
    ...fusionEncoder.parameters(),
  ];
}
