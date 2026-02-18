import 'gpu_tensor.dart';
import 'nn.dart';
import 'aft_transformer_encoder.dart';

class AudioTransformer extends Module {
  final int featureDim; // e.g., 40 (MFCCs)
  final int maxSequenceLength; // e.g., 100 frames
  final int embedSize;
  final int numClasses;

  // GPU Layers/Tensors
  late Layer featureProjection; // Projects featureDim -> embedSize
  late Tensor posEmbeddings; // [maxSequenceLength, embedSize]

  final TransformerEncoder transformerEncoder;
  late Layer classificationHead; // Final MLP for logits

  AudioTransformer({
    required this.featureDim,
    required this.maxSequenceLength,
    required this.embedSize,
    required this.numClasses,
    int numLayers = 4,
    int numHeads = 4,
  }) : transformerEncoder = TransformerEncoder(
         vocabSize: 0,
         embedSize: embedSize,
         blockSize: maxSequenceLength,
         numLayers: numLayers,
         numHeads: numHeads,
       ) {
    // 1. Audio Feature Projection (similar to patchProjection)
    featureProjection = Layer(featureDim, embedSize, useGelu: true);

    // 2. Positional Embeddings for the audio timeline
    posEmbeddings = Tensor.random([maxSequenceLength, embedSize], scale: 0.02);

    // 3. Output Head
    classificationHead = Layer(embedSize, numClasses, useGelu: false);
  }

  /// Takes input [actualSequenceLength, featureDim]
  Tensor forward(Tensor audioFeatures, List<Tensor> tracker) {
    // 1. Project MFCCs/Spectrogram features to embedding space
    // [Seq, featureDim] -> [Seq, embedSize]
    final xEmbed = featureProjection.forward(audioFeatures, tracker);

    // 2. Sliced Positional Embeddings
    // If the input audio is shorter than maxSequenceLength, we slice pos
    final actualSeqLen = audioFeatures.shape[0];
    final currentPos = posEmbeddings.slice(0, actualSeqLen);
    // Note: slice creates a view, add it to tracker if your engine requires it

    // 3. Add Positions
    final x = xEmbed + currentPos;
    tracker.add(x);

    // 4. Transformer Backbone
    final encoded = transformerEncoder.forwardEmbeddings(x, tracker);

    // 5. Global Average Pooling (Collapse [Seq, Embed] -> [1, Embed])
    // This aggregates the timeline into a single "audio fingerprint"
    final pooled = encoded.mean(); // Uses the sum/size reduction we discussed
    tracker.add(pooled);

    // 6. Classification Logits
    final logits = classificationHead.forward(pooled, tracker);

    return logits;
  }

  @override
  List<Tensor> parameters() => [
    ...featureProjection.parameters(),
    posEmbeddings,
    ...transformerEncoder.parameters(),
    ...classificationHead.parameters(),
  ];
}
