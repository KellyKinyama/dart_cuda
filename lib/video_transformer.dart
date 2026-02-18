import 'aft_transformer_encoder.dart';
import 'gpu_tensor.dart';
import 'nn.dart';

class VideoTransformer extends Module {
  // Inherit from Module for the override
  final int frameEmbedDim;
  final int embedSize;
  final int maxVideoSequenceLength;
  final int numClasses;

  final Layer? frameProjection;
  final Tensor
  posEmbeddings; // Renamed from positionEmbeddings to match parameters()
  final TransformerEncoder transformerEncoder;
  final Layer mlpHead;

  VideoTransformer({
    required this.frameEmbedDim,
    required this.embedSize,
    required this.maxVideoSequenceLength,
    required this.numClasses,
    int numLayers = 2,
    int numHeads = 4,
  }) : frameProjection = (frameEmbedDim != embedSize)
           ? Layer(frameEmbedDim, embedSize)
           : null,
       posEmbeddings = Tensor.random([maxVideoSequenceLength, embedSize]),
       transformerEncoder = TransformerEncoder(
         vocabSize: 0, // Added to match standard signature
         embedSize: embedSize,
         blockSize: maxVideoSequenceLength,
         numLayers: numLayers,
         numHeads: numHeads,
       ),
       mlpHead = Layer(embedSize, numClasses);

  /// Forward pass on the GPU
  Tensor forward(Tensor videoEmbeddings, List<Tensor> tracker) {
    final int numFrames = videoEmbeddings.shape[0];

    if (numFrames > maxVideoSequenceLength) {
      throw ArgumentError("Video too long for maxVideoSequenceLength");
    }

    // 1. Projection
    Tensor x = frameProjection != null
        ? frameProjection!.forward(videoEmbeddings, tracker)
        : videoEmbeddings;

    // 2. Slice and Add Positions
    final pos = posEmbeddings.slice(0, numFrames);
    // Note: slice is a view, usually doesn't need tracking unless you want to auto-dispose

    final xWithPos = x + pos;
    tracker.add(xWithPos);

    // 3. Transformer Encoder
    final encoded = transformerEncoder.forwardEmbeddings(xWithPos, tracker);

    // 4. Global Average Pooling
    final pooled = encoded.mean();
    tracker.add(pooled);

    // 5. MLP Head
    final logits = mlpHead.forward(pooled, tracker);

    return logits;
  }

  @override
  List<Tensor> parameters() {
    return [
      if (frameProjection != null) ...frameProjection!.parameters(),
      posEmbeddings, // Now this matches the field name above!
      ...transformerEncoder.parameters(),
      ...mlpHead.parameters(),
    ];
  }
}
