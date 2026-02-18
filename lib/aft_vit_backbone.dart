import 'gpu_tensor.dart';
import 'nn.dart';
import 'aft_transformer_encoder.dart';

class ViTBackbone extends Module {
  final int imageSize;
  final int patchSize;
  final int numChannels;
  final int embedSize;

  // Weights on GPU
  late Layer
  patchProjection; // Linear(patchSize * patchSize * numChannels, embedSize)
  late Tensor clsToken; // [1, embedSize]
  late Tensor posEmbeddings; // [numPatches + 1, embedSize]

  final TransformerEncoder transformerEncoder;

  ViTBackbone({
    required this.imageSize,
    required this.patchSize,
    this.numChannels = 3,
    required this.embedSize,
    int numLayers = 4,
    int numHeads = 4,
  }) : transformerEncoder = TransformerEncoder(
         vocabSize: 0, // Not used for vision
         embedSize: embedSize,
         blockSize: (imageSize ~/ patchSize) * (imageSize ~/ patchSize) + 1,
         numLayers: numLayers,
         numHeads: numHeads,
       ) {
    int numPatches = (imageSize ~/ patchSize) * (imageSize ~/ patchSize);

    // 1. Projection layer (GPU weights initialized via Layer class)
    patchProjection = Layer(
      patchSize * patchSize * numChannels,
      embedSize,
      useGelu: false,
    );

    // 2. Learnable [CLS] token [1, embedSize]
    clsToken = Tensor.random([1, embedSize], scale: 0.02);

    // 3. Positional Embeddings [N+1, embedSize]
    posEmbeddings = Tensor.random([numPatches + 1, embedSize], scale: 0.02);
  }

  /// Takes a Tensor of shape [numPatches, patch_pixels]
  /// For a 224x224 image with 16x16 patches, input is [196, 768]
  Tensor forward(Tensor patchifiedImage, List<Tensor> tracker) {
    // 1. Linear Projection of Patches
    // [numPatches, pixels] -> [numPatches, embedSize]
    final xPatches = patchProjection.forward(patchifiedImage, tracker);

    // 2. Prepend CLS Token
    // Result shape: [numPatches + 1, embedSize]
    final xSeq = Tensor.concat([clsToken, xPatches]);
    tracker.add(xSeq);

    // 3. Add Positional Embeddings
    // (Element-wise addition on GPU)
    final xPos = xSeq + posEmbeddings;
    tracker.add(xPos);

    // 4. Transformer Encoder Blocks
    // Contextualizes all patches using AFT
    final encoded = transformerEncoder.forwardEmbeddings(xPos, tracker);

    return encoded;
  }

  @override
  List<Tensor> parameters() => [
    ...patchProjection.parameters(),
    clsToken,
    posEmbeddings,
    ...transformerEncoder.parameters(),
  ];
}
