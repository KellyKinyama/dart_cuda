import 'aft_vit_backbone.dart';
import 'gpu_tensor.dart';
import 'nn.dart';

class ViTFaceEmbeddingGPU extends Module {
  final ViTBackbone backbone;
  late Layer? finalProjection;

  ViTFaceEmbeddingGPU({
    required int imageSize,
    required int patchSize,
    required int embedSize,
    int outputDim = 512,
    int numLayers = 4,
  }) : backbone = ViTBackbone(
         imageSize: imageSize,
         patchSize: patchSize,
         embedSize: embedSize,
         numLayers: numLayers,
       ) {
    if (embedSize != outputDim) {
      finalProjection = Layer(embedSize, outputDim, useGelu: false);
    } else {
      finalProjection = null;
    }
  }

  /// Input: [numPatches, patchPixels]
  /// Output: [1, outputDim] L2-normalized vector
  /// Input: [numPatches, patchPixels]
  /// Output: [1, outputDim] L2-normalized vector
  Tensor getFaceEmbedding(Tensor patchifiedImage, List<Tensor> tracker) {
    // 1. Get sequence output from ViT backbone
    final encoded = backbone.forward(patchifiedImage, tracker);
    tracker.add(encoded);

    // 2. Extract the [CLS] token (assumed to be the first row)
    final clsOutput = encoded.slice(0, 1);
    tracker.add(clsOutput);

    Tensor faceVec = clsOutput;

    // 3. Project to the embedding dimension (e.g., 512)
    if (finalProjection != null) {
      faceVec = finalProjection!.forward(clsOutput, tracker);
      tracker.add(faceVec);
    }

    // 4. Native L2 Normalization
    // Replaces the manual math with a single optimized GPU kernel call
    final normVec = faceVec.normalize(eps: 1e-10);
    tracker.add(normVec);

    return normVec;
  }

  @override
  List<Tensor> parameters() => [
    ...backbone.parameters(),
    if (finalProjection != null) ...finalProjection!.parameters(),
  ];
}
