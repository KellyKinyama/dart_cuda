import 'gpu_tensor.dart';
import 'nn.dart';
import 'aft_vit_backbone.dart';

class ViTObjectDetector extends Module {
  final ViTBackbone backbone;
  late Layer classHead;
  late Layer boxHead;
  final int numQueries;
  final int embedSize;
  final int numClasses;

  ViTObjectDetector({
    required int imageSize,
    required int patchSize,
    required int numChannels,
    required this.embedSize,
    required int numLayers,
    required int numHeads,
    required this.numClasses, // Store this to reshape correctly
    required this.numQueries,
  }) : backbone = ViTBackbone(
         imageSize: imageSize,
         patchSize: patchSize,
         numChannels: numChannels,
         embedSize: embedSize,
         numLayers: numLayers,
         numHeads: numHeads,
       ) {
    classHead = Layer(embedSize, numQueries * (numClasses + 1), useGelu: false);
    boxHead = Layer(embedSize, numQueries * 4, useGelu: false);
  }

  Map<String, Tensor> forward(Tensor patchifiedImage, List<Tensor> tracker) {
    // 1. Backbone Forward
    // This adds many intermediate tensors (LayerNorm, AFT, MLP) to the tracker
    final encodedFeatures = backbone.forward(patchifiedImage, tracker);

    // 2. Extract CLS token
    // getRow creates a NEW handle (a slice), so we must track it
    final clsFeature = encodedFeatures.getRow(0);
    tracker.add(clsFeature);

    // 3. Class Head Prediction
    final rawLogits = classHead.forward(clsFeature, tracker);

    // Reshape rawLogits from [1, numQueries * (numClasses + 1)] to [numQueries, numClasses + 1]
    // Marked as isView: true in Tensor class, so we don't need to track it separately
    final reshapedLogits = rawLogits.reshape([numQueries, numClasses + 1]);

    // 4. Box Head Prediction
    final rawBoxes = boxHead.forward(clsFeature, tracker);

    // Reshape and Sigmoid
    final reshapedBoxes = rawBoxes.reshape([numQueries, 4]);
    final boxes = reshapedBoxes.sigmoid();

    // IMPORTANT: Track the sigmoid result because it's a NEW handle
    tracker.add(boxes);

    // We return the reshaped versions so the Training Loop doesn't have to
    // do metadata math before the Hungarian Matching.
    return {'logits': reshapedLogits, 'boxes': boxes};
  }

  @override
  List<Tensor> parameters() => [
    ...backbone.parameters(),
    ...classHead.parameters(),
    ...boxHead.parameters(),
  ];
}
