// file: multimodal_transformer.dart

import 'gpu_tensor.dart';
import 'audio_transformer.dart';
import 'nn.dart';
import 'video_transformer.dart';

class MultimodalTransformer extends Module {
  final AudioTransformer audioModel;
  final VideoTransformer videoModel;
  final Layer fusionHead;
  final int numClasses;

  MultimodalTransformer({
    required this.audioModel,
    required this.videoModel,
    required this.numClasses,
  }) : fusionHead = Layer(
         audioModel.embedSize + videoModel.embedSize, // 64 + 128 = 192
         numClasses,
         useGelu: false,
       );

  /// Forward pass performing 'Intermediate Fusion' on GPU
  Tensor forward(Tensor audioIn, Tensor videoIn, List<Tensor> tracker) {
    // 1. Extract Audio Features
    final audioEmbed = audioModel.featureProjection.forward(audioIn, tracker);
    final aPos = audioModel.posEmbeddings.slice(0, audioIn.shape[0]);
    final aCombined = audioEmbed + aPos;
    tracker.add(aCombined);
    final aEncoded = audioModel.transformerEncoder.forwardEmbeddings(
      aCombined,
      tracker,
    );
    final aPooled = aEncoded.mean(); // [1, 64]
    tracker.add(aPooled);

    // 2. Extract Video Features
    Tensor vProjected = videoModel.frameProjection != null
        ? videoModel.frameProjection!.forward(videoIn, tracker)
        : videoIn;
    final vPos = videoModel.posEmbeddings.slice(0, videoIn.shape[0]);
    final vCombined = vProjected + vPos;
    tracker.add(vCombined);
    final vEncoded = videoModel.transformerEncoder.forwardEmbeddings(
      vCombined,
      tracker,
    );
    final vPooled = vEncoded.mean(); // [1, 128]
    tracker.add(vPooled);

    // 3. Fusion via Concatenation [1, 192]
    final fused = Tensor.concat([aPooled, vPooled]);
    tracker.add(fused);

    // 4. Final Classification
    return fusionHead.forward(fused, tracker);
  }

  @override
  List<Tensor> parameters() => [
    ...audioModel.parameters(),
    ...videoModel.parameters(),
    ...fusionHead.parameters(),
  ];
}
