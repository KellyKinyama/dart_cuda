// file: lib/multi_modal_transformer.dart
import 'audio_transformer.dart';
import 'gpu_tensor.dart';
import 'nn.dart';
import 'text_transformer.dart';
import 'video_transformer.dart';

class MultimodalTransformer extends Module {
  final AudioTransformer audioModel;
  final VideoTransformer videoModel;
  final TextTransformer textModel;
  final Layer fusionLayer;

  MultimodalTransformer({
    required this.audioModel,
    required this.videoModel,
    required this.textModel,
    required int numClasses,
  }) : fusionLayer = Layer(
         // Logic: Audio(64) + Video(128) + Text(128) = 320
         audioModel.embedSize + videoModel.embedSize + textModel.embedSize,
         numClasses,
         useGelu: false,
       );

  Tensor forward(
    Tensor audio,
    Tensor video,
    List<int> textTokens,
    List<Tensor> tracker,
  ) {
    // 1. Get embeddings (Pooling across sequence dimension to get a single vector per modality)
    final a = audioModel.forward(audio, tracker).mean();
    final v = videoModel.forward(video, tracker).mean();
    final t = textModel.forward(textTokens, tracker).mean();

    // 2. Multimodal Fusion via Concatenation
    final combined = Tensor.concat([a, v, t]);
    tracker.add(combined);

    // 3. Final Classification Head
    return fusionLayer.forward(combined, tracker);
  }

  @override
  List<Tensor> parameters() => [
    ...audioModel.parameters(),
    ...videoModel.parameters(),
    ...textModel.parameters(),
    ...fusionLayer.parameters(),
  ];
}
