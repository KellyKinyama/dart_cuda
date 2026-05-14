import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/transformers/modalities/text_transformer.dart';

void disposeAll(Iterable<Tensor> ts) { for (final t in ts) t.dispose(); }

void main() {
  for (int iter = 0; iter < 5; iter++) {
    print('iter $iter');
    final tt = TextTransformer(
      vocabSize: 32,
      maxSeqLen: 8,
      embedSize: 16,
      numLayers: 1,
      numHeads: 4,
    );
    final ids = [0, 1, 2, 3, 4];
    final tracker = <Tensor>[];
    final y = tt.forward(ids, tracker);
    print('  y.shape=${y.shape} first=${y.fetchData().take(3).toList()}');
    disposeAll(tracker);
    disposeAll(tt.parameters());
  }
  print('OK');
}
