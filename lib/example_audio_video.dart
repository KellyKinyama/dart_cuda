// file: test_run_multimodal_gpu.dart

import 'dart:math';
import 'audio_transformer.dart';
import 'video_transformer.dart';
import 'gpu_tensor.dart';
import 'adam.dart';

void main() {
  final random = Random();

  // --- 1. Model Initialization ---
  final audioModel = AudioTransformer(
    featureDim: 40,
    maxSequenceLength: 100,
    embedSize: 64,
    numClasses: 5,
  );

  final videoModel = VideoTransformer(
    frameEmbedDim: 128,
    embedSize: 128,
    maxVideoSequenceLength: 30,
    numClasses: 10,
  );

  final allParameters = [
    ...audioModel.parameters(),
    ...videoModel.parameters(),
  ];

  final optimizer = Adam(allParameters, lr: 0.001);

  // --- 2. Mock Data Generation ---
  final dummyAudioInput = Tensor.random([50, 40]);
  // FIX: Since the model outputs [1, numClasses], we only need 1 target index
  final List<int> audioTargets = [random.nextInt(5)];

  final dummyVideoInput = Tensor.random([20, 128]);
  // FIX: Same here, 1 target index for the whole video sequence
  final List<int> videoTargets = [random.nextInt(10)];

  print("🚀 Starting Multimodal GPU Training...");

  for (int epoch = 0; epoch < 1000; epoch++) {
    List<Tensor> tracker = [];

    // --- 3. Forward Passes ---
    // Returns [1, 5]
    final audioLogits = audioModel.forward(dummyAudioInput, tracker);
    // Returns [1, 10]
    final videoLogits = videoModel.forward(dummyVideoInput, tracker);

    // --- 4. Loss Calculation ---
    // crossEntropy now compares [1, V] logits with 1 target index
    final audioLoss = audioLogits.crossEntropy(audioTargets);
    final videoLoss = videoLogits.crossEntropy(videoTargets);

    // Combine losses (Multitask Learning)
    final totalLoss = audioLoss + videoLoss;
    tracker.add(totalLoss);

    // --- 5. Optimization ---
    optimizer.zeroGrad();

    // Backpropagate through the summed loss
    totalLoss.backward();
    optimizer.step();

    // --- 6. Logging ---
    if (epoch % 20 == 0) {
      final aL = audioLoss.fetchData()[0];
      final vL = videoLoss.fetchData()[0];
      print(
        "Epoch $epoch | Audio Loss: ${aL.toStringAsFixed(4)} | Video Loss: ${vL.toStringAsFixed(4)}",
      );
    }

    // --- 7. GPU MEMORY CLEANUP ---
    // Dispose of intermediate Tensors created during the forward pass
    for (var t in tracker) {
      t.dispose();
    }
    // Explicitly dispose of logits and losses not caught by tracker logic
    audioLogits.dispose();
    videoLogits.dispose();
    audioLoss.dispose();
    videoLoss.dispose();
  }

  print("\n✅ Multimodal training complete.");

  // --- 8. Final Cleanup ---
  for (var p in allParameters) p.dispose();
  dummyAudioInput.dispose();
  dummyVideoInput.dispose();
}
