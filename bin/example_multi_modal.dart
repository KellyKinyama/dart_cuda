// file: test_run_multimodal.dart

import 'dart:math';

import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/audio_transformer.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/multi_modal_transformer.dart';
import 'package:dart_cuda/video_transformer.dart';

void main() {
  final random = Random();

  // --- 1. Initialize Sub-Models ---
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

  // --- 2. Initialize Multimodal Wrapper ---
  final mmModel = MultimodalTransformer(
    audioModel: audioModel,
    videoModel: videoModel,
    numClasses: 7, // Joint classes (e.g., specific audio-visual events)
  );

  final optimizer = Adam(mmModel.parameters(), lr: 0.001);

  // --- 3. Create GPU Inputs (Training Data) ---
  final audioInput = Tensor.random([60, 40]);
  final videoInput = Tensor.random([25, 128]);
  final int targetLabel = random.nextInt(7);
  final List<int> jointTarget = [targetLabel];

  print("🚀 Starting Multimodal Fusion Training on GPU...");
  print("Target Class: $targetLabel");

  // --- 4. Training Loop ---
  for (int epoch = 0; epoch <= 1000; epoch++) {
    List<Tensor> tracker = [];

    // Forward Pass
    final logits = mmModel.forward(audioInput, videoInput, tracker);

    // Loss Calculation
    final loss = logits.crossEntropy(jointTarget);

    // Backward Pass & Optimization
    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    // Log Progress
    if (epoch % 100 == 0) {
      final lVal = loss.fetchData()[0];
      print("Epoch $epoch | Joint Loss: ${lVal.toStringAsFixed(6)}");
    }

    // Intermediate GPU Cleanup
    for (var t in tracker) t.dispose();
    loss.dispose();
    logits.dispose();
  }

  print("\n✅ Fusion training complete.");

  // --- 5. Inference Example ---
  print("--- Running Inference ---");

  // Create "new" data (slightly different lengths)
  final testAudio = Tensor.random([70, 40]);
  final testVideo = Tensor.random([20, 128]);

  List<Tensor> inferenceTracker = [];

  // Forward pass
  final inferenceLogits = mmModel.forward(
    testAudio,
    testVideo,
    inferenceTracker,
  );

  // Get probabilities via Softmax
  final probs = inferenceLogits.softmax();
  inferenceTracker.add(probs);

  // Fetch result to CPU
  final List<double> probDist = probs.fetchData();

  // Argmax to find the predicted class
  double maxProb = -1.0;
  int predictedClass = -1;
  for (int i = 0; i < probDist.length; i++) {
    if (probDist[i] > maxProb) {
      maxProb = probDist[i];
      predictedClass = i;
    }
  }

  print("Inference Result: Class $predictedClass");
  print("Confidence: ${(maxProb * 100).toStringAsFixed(2)}%");
  print(
    "Probability Distribution: ${probDist.map((e) => e.toStringAsFixed(3)).toList()}",
  );

  // --- 6. Final Cleanup ---
  print("\nCleaning up GPU memory...");

  // Dispose inference Tensors
  for (var t in inferenceTracker) t.dispose();
  inferenceLogits.dispose();
  testAudio.dispose();
  testVideo.dispose();

  // Dispose Training Tensors
  audioInput.dispose();
  videoInput.dispose();

  // Dispose Model Weights
  for (var p in mmModel.parameters()) {
    p.dispose();
  }

  print("✨ Done.");
}
