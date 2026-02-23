import 'dart:math';
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/audio_transformer.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/multi_modal_transformer2.dart';
import 'package:dart_cuda/video_transformer.dart';
import 'package:dart_cuda/text_transformer.dart';

void main() {
  final random = Random();

  // --- 1. Initialize Modalities with full parameters ---
  // Each sub-model needs its standalone numClasses for internal heads
  final audioModel = AudioTransformer(
    featureDim: 40,
    maxSequenceLength: 100,
    embedSize: 64,
    numClasses: 5, // <--- REQUIRED
    numLayers: 2,
    numHeads: 4,
  );

  final videoModel = VideoTransformer(
    frameEmbedDim: 128,
    embedSize: 128,
    maxVideoSequenceLength: 30,
    numClasses: 10, // <--- REQUIRED
    numLayers: 2,
    numHeads: 4,
  );

  // final textModel = TextTransformer(
  //   vocabSize: 1000,
  //   maxSeqLen: 50,
  //   embedSize: 128,
  //   numClasses: 5, // <--- REQUIRED
  //   numLayers: 2,
  //   numHeads: 4,
  // );

  // Text is a pure encoder (no numClasses needed here)
  final textModel = TextTransformer(
    vocabSize: 1000,
    maxSeqLen: 50,
    embedSize: 128,
  );

  // --- 2. Initialize Multimodal Wrapper ---
  final mmModel = MultimodalTransformer(
    audioModel: audioModel,
    videoModel: videoModel,
    textModel: textModel,
    numClasses: 7, // This is the "Joint" output class count
  );

  final optimizer = Adam(mmModel.parameters(), lr: 0.001);

  // --- 3. Synthetic Training Data ---
  final audioInput = Tensor.random([60, 40]);
  final videoInput = Tensor.random([25, 128]);
  final List<int> textInput = [10, 25, 4, 88, 1]; // Dummy tokens

  final int targetLabel = random.nextInt(7);
  final List<int> jointTarget = [targetLabel];

  print("🚀 Starting Triple-Modality Fusion (A+V+T) on GPU...");
  print(
    "Target Class: $targetLabel | Params: ${mmModel.parameters().length} tensors",
  );

  // --- 4. Training Loop ---
  for (int epoch = 0; epoch <= 500; epoch++) {
    List<Tensor> tracker = [];

    // Forward Pass (Audio, Video, Text)
    final logits = mmModel.forward(audioInput, videoInput, textInput, tracker);

    // Loss calculation (Cross Entropy on GPU)
    final loss = logits.crossEntropy(jointTarget);

    optimizer.zeroGrad();
    loss.backward();
    optimizer.step();

    if (epoch % 50 == 0) {
      final lVal = loss.fetchData()[0];
      print("Epoch $epoch | Multi-Loss: ${lVal.toStringAsFixed(6)}");
      if (lVal.isNaN) break;
    }

    // Memory Cleanup
    for (var t in tracker) t.dispose();
    loss.dispose();
    logits.dispose();
  }

  print("\n✅ Multimodal Training Complete.");

  // --- 5. Multi-Modal Inference ---
  print("--- Running Inference (Simulated Input) ---");

  final testAudio = Tensor.random([45, 40]);
  final testVideo = Tensor.random([15, 128]);
  final testText = [1, 2, 3];

  List<Tensor> iTracker = [];
  final iLogits = mmModel.forward(testAudio, testVideo, testText, iTracker);
  final probs = iLogits.softmax();

  final probDist = probs.fetchData();
  int pred = probDist.indexOf(probDist.reduce(max));

  print(
    "Inference Result: Class $pred with confidence ${(probDist[pred] * 100).toStringAsFixed(2)}%",
  );

  // --- 6. Deep Cleanup ---
  print("Cleaning up all GPU resources...");
  for (var t in iTracker) t.dispose();
  iLogits.dispose();
  testAudio.dispose();
  testVideo.dispose();
  audioInput.dispose();
  videoInput.dispose();

  for (var p in mmModel.parameters()) p.dispose();
  print("✨ GPU Memory Cleared.");
}
