import 'dart:math';
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/audio_transformer.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/multi_modal_trnasformer_encoder.dart';
import 'package:dart_cuda/text_decoder.dart';
// import 'package:dart_cuda/multi_modal_transformer2.dart';
import 'package:dart_cuda/video_transformer.dart';
import 'package:dart_cuda/text_transformer.dart';

// test_run_multimodal.dart (Updated for Generative Task)
void main() {
  final random = Random();
  final vocabSize = 1000;
  final jointEmbedSize = 128; // All embeddings should be this size
  final maxDecoderSeqLen = 50;

  // --- 1. Initialize Encoders ---
  final audioEncoder = AudioTransformer(
    featureDim: 40,
    maxSequenceLength: 100, // Audio sequence length
    embedSize: jointEmbedSize, // Match joint embed size
    numClasses: 10, // <--- REQUIRED
  );

  final videoEncoder = VideoTransformer(
    frameEmbedDim: 128,
    maxVideoSequenceLength: 30, // Video sequence length
    embedSize: jointEmbedSize, // Match joint embed size
    numClasses: 10, // <--- REQUIRED
  );

  final textEncoderForInput = TextTransformer(
    // For encoding any input text prompts
    vocabSize: vocabSize,
    maxSeqLen: 20, // Max length for input text prompt
    embedSize: jointEmbedSize, // Match joint embed size
  );

  // --- 2. Initialize Multimodal Encoder (Your combined A/V/T input) ---
  final multimodalEncoder = MultimodalTransformer(
    audioEncoder: audioEncoder,
    videoEncoder: videoEncoder,
    textEncoder: textEncoderForInput,
    jointEmbedSize: jointEmbedSize,
    maxTotalSeqLen: 100 + 30 + 20, // Sum of max sequences
  );

  // --- 3. Initialize Text Decoder ---
  final textDecoder = TextDecoder(
    vocabSize: vocabSize,
    maxSeqLen: maxDecoderSeqLen, // Max length of generated text
    embedSize:
        jointEmbedSize, // Must match multimodal encoder's output embedSize
  );

  // --- Combine all parameters for the optimizer ---
  final allParameters = [
    ...multimodalEncoder.parameters(),
    ...textDecoder.parameters(),
  ];
  final optimizer = Adam(allParameters, lr: 0.0001);

  // --- 4. Synthetic Training Data (Example) ---
  final audioInput = Tensor.random([60, 40]);
  final videoInput = Tensor.random([25, 128]);
  final List<int> inputTextPrompt = [
    1,
  ]; // E.g., just a [CLS] token, or "Describe this:"

  // Target text sequence (e.g., "a person is singing loudly")
  final List<int> targetTextSequence = [
    10,
    25,
    4,
    88,
    1,
    0,
  ]; // (includes an EOS token '0')

  print("🚀 Starting Multimodal Text Generation Training on GPU...");
  print("Target Text Sequence Length: ${targetTextSequence.length}");

  // --- 5. Training Loop ---
  for (int epoch = 0; epoch <= 100; epoch++) {
    List<Tensor> tracker = [];
    optimizer.zeroGrad();

    // Multimodal Encoder: Get the context
    final encoderOutput = multimodalEncoder.forward(
      audioInput,
      videoInput,
      inputTextPrompt,
      tracker,
    );

    // Text Decoder: Predict the next tokens given the context and partial target
    // We train the decoder to predict each token in `targetTextSequence`
    // using 'teacher forcing' (feeding the ground truth previous tokens).
    final decoderInputTokens = targetTextSequence.sublist(
      0,
      targetTextSequence.length - 1,
    ); // Exclude last token
    final targetTokensForLoss = targetTextSequence.sublist(1); // Shifted target

    final decoderLogits = textDecoder.forward(
      decoderInputTokens,
      encoderOutput,
      tracker,
    );

    // Loss Calculation (Cross Entropy over the vocabulary for each token)
    final loss = decoderLogits.crossEntropy(targetTokensForLoss);

    loss.backward();
    optimizer.step();

    if (epoch % 10 == 0) {
      final lVal = loss.fetchData()[0];
      print("Epoch $epoch | Generative Loss: ${lVal.toStringAsFixed(6)}");
    }

    // Cleanup
    for (var t in tracker) t.dispose();
    loss.dispose();
    decoderLogits.dispose();
    encoderOutput.dispose(); // Also dispose encoder output
  }

  print("\n✅ Multimodal Text Generation Training Complete.");

  // --- 6. Inference (Text Generation) ---
  print("--- Running Text Generation Inference ---");

  List<int> generatedTokens = [1]; // Start with a [CLS] or BOS token
  List<Tensor> inferenceTracker = [];

  for (int i = 0; i < maxDecoderSeqLen - 1; i++) {
    // Get encoder output (only once per generation)
    final encoderOutput = multimodalEncoder.forward(
      Tensor.random([60, 40]), // Dummy input for inference
      Tensor.random([25, 128]), // Dummy input for inference
      [1], // Dummy input text prompt
      inferenceTracker,
    );

    // Pass current generated sequence to the decoder
    final decoderLogits = textDecoder.forward(
      generatedTokens,
      encoderOutput,
      inferenceTracker,
    );

    // Get the last token's probabilities
    final lastTokenLogits = decoderLogits.slice(decoderLogits.shape[0] - 1, 1);
    final probs = lastTokenLogits.softmax();

    final probDist = probs.fetchData();

    // Sample or Argmax the next token
    int nextToken = probDist.indexOf(probDist.reduce(max)); // Argmax sampling
    generatedTokens.add(nextToken);

    // Stop if EOS token is generated (assuming 0 is EOS)
    if (nextToken == 0) break;

    // Dispose current encoderOutput and decoderLogits for the next step
    encoderOutput.dispose();
    decoderLogits.dispose();
    probs.dispose();
    lastTokenLogits.dispose();
  }

  print("Generated Token IDs: $generatedTokens");
  // You would convert these token IDs back to words using a vocabulary
  print("✨ Text Generation Inference Done.");

  // --- 7. Final Deep Cleanup ---
  print("Cleaning up all GPU resources...");
  // Dispose all model parameters
  for (var p in allParameters) p.dispose();
  audioInput.dispose();
  videoInput.dispose();
  // Any remaining inference tensors in the tracker
  for (var t in inferenceTracker) t.dispose();
  print("✨ GPU Memory Cleared.");
}
