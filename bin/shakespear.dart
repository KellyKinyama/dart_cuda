import 'dart:io';
import 'dart:math' as math;

import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/aft_transformer_decoder.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/network_utils.dart';

class CharTokenizer {
  late List<String> chars;
  late Map<String, int> stoi;
  late Map<int, String> itos;

  CharTokenizer(String text) {
    // Get unique characters and sort them for consistency
    chars = (text.split('').toSet().toList())..sort();
    stoi = {for (var i = 0; i < chars.length; i++) chars[i]: i};
    itos = stoi.map((k, v) => MapEntry(v, k));
  }

  int get vocabSize => chars.length;

  List<int> encode(String s) => s.split('').map((c) => stoi[c] ?? 0).toList();
  String decode(List<int> l) => l.map((i) => itos[i] ?? '').join('');
}

(List<int>, List<int>) getBatch(List<int> data, int blockSize) {
  final rng = math.Random();
  // Pick a random starting index
  int start = rng.nextInt(data.length - blockSize - 1);

  // X is the sequence, Y is the same sequence shifted by 1
  final x = data.sublist(start, start + blockSize);
  final y = data.sublist(start + 1, start + blockSize + 1);

  return (x, y);
}

void main() async {
  // 1. Load the data
  final File file = File('tiny_shakespeare.txt');
  if (!await file.exists()) {
    print("Please download tiny_shakespeare.txt first!");
    return;
  }
  final String rawText = await file.readAsString();
  final tokenizer = CharTokenizer(rawText);
  final data = tokenizer.encode(rawText);

  // 2. Model Configuration (Larger for Shakespeare)
  const int blockSize = 64; // Context window
  const int embedSize = 256; // Vector size
  const int numLayers = 6;

  final gpt = TransformerDecoder(
    vocabSize: tokenizer.vocabSize,
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    numLayers: numLayers,
    numHeads: 8,
    blockSize: blockSize,
  );

  final optimizer = Adam(gpt.parameters(), lr: 0.0005);
  final dummyEnc = Tensor.zeros([1, embedSize]);

  print(
    "🎭 Training on Shakespeare (${tokenizer.vocabSize} unique characters)...",
  );

  // 3. Training Loop
  for (int step = 0; step < 2000; step++) {
    List<Tensor> tracker = [];
    optimizer.zeroGrad();

    // Get a random batch
    final (x, y) = getBatch(data, blockSize);

    // Forward
    final logits = gpt.forward(x, dummyEnc, tracker);
    final loss = logits.crossEntropy(y);

    // Backward
    loss.backward();
    optimizer.step();

    if (step % 100 == 0) {
      print("Step $step | Loss: ${loss.fetchData()[0].toStringAsFixed(4)}");
      // Save a checkpoint every 100 steps
      await saveModuleBinary(gpt, 'shakespeare_gpt.bin');
    }

    // Clean up
    for (var t in tracker) t.dispose();
    loss.dispose();
  }

  // 4. Generate some "Art"
  print("\n--- Generating Shakespearean Text ---");
  // String contextStr = "ROMEO: ";
  // List<int> gen = tokenizer.encode(contextStr);

  // 4. Generate some "Art"
  print("\n--- Generating Shakespearean Text ---");

  // Let's generate 200 characters starting with ROMEO:
  generateShakespeare(gpt, tokenizer, "ROMEO: ", 200, blockSize, dummyEnc);
}

void generateShakespeare(
  TransformerDecoder model,
  CharTokenizer tokenizer,
  String startText,
  int maxNewTokens,
  int blockSize,
  Tensor dummyEnc,
) {
  List<int> gen = tokenizer.encode(startText);
  stdout.write(startText); // Start printing the prompt

  for (int i = 0; i < maxNewTokens; i++) {
    List<Tensor> tracker = [];

    // 1. Crop context to blockSize
    List<int> context = gen.length > blockSize
        ? gen.sublist(gen.length - blockSize)
        : gen;

    // 2. GPU Forward Pass
    final logits = model.forward(context, dummyEnc, tracker);

    // 3. Fetch ONLY the last character's logits from GPU
    // Logits shape is [context.length, vocabSize]
    List<double> lastLogits = logits.fetchRow(context.length - 1);

    // 4. Sample using Temperature and Top-P
    int nextId = sampleNucleus(lastLogits, temp: 0.8, topP: 0.9);

    // 5. Update sequence and print immediately
    gen.add(nextId);
    stdout.write(tokenizer.decode([nextId]));

    // 6. Memory Cleanup (Crucial for long generation)
    for (var t in tracker) t.dispose();
    logits.dispose();
  }
  print('\n[Generation Complete]');
}

int sampleNucleus(List<double> row, {double temp = 1.0, double topP = 0.9}) {
  // Apply Temperature
  double maxL = row.reduce(math.max);
  List<double> probs = row.map((v) => math.exp((v - maxL) / temp)).toList();

  // Normalize
  double sumExp = probs.reduce((a, b) => a + b);
  for (int i = 0; i < probs.length; i++) probs[i] /= sumExp;

  // Sort for Nucleus
  List<MapEntry<int, double>> indexedProbs = probs.asMap().entries.toList();
  indexedProbs.sort((a, b) => b.value.compareTo(a.value));

  // Find Top-P threshold
  double cumulativeProb = 0.0;
  int cutoffIndex = 0;
  for (int i = 0; i < indexedProbs.length; i++) {
    cumulativeProb += indexedProbs[i].value;
    cutoffIndex = i;
    if (cumulativeProb >= topP) break;
  }

  // Re-normalize top candidates
  List<MapEntry<int, double>> candidates = indexedProbs.sublist(
    0,
    cutoffIndex + 1,
  );
  double candidateSum = candidates.fold(0, (sum, item) => sum + item.value);

  // Random Weighted Sample
  double r = math.Random().nextDouble() * candidateSum;
  double current = 0;
  for (var entry in candidates) {
    current += entry.value;
    if (r <= current) return entry.key;
  }
  return candidates.first.key;
}
