import 'dart:io';
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/aft_transformer_decoder.dart';
import 'package:dart_cuda/gpu_tensor.dart';

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

void main() async {
  final String rawText = "the cat sat on the mat";
  final tokenizer = CharTokenizer(rawText);
  final data = tokenizer.encode(rawText);

  const int blockSize = 8; // Increased from 4
  const int embedSize = 32; // Increased from 16

  final gpt = TransformerDecoder(
    vocabSize: tokenizer.vocabSize,
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    numLayers: 1,
    numHeads: 2,
    blockSize: blockSize,
  );

  final optimizer = Adam(gpt.parameters(), lr: 0.01);
  final dummyEnc = Tensor.zeros([1, embedSize]);

  print("🚀 Training to overfit the full sentence...");

  // 1. IMPROVED TRAINING: Slide through the whole sentence
  for (int step = 0; step < 501; step++) {
    optimizer.zeroGrad();
    double epochLoss = 0;
    int count = 0;

    // Slide over the text so the model sees all transitions
    for (int i = 0; i < data.length - blockSize; i++) {
      List<Tensor> tracker = [];
      final x = data.sublist(i, i + blockSize);
      final y = data.sublist(i + 1, i + blockSize + 1);

      final logits = gpt.forward(x, dummyEnc, tracker);
      final loss = logits.crossEntropy(y);

      loss.backward();
      epochLoss += loss.fetchData()[0];
      count++;

      // Cleanup
      for (var t in tracker) {
        if (!gpt.parameters().contains(t)) t.dispose();
      }
      loss.dispose();
    }
    optimizer.step();

    if (step % 100 == 0) {
      print("Step $step | Avg Loss: ${(epochLoss / count).toStringAsFixed(4)}");
    }
  }

  // 2. GENERATION: The Autoregressive Loop
  print("\n--- Generating Entire Sequence ---");

  // Start with the initial prompt "the "
  List<int> generatedIds = data.sublist(0, blockSize);
  stdout.write(tokenizer.decode(generatedIds));

  // Generate until we reach the length of the original text
  for (int i = 0; i < rawText.length - blockSize; i++) {
    List<Tensor> evalTracker = [];

    // Always take the most recent context (blockSize)
    List<int> context = generatedIds.sublist(generatedIds.length - blockSize);

    final logits = gpt.forward(context, dummyEnc, evalTracker);

    // Get the logits for the very last token in the window
    List<double> lastRow = logits.fetchRow(context.length - 1);

    // Greedy Search: Pick the absolute best character (ArgMax)
    int predId = 0;
    double maxVal = -double.infinity;
    for (int v = 0; v < lastRow.length; v++) {
      if (lastRow[v] > maxVal) {
        maxVal = lastRow[v];
        predId = v;
      }
    }

    generatedIds.add(predId);
    stdout.write(tokenizer.decode([predId]));

    // Cleanup memory
    for (var t in evalTracker) {
      t.dispose();
    }
    logits.dispose();
  }

  print("\n\nFinished!");
}
