import 'dart:io';
import 'dart:math' as math;
import 'package:dart_cuda/adam.dart';
// import 'package:dart_cuda/aft_transformer_decoder.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/network_utils.dart';
import '../aft_muzero_transformer_decoder.dart';
import 'muzero_greedy_agent.dart'; // Your refined Agent class

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
  final File file = File('tiny_shakespeare.txt');
  if (!await file.exists()) return;
  final String rawText = await file.readAsString();
  final tokenizer = CharTokenizer(rawText);
  final data = tokenizer.encode(rawText);

  const int blockSize = 32; // Reduced for VRAM
  const int embedSize = 96; // Reduced for VRAM
  const int numLayers = 4;
  const int numHeads = 6;

  final gpt = TransformerDecoder(
    vocabSize: tokenizer.vocabSize,
    embedSize: embedSize,
    encoderEmbedSize: embedSize,
    numLayers: numLayers,
    numHeads: numHeads,
    blockSize: blockSize,
  );

  final agent = MuZeroGreedyAgent(gpt, embedSize);
  final optimizer = Adam(gpt.parameters(), lr: 0.001);

  print("🎭 Training MuZero-Shakespeare...");

  for (int epoch = 0; epoch < 5000; epoch++) {
    optimizer.zeroGrad();
    List<Tensor> tracker = [];
    double totalLoss = 0;

    // 1. Get Batch
    final (x, y) = getBatch(data, blockSize);

    // 2. Initial Representation
    Tensor currentState = agent.representation([x[0]], tracker);

    // 3. Unrolled Latent Training
    // We alternate between Policy and Dynamics to keep gradients stable
    bool isPolicyStep = (epoch % 2 == 0);

    for (int t = 0; t < x.length - 1; t++) {
      if (isPolicyStep) {
        final logits = agent.predictPolicy(currentState, tracker);
        final loss = logits.crossEntropy([y[t]]);
        loss.backward();
        totalLoss += loss.fetchData()[0];

        // Teacher forcing for Policy mode
        currentState = agent.representation(x.sublist(0, t + 2), tracker);
      } else {
        // Imagination mode: Move state forward using g(s, a)
        Tensor nextState = agent.dynamics(currentState, x[t], t + 1, tracker);
        final logits = agent.predictPolicy(nextState, tracker);
        final loss = logits.crossEntropy([y[t]]);
        loss.backward();
        totalLoss += loss.fetchData()[0];

        currentState = nextState.detach();
      }
    }

    optimizer.step();
    // _safeCleanup(tracker, gpt.parameters());
    _safeCleanup(tracker, gpt.parameters());

    if (epoch % 100 == 0) {
      print(
        "Epoch $epoch | Loss: ${(totalLoss / blockSize).toStringAsFixed(4)}",
      );
    }
  }

  print("\n--- Generating Pure Latent Shakespeare ---");
  generateMuZeroShakespeare(agent, tokenizer, "ROMEO: ", 200);
}

void generateMuZeroShakespeare(
  MuZeroGreedyAgent agent,
  CharTokenizer tokenizer,
  String prompt,
  int length,
) {
  List<int> promptTokens = tokenizer.encode(prompt);
  stdout.write(prompt);

  List<Tensor> initTracker = [];
  // Initial thought
  Tensor rawState = agent.representation(promptTokens, initTracker);
  Tensor currentState = rawState.detach();
  for (var t in initTracker) t.dispose();

  for (int i = 0; i < length; i++) {
    List<Tensor> stepTracker = [];

    // 1. Predict next char from current latent state
    final logits = agent.predictPolicy(currentState, stepTracker);
    final row = logits.fetchData();

    // 2. Sample
    int nextId = sampleNucleus(row, temp: 0.8, topP: 0.9);
    stdout.write(tokenizer.decode([nextId]));

    // 3. Move latent state forward (Imagination)
    // We pass the chosen character BACK into the dynamics head
    Tensor nextStateRaw = agent.dynamics(
      currentState,
      nextId,
      promptTokens.length + i,
      stepTracker,
    );
    Tensor nextState = nextStateRaw.detach();

    // 4. Memory Handover
    currentState.dispose();
    for (var t in stepTracker) t.dispose();
    currentState = nextState;
  }

  currentState.dispose();
  print("\n[Latent Generation Complete]");
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

/// Prevents Double-Disposal and Parameter Deletion
void _safeCleanup(
  List<Tensor> tracker,
  // Tensor lossTensor,
  List<Tensor> params,
) {
  final freedAddresses = <int>{};

  // 1. Map current parameter addresses to protect them from disposal
  final paramAddresses = params.map((p) => p.handle.address).toSet();

  // 2. Clean the tracker
  for (var t in tracker) {
    final addr = t.handle.address;
    if (addr != 0 &&
        !freedAddresses.contains(addr) &&
        !paramAddresses.contains(addr) &&
        t.isView != true) {
      t.dispose();
      freedAddresses.add(addr);
    }
  }

  // 3. Explicitly clean the loss tensor if not already handled
  // final lossAddr = lossTensor.handle.address;
  // if (lossAddr != 0 &&
  //     !freedAddresses.contains(lossAddr) &&
  //     !paramAddresses.contains(lossAddr)) {
  // lossTensor.dispose();
  // }
}
