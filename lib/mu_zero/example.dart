import '../aft_muzero_transformer_decoder.dart';
import '../gpu_tensor.dart';
import '../adam.dart'; // Ensure this path is correct for your Adam optimizer
import 'muzero_greedy_agent.dart';

void main() {
  print("--- Stable Tensor-Engine MuZero-GPT: Training & Generation ---");

  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    ".": 2,
    "<start>": 3,
    "<pad>": 4,
  };
  final Map<int, String> itos = stoi.map((k, v) => MapEntry(v, k));

  final model = TransformerDecoder(
    vocabSize: 5,
    embedSize: 32,
    encoderEmbedSize: 32,
    blockSize: 16,
    numLayers: 2,
    numHeads: 4,
  );
  final agent = MuZeroGreedyAgent(model, 32);

  final List<int> trainingData = [
    stoi["<start>"]!,
    stoi["hello"]!,
    stoi["world"]!,
    stoi["."]!,
  ];

  // --- TRAIN FOR MORE THAN 0 EPOCHS ---
  trainMuZero(agent, trainingData, 100); // 500 epochs to actually learn

  print("\n--- Generation After Training ---");
  generateMuZeroGreedy(agent, [stoi["<start>"]!], 4, itos);
}

void trainMuZero(
  MuZeroGreedyAgent agent,
  List<int> targetSequence,
  int epochs,
) {
  final optimizer = Adam(agent.model.parameters(), lr: 0.001);

  for (int epoch = 0; epoch < epochs; epoch++) {
    optimizer.zeroGrad();
    double totalEpochLoss = 0.0;

    // Tracker is per-epoch
    final List<Tensor> tracker = [];

    // Start from first token
    Tensor currentState = agent.representation([targetSequence[0]], tracker);

    for (int t = 0; t < targetSequence.length - 1; t++) {
      final int target = targetSequence[t + 1];

      // --- Policy ---
      final Tensor logits = agent.predictPolicy(currentState, tracker);

      final Tensor loss = logits.crossEntropy([target]);

      // Backprop while graph exists
      loss.backward();

      totalEpochLoss += loss.fetchData()[0];

      // --- Dynamics ---
      final Tensor nextState = agent.dynamics(
        currentState,
        target,
        t + 1,
        tracker,
      );

      currentState = nextState.detach();

      // 🔑 HARD GRAPH CUT:
      // Dispose entire step graph and re-enter model cleanly
      for (final t in tracker) {
        t.dispose();
      }
      tracker.clear();

      // Recompute state WITHOUT history (no BPTT)
      currentState = agent.representation(
        targetSequence.sublist(0, t + 2),
        tracker,
      );
    }

    // Update parameters
    optimizer.step();

    // Final cleanup
    for (final t in tracker) {
      t.dispose();
    }
    tracker.clear();

    if (epoch % 50 == 0) {
      final wSample = agent.model.lmHead.parameters()[0].fetchData().sublist(
        0,
        3,
      );

      print(
        "Epoch ${epoch.toString().padLeft(3)} | "
        "Loss: ${totalEpochLoss.toStringAsFixed(6)} | "
        "Weights: $wSample",
      );
    }
  }
}

void generateMuZeroGreedy(
  MuZeroGreedyAgent agent,
  List<int> prompt,
  int maxLength,
  Map<int, String> itos,
) {
  final List<int> generated = List.from(prompt);

  for (int i = 0; i < maxLength; i++) {
    // Fresh tracker PER STEP (no graphs survive)
    final List<Tensor> tracker = [];

    // Recompute state from all known tokens
    final Tensor state = agent.representation(generated, tracker);

    final Tensor logits = agent.predictPolicy(state, tracker);

    final int bestToken = argMax(logits.fetchData());

    generated.add(bestToken);
    print("Step $i -> ${itos[bestToken]}");

    // Safe cleanup
    for (final t in tracker) {
      t.dispose();
    }

    if (bestToken == 2) break; // "."
  }

  print("\nFinal Result: ${generated.map((id) => itos[id]).join(" ")}");
}

int argMax(List<double> list) {
  int maxIdx = 0;
  double maxVal = -double.infinity;
  for (int i = 0; i < list.length; i++) {
    if (list[i] > maxVal) {
      maxVal = list[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}
