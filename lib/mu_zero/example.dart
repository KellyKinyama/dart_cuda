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
  trainMuZero(agent, trainingData, 500); // 500 epochs to actually learn

  print("\n--- Generation After Training ---");
  generateMuZeroGreedy(agent, [stoi["<start>"]!], 4, itos);
}

void trainMuZero(
  MuZeroGreedyAgent agent,
  List<int> targetSequence,
  int epochs,
) {
  final optimizer = Adam(agent.model.parameters(), lr: 0.01);
  final List<Tensor> tracker = [];

  for (int epoch = 0; epoch <= epochs; epoch++) {
    optimizer.zeroGrad();
    double totalEpochLoss = 0;

    // Start state
    Tensor currentState = agent.representation([targetSequence[0]], tracker);

    for (int t = 0; t < targetSequence.length - 1; t++) {
      int target = targetSequence[t + 1];

      Tensor logits = agent.predictPolicy(currentState, tracker);
      final loss = logits.crossEntropy([target]);

      // 1. Backward must happen while logits and currentState are ALIVE
      loss.backward();

      totalEpochLoss += loss.fetchData()[0];

      // 2. Dynamics
      Tensor nextState = agent.dynamics(currentState, target, t + 1, tracker);
      currentState = nextState;
    }

    // 3. THE STEP must happen BEFORE the tracker disposal
    optimizer.step();

    // 4. NOW it is safe to dispose
    for (var t in tracker) {
      t.dispose();
    }
    tracker.clear();

    if (epoch % 50 == 0) {
      // Debug: Print first 3 weights of lmHead to see if they are changing
      var wSample = agent.model.lmHead.parameters()[0].fetchData().sublist(
        0,
        3,
      );
      print(
        "Epoch ${epoch.toString().padLeft(3)} | Loss: ${totalEpochLoss.toStringAsFixed(6)} | Weights: $wSample",
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
  List<Tensor> tracker = [];
  List<int> generated = List.from(prompt);
  Tensor currentState = agent.representation(prompt, tracker);

  for (int i = 0; i < maxLength; i++) {
    Tensor logits = agent.predictPolicy(currentState, tracker);
    int bestToken = argMax(logits.fetchData());
    generated.add(bestToken);

    print("Step $i -> ${itos[bestToken]}");

    // Compute next before destroying current
    Tensor nextState = agent.dynamics(currentState, bestToken, i + 1, tracker);

    // Clean up step T
    logits.dispose();
    if (i > 0) currentState.dispose();

    // Wipe intermediate math from tracker but KEEP the weights
    // (Note: ensure your agent weights aren't in the tracker!)
    for (var t in tracker) t.dispose();
    tracker.clear();

    currentState = nextState;
    if (bestToken == 2) break;
  }
  currentState.dispose();
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
