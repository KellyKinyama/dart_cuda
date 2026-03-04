import '../aft_muzero_transformer_decoder.dart';
import '../gpu_tensor.dart';
import '../adam.dart';
import 'mu_zero_greedy_agent2.dart';

void main() {
  print("--- MuZero-GPT: 'Quick Brown Fox' Stability Build ---");

  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    "the": 2,
    "quick": 3,
    "brown": 4,
    "fox": 5,
    ".": 6,
    "<start>": 7,
    "jumps": 8,
    "over": 9,
    "lazy": 10,
    "dog": 11,
  };
  final Map<int, String> itos = stoi.map((k, v) => MapEntry(v, k));

  final model = TransformerDecoder(
    vocabSize: 32,
    embedSize: 64,
    encoderEmbedSize: 64,
    blockSize: 16,
    numLayers: 2,
    numHeads: 4,
  );

  // lmHead bias zeroing to prevent index-collapse
  // model.lmHead.zeroBias();

  final agent = MuZeroGreedyAgent(model, 64);

  final List<int> trainingData = [
    stoi["<start>"]!,
    stoi["the"]!,
    stoi["quick"]!,
    stoi["brown"]!,
    stoi["fox"]!,
    stoi["jumps"]!,
    stoi["over"]!,
    stoi["the"]!,
    stoi["lazy"]!,
    stoi["dog"]!,
    stoi["."]!,
  ];

  // 1000 Epochs for deep convergence
  trainMuZero(agent, trainingData, 1000, stoi);

  print("\n--- Running Pure Dynamics Generation ---");
  generateMuZeroPure(agent, [stoi["<start>"]!], 12, itos, stoi);
}

void trainMuZero(
  MuZeroGreedyAgent agent,
  List<int> targetSequence,
  int epochs,
  Map<String, int> stoi,
) {
  final optimizer = Adam(agent.model.parameters(), lr: 0.001);

  for (int epoch = 0; epoch <= epochs; epoch++) {
    optimizer.zeroGrad();
    double totalEpochLoss = 0.0;
    final List<Tensor> tracker = [];

    // Alternating modes to keep gradients stable on custom CUDA engine
    bool isPolicyMode = (epoch % 3 == 0);

    Tensor currentState = agent.representation([targetSequence[0]], tracker);

    for (int t = 0; t < targetSequence.length - 1; t++) {
      final int target = targetSequence[t + 1];

      if (isPolicyMode) {
        // Mode A: Train Representation + Policy Head
        final Tensor logits = agent.predictPolicy(currentState, tracker);
        final Tensor pLoss = logits.crossEntropy([target]);
        pLoss.backward();
        totalEpochLoss += pLoss.fetchData()[0];

        // Teacher Forcing for Policy
        currentState = agent.representation(
          targetSequence.sublist(0, t + 2),
          tracker,
        );
      } else {
        // Mode B: Train Dynamics (Imagination)
        // We use t+1 as the position index to align with the target word's position
        Tensor nextState = agent.dynamics(
          currentState,
          targetSequence[t],
          t + 1,
          tracker,
        );
        final Tensor iLogits = agent.predictPolicy(nextState, tracker);
        final Tensor iLoss = iLogits.crossEntropy([target]);

        iLoss.backward();
        totalEpochLoss += iLoss.fetchData()[0];

        // Recurrent Handover (No Teacher Forcing)
        currentState = nextState.detach();
      }
    }

    optimizer.step();
    for (final t in tracker) t.dispose();
    tracker.clear();

    if (epoch % 100 == 0) {
      String mode = isPolicyMode ? "POLICY  " : "DYNAMICS";
      print(
        "Epoch ${epoch.toString().padLeft(4)} | $mode | Loss: ${totalEpochLoss.toStringAsFixed(6)}",
      );
    }
  }
}

void generateMuZeroPure(
  MuZeroGreedyAgent agent,
  List<int> prompt,
  int maxLength,
  Map<int, String> itos,
  Map<String, int> stoi,
) {
  final List<int> generated = List.from(prompt);
  final List<Tensor> initTracker = [];

  Tensor rawInitState = agent.representation(prompt, initTracker);
  Tensor currentState = rawInitState.detach();
  for (var t in initTracker) t.dispose();

  for (int i = 0; i < maxLength; i++) {
    final List<Tensor> stepTracker = [];

    // 1. Predict
    final Tensor logits = agent.predictPolicy(currentState, stepTracker);
    final List<double> data = logits.fetchData();

    // 2. Loop-Breaker: Repetition Penalty for Step 1
    // If we just said a word, slightly discourage saying it again immediately
    if (generated.isNotEmpty) {
      data[generated.last] -= 2.0;
    }

    final int bestToken = argMax(data);
    generated.add(bestToken);
    print("Step ${i.toString().padLeft(2)} -> ${itos[bestToken]}");

    if (bestToken == stoi["."]!) break;

    // 3. Dynamics (The "Imagination" Step)
    // Pass i + 1 to ensure the Positional Embedding shifts forward
    Tensor nextStateRaw = agent.dynamics(
      currentState,
      bestToken,
      i + 1,
      stepTracker,
    );
    Tensor nextState = nextStateRaw.detach();

    currentState.dispose();
    for (var t in stepTracker) t.dispose();
    currentState = nextState;
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
