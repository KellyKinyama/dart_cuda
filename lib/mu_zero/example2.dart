import '../aft_muzero_transformer_decoder.dart';
import '../gpu_tensor.dart';
import '../adam.dart';
import 'muzero_greedy_agent.dart';

void main() {
  print("--- Stable Tensor-Engine MuZero-GPT: Training & Generation ---");

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
    vocabSize: 12,
    embedSize: 64, // Increased slightly for better latent separation
    encoderEmbedSize: 64,
    blockSize: 16,
    numLayers: 2,
    numHeads: 4,
  );

  final agent = MuZeroGreedyAgent(model, 64);

  // The full sequence to learn
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

  // --- TRAIN ---
  // We use 600 epochs to give the alternating modes enough time to co-evolve
  trainMuZero(agent, trainingData, 1000);

  print("\n--- Generation After Training ---");
  // We start with "<start>" and let the Dynamics head imagine the rest
  generateMuZeroPure(agent, [stoi["<start>"]!], 12, itos);
}

/// Training with Unrolled Dynamics to prevent "Latent Collapse"
void trainMuZero(
  MuZeroGreedyAgent agent,
  List<int> targetSequence,
  int epochs,
) {
  final optimizer = Adam(agent.model.parameters(), lr: 0.001);

  for (int epoch = 0; epoch <= epochs; epoch++) {
    optimizer.zeroGrad();
    double totalEpochLoss = 0.0;
    final List<Tensor> tracker = [];

    // Mode switching: Policy trains Representation/Head, Dynamics trains Transitions
    bool isPolicyMode = (epoch % 3 == 0);

    // Initial State
    Tensor currentState = agent.representation([targetSequence[0]], tracker);

    for (int t = 0; t < targetSequence.length - 1; t++) {
      final int target = targetSequence[t + 1];

      if (isPolicyMode) {
        // Mode A: Policy (f) + Representation (h)
        final Tensor logits = agent.predictPolicy(currentState, tracker);
        final Tensor pLoss = logits.crossEntropy([target]);
        pLoss.backward();
        totalEpochLoss += pLoss.fetchData()[0];

        // Teacher Forcing: Reset to ground truth for Policy training
        currentState = agent.representation(
          targetSequence.sublist(0, t + 2),
          tracker,
        );
      } else {
        // Mode B: Dynamics (g) - UNROLLED
        // We force g() to produce a state that f() recognizes as the next word
        final Tensor nextState = agent.dynamics(
          currentState,
          targetSequence[t],
          t + 1,
          tracker,
        );
        final Tensor iLogits = agent.predictPolicy(nextState, tracker);
        final Tensor iLoss = iLogits.crossEntropy([target]);

        iLoss.backward();
        totalEpochLoss += iLoss.fetchData()[0];

        // NO TEACHER FORCING: Use imagined state for the next step of the loop
        // We detach to keep the CUDA gradient chain from getting too deep
        currentState = nextState.detach();
      }
    }

    optimizer.step();
    for (final t in tracker) t.dispose();
    tracker.clear();

    if (epoch % 50 == 0) {
      String mode = isPolicyMode ? "POLICY  " : "DYNAMICS";
      print(
        "Epoch ${epoch.toString().padLeft(3)} | $mode | Loss: ${totalEpochLoss.toStringAsFixed(6)}",
      );
    }
  }
}

/// Pure MuZero Generation: Only uses Representation ONCE, then relies on Dynamics
void generateMuZeroPure(
  MuZeroGreedyAgent agent,
  List<int> prompt,
  int maxLength,
  Map<int, String> itos,
) {
  final List<int> generated = List.from(prompt);
  print("--- Running Pure Dynamics Generation (True MuZero Test) ---");

  final List<Tensor> initTracker = [];
  // 1. Initial Representation (h)
  final Tensor rawInitState = agent.representation(prompt, initTracker);
  Tensor currentState = rawInitState.detach(); // Ownership transfer

  for (var t in initTracker) t.dispose();

  for (int i = 0; i < maxLength; i++) {
    final List<Tensor> stepTracker = [];

    // 2. Predict: p = f(s)
    final Tensor logits = agent.predictPolicy(currentState, stepTracker);

    // 3. Select Action
    final int bestToken = argMax(logits.fetchData());
    generated.add(bestToken);
    print("Step ${i.toString().padLeft(2)} -> ${itos[bestToken]}");

    if (bestToken == 6) break; // "." is the stop token

    // 4. Transition: s = g(s, a)
    Tensor nextStateRaw = agent.dynamics(
      currentState,
      bestToken,
      i + 1,
      stepTracker,
    );
    Tensor nextState = nextStateRaw.detach();

    // 5. Cleanup current step
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
