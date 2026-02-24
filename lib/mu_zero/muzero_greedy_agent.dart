import 'dart:math' as math;
import '../aft_muzero_transformer_decoder.dart';
import '../gpu_tensor.dart';

class MuZeroGreedyAgent {
  final TransformerDecoder model;
  final int embedSize;
  final Tensor dummyEnc;

  // MuZero specific heads
  late Tensor vW1, vW2, rewardW;

  MuZeroGreedyAgent(this.model, this.embedSize)
    : dummyEnc = Tensor.zeros([1, embedSize]) {
    vW1 = Tensor.random([embedSize, 64]);
    vW2 = Tensor.random([64, 1]);
    rewardW = Tensor.random([embedSize, 1]);
  }

  /// Tanh approximation: tanh(x) = 2 * sigmoid(2x) - 1
  Tensor _tanh(Tensor x, List<Tensor> tracker) {
    final x2 = x * 2.0;
    final sig = x2.sigmoid();
    final res = (sig * 2.0) - 1.0;
    tracker.addAll([x2, sig]);
    return res;
  }

  // --- h(o): REPRESENTATION ---
  Tensor representation(List<int> tokens, List<Tensor> tracker) {
    return model.getLatentState(tokens, dummyEnc, tracker);
  }

  // --- g(s, a): DYNAMICS ---
  // Tensor dynamics(Tensor state, int action, List<Tensor> tracker) {
  //   // Get embedding for the token
  //   final actionEmb = model.wte.getRow(action);
  //   tracker.add(actionEmb);

  //   // Transition: nextState = first transformer block forward
  //   // We treat the current latent state as the input to the next "time step"
  //   final combined = state + actionEmb;
  //   tracker.add(combined);

  //   return model.blocks[0].forward(combined, dummyEnc, tracker);
  // }

  Tensor dynamics(Tensor state, int action, int step, List<Tensor> tracker) {
    final actionEmb = model.wte.getRow(action);
    final posEmb = model.wpe.getRow(step % model.blockSize);

    final combined = state + actionEmb + posEmb;
    final nextState = model.blocks[0].forward(combined, dummyEnc, tracker);

    // Ensure intermediate results are tracked so they aren't cleaned up too early
    tracker.addAll([actionEmb, posEmb, combined]);
    tracker.add(nextState);

    return nextState;
  }

  // --- f(s): PREDICTION ---
  Tensor predictPolicy(Tensor state, List<Tensor> tracker) {
    return model.lmHead.forward(state, tracker);
  }

  Tensor predictValue(Tensor state, List<Tensor> tracker) {
    final h = (state.matmul(vW1)).relu();
    final rawV = h.matmul(vW2);
    final val = _tanh(rawV, tracker);
    tracker.addAll([h, rawV]);
    return val;
  }
}

// --- GENERATION LOOP ---

// void generateMuZeroGreedy(
//   MuZeroGreedyAgent agent,
//   List<int> prompt,
//   int maxLength,
// ) {
//   List<Tensor> tracker = [];
//   List<int> generated = List.from(prompt);

//   // 1. Initial latent state s0
//   Tensor currentState = agent.representation(prompt, tracker);

//   print("Prompt: $prompt");

//   for (int i = 0; i < maxLength; i++) {
//     // 2. Predict next move (Policy)
//     Tensor logits = agent.predictPolicy(currentState, tracker);

//     // 3. Argmax selection from GPU data
//     final data = logits.fetchData();
//     int bestToken = 0;
//     double maxVal = -double.infinity;
//     for (int j = 0; j < data.length; j++) {
//       if (data[j] > maxVal) {
//         maxVal = data[j];
//         bestToken = j;
//       }
//     }

//     generated.add(bestToken);
//     print("Token $i: $bestToken");

//     // 4. Update latent state (Dynamics)
//     // We imagine the state moving forward with the token we just chose
//     Tensor nextState = agent.dynamics(currentState, bestToken, tracker);

//     // 5. Cleanup GPU memory for this step
//     // Disposing currentState only if it's not a view of a tracked tensor
//     for (var t in tracker) t.dispose();
//     tracker.clear();
//     logits.dispose();

//     currentState = nextState;

//     if (bestToken == 2) break; // Stop at "."
//   }

//   currentState.dispose();
//   print("Final sequence: $generated");
// }

// void main() {
//   // 1. Vocabulary setup
//   final Map<String, int> stoi = {"hello": 0, "world": 1, ".": 2, "<start>": 3};

//   // 2. Initialize the Transformer and Agent
//   final model = TransformerDecoder(
//     vocabSize: 5,
//     embedSize: 32,
//     numLayers: 2,
//     blockSize: 16,
//   );

//   final agent = MuZeroGreedyAgent(model, 32);

//   // 3. Start Generation from "<start>"
//   print("--- MuZero Latent Generation ---");
//   generateMuZeroGreedy(agent, [stoi["<start>"]!], 3);
// }
