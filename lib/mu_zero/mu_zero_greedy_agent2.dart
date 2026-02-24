import '../aft_muzero_transformer_decoder.dart';
import '../gpu_tensor.dart';

class MuZeroGreedyAgent {
  final TransformerDecoder model;
  final int embedSize;
  final Tensor dummyEnc;

  // MuZero specific heads for Value and Reward (Standard MuZero)
  late Tensor vW1, vW2, rewardW;

  MuZeroGreedyAgent(this.model, this.embedSize)
    : dummyEnc = Tensor.zeros([1, embedSize]) {
    // Initializing auxiliary heads
    vW1 = Tensor.random([embedSize, 64]);
    vW2 = Tensor.random([64, 1]);
    rewardW = Tensor.random([embedSize, 1]);
  }

  /// Tanh approximation for value clipping
  Tensor _tanh(Tensor x, List<Tensor> tracker) {
    final x2 = x * 2.0;
    final sig = x2.sigmoid();
    final res = (sig * 2.0) - 1.0;
    tracker.addAll([x2, sig]);
    return res;
  }

  // --- h(o): REPRESENTATION ---
  // Encodes the actual observed tokens into the first latent state
  Tensor representation(List<int> tokens, List<Tensor> tracker) {
    return model.getLatentState(tokens, dummyEnc, tracker);
  }

  // --- g(s, a): DYNAMICS ---
  // Transforms the current state into the next state based on an action
  Tensor dynamics(Tensor state, int action, int step, List<Tensor> tracker) {
    // 1. Get the 'meaning' of the action and the 'meaning' of the new position
    final actionEmb = model.wte.getRow(action);
    final posEmb = model.wpe.getRow(step % model.blockSize);

    // 2. 🔥 THE FIX: Action Amplification
    // We multiply the action by 3.0 to ensure the transformer block
    // notices that the "context" has changed significantly.
    final strongAction = actionEmb * 3.0;

    // 3. Combine current memory with the new action and position
    final stateWithAction = state + strongAction;
    final combined = stateWithAction + posEmb;

    // 4. Processing through a Transformer Block
    // This allows the model to 're-map' the state so it is compatible
    // with the predictPolicy (lmHead) function.
    final nextState = model.blocks[0].forward(combined, dummyEnc, tracker);

    // Track every step for the CUDA gradient chain
    tracker.addAll([
      actionEmb,
      strongAction,
      posEmb,
      stateWithAction,
      combined,
      nextState,
    ]);

    return nextState;
  }

  // --- f(s): PREDICTION ---
  // Predicts token probabilities from the latent state
  Tensor predictPolicy(Tensor state, List<Tensor> tracker) {
    return model.lmHead.forward(state, tracker);
  }

  // Predicts the expected future value from the current state
  Tensor predictValue(Tensor state, List<Tensor> tracker) {
    final h = (state.matmul(vW1)).relu();
    final rawV = h.matmul(vW2);
    final val = _tanh(rawV, tracker);
    tracker.addAll([h, rawV]);
    return val;
  }
}
