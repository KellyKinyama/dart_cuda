import '../adam.dart';
import '../gpu_tensor.dart';
import 'muzero_greedy_agent.dart';

void trainMuZero(
  MuZeroGreedyAgent agent,
  List<int> targetSequence,
  int epochs,
) {
  final optimizer = Adam(agent.model.parameters(), lr: 0.001);
  final List<Tensor> tracker = [];

  print("🚀 Starting MuZero Training Loop...");

  for (int epoch = 0; epoch <= epochs; epoch++) {
    optimizer.zeroGrad();
    double totalEpochLoss = 0;

    // 1. Initial Representation: s0 = h(start_token)
    // We start with the first token of our target sequence
    List<int> rootInput = [targetSequence[0]];
    Tensor currentState = agent.representation(rootInput, tracker);

    // 2. Unroll the sequence through the Dynamics model
    // This is "Backpropagation Through Time" (BPTT) in latent space
    for (int t = 0; t < targetSequence.length - 1; t++) {
      int actualNextToken = targetSequence[t + 1];

      // A. Predict Policy: p = f(s)
      Tensor logits = agent.predictPolicy(currentState, tracker);

      // B. Calculate Loss: CrossEntropy between predicted logits and actual next token
      // We wrap the target in a list as your crossEntropy expects List<int>
      final loss = logits.crossEntropy([actualNextToken]);
      totalEpochLoss += loss.fetchData()[0];

      // C. Backpropagate the loss for this step
      loss.backward();

      // D. Transition: s_next = g(s, actual_action)
      // We feed the CORRECT token back in to keep the "imagination" on track (Teacher Forcing)
      Tensor nextState = agent.dynamics(currentState, actualNextToken, tracker);

      // We don't dispose nextState yet as it's needed for the next loop iteration
      // but we cleanup the logits and loss handles
      loss.dispose();
      logits.dispose();

      currentState = nextState;
    }

    // 3. Update Weights
    optimizer.step();

    // 4. Memory Cleanup and Logging
    for (var t in tracker) t.dispose();
    tracker.clear();
    currentState.dispose();

    if (epoch % 100 == 0) {
      print("Epoch $epoch | Loss: ${totalEpochLoss.toStringAsFixed(6)}");
    }

    if (totalEpochLoss < 0.01) {
      print("✅ Converged at epoch $epoch");
      break;
    }
  }
}
