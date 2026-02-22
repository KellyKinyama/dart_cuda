import 'dart:math' as math;

import '/gpu_tensor.dart';

Tensor crossEntropy(Tensor logits, List<int> targets, int vocabSize) {
  int numTokens = targets.length;
  double totalLoss = 0;

  for (int t = 0; t < numTokens; t++) {
    int target = targets[t];
    int offset = t * vocabSize;
    double maxL = -double.infinity;
    for (int v = 0; v < vocabSize; v++) {
      if (logits.data[offset + v] > maxL) maxL = logits.data[offset + v];
    }
    double sumExp = 0;
    for (int v = 0; v < vocabSize; v++) {
      sumExp += math.exp(logits.data[offset + v] - maxL);
    }
    totalLoss +=
        (maxL + math.log(sumExp + 1e-12) - logits.data[offset + target]);
  }

  final loss = Tensor([1], children: {logits});
  loss.data[0] = totalLoss / numTokens;

  loss.onBackward = () {
    double gradFromLoss = 1.0 / numTokens;
    for (int t = 0; t < numTokens; t++) {
      int target = targets[t];
      int offset = t * vocabSize;
      double maxL = -double.infinity;
      for (int v = 0; v < vocabSize; v++) {
        if (logits.data[offset + v] > maxL) maxL = logits.data[offset + v];
      }
      double sumExp = 0;
      for (int v = 0; v < vocabSize; v++) {
        sumExp += math.exp(logits.data[offset + v] - maxL);
      }
      for (int v = 0; v < vocabSize; v++) {
        double prob = math.exp(logits.data[offset + v] - maxL) / sumExp;
        logits.grad[offset + v] +=
            (prob - ((v == target) ? 1.0 : 0.0)) * gradFromLoss;
      }
    }
  };
  return loss;
}
