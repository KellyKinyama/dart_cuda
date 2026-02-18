import 'dart:math' as math;

import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/hungarian_algorithm.dart';
import 'package:dart_cuda/vit_object_detector.dart';

void main() {
  print("--- 🚀 GPU ViT Object Detection with Hungarian Matching ---");

  final numQueries = 3;
  final numClasses = 5;
  final detector = ViTObjectDetector(
    imageSize: 32,
    patchSize: 8,
    numChannels: 3,
    embedSize: 64,
    numLayers: 2,
    numHeads: 4,
    numClasses: numClasses,
    numQueries: numQueries,
  );

  final optimizer = Adam(detector.parameters(), lr: 0.001);
  final xInput = Tensor.random([16, 192]);

  final List<Map<String, dynamic>> gtObjects = [
    {
      'bbox': [0.1, 0.1, 0.2, 0.2],
      'class_id': 1,
    },
    {
      'bbox': [0.5, 0.5, 0.3, 0.3],
      'class_id': 2,
    },
  ];

  for (int epoch = 0; epoch < 100; epoch++) {
    // 1. Tracker only for tensors that need to stay alive for .backward()
    List<Tensor> backpropTracker = [];
    optimizer.zeroGrad();

    // --- Forward Pass ---
    final preds = detector.forward(xInput, backpropTracker);

    // Reshape views (In your C++ engine, these usually point to same memory)
    final logits = preds['logits']!.reshape([numQueries, numClasses + 1]);
    final boxes = preds['boxes']!.reshape([numQueries, 4]);

    // --- 2. Cost Matrix Calculation ---
    final List<List<int>> costMatrix = List.generate(
      numQueries,
      (i) => List.filled(numQueries, 1000000),
    );

    for (int q = 0; q < numQueries; q++) {
      final qBox = boxes.getRow(q);
      final qLogits = logits.getRow(q);

      for (int g = 0; g < gtObjects.length; g++) {
        // GPU Box Math
        final gtBox = Tensor.fromList([1, 4], gtObjects[g]['bbox']);
        final diff = (qBox - gtBox).abs();
        double boxCost = diff.fetchData().reduce((a, b) => a + b);

        // GPU Class Math
        final sTensor = qLogits.softmax();
        final probs = sTensor.fetchData();
        double classCost = -math.log(probs[gtObjects[g]['class_id']] + 1e-8);

        costMatrix[q][g] = ((boxCost + classCost) * 10000).round();

        // CLEANUP IMMEDIATELY: These are not used for backprop
        gtBox.dispose();
        diff.dispose();
        sTensor.dispose();
      }
      // Rows are not needed anymore after g-loop
      qBox.dispose();
      qLogits.dispose();
    }

    // --- 3. Hungarian Matching ---
    final assignments = HungarianAlgorithm(costMatrix).getAssignment();

    // --- 4. Loss Accumulation (The Stable Way) ---
    Tensor? totalLoss;

    for (int q = 0; q < numQueries; q++) {
      int assignedGt = assignments[q];
      Tensor? currentStepLoss;

      if (assignedGt != -1 && assignedGt < gtObjects.length) {
        // Create components for this specific query
        final targetBox = Tensor.fromList([
          1,
          4,
        ], gtObjects[assignedGt]['bbox']);
        final qBoxRow = boxes.getRow(q);
        final qLogitsRow = logits.getRow(q);

        // Calculate losses
        final boxDiff = (qBoxRow - targetBox).abs();
        final classLoss = qLogitsRow.crossEntropy([
          gtObjects[assignedGt]['class_id'],
        ]);

        // Sum them up
        currentStepLoss = classLoss + boxDiff;

        // CLEANUP components immediately: We don't need these anymore,
        // they are now baked into currentStepLoss
        targetBox.dispose();
        qBoxRow.dispose();
        qLogitsRow.dispose();
        boxDiff.dispose();
        classLoss.dispose();
      } else {
        // Handle background class
        final qLogitsRow = logits.getRow(q);
        currentStepLoss = qLogitsRow.crossEntropy([numClasses]);
        qLogitsRow.dispose();
      }

      // --- CRITICAL: Accumulate into totalLoss without leaking ---
      if (totalLoss == null) {
        totalLoss = currentStepLoss;
      } else {
        final oldTotal = totalLoss;
        totalLoss = oldTotal + currentStepLoss;

        // Dispose the previous sum and the branch result
        // This keeps the GPU memory flat
        oldTotal.dispose();
        currentStepLoss.dispose();
      }
    }

    // --- 5. Optimize ---
    if (totalLoss != null) {
      totalLoss.backward();
      optimizer.step();

      if (epoch % 10 == 0) {
        print(
          "Epoch $epoch | Loss: ${totalLoss.fetchData()[0].toStringAsFixed(6)}",
        );
      }

      // The final totalLoss handle is destroyed here
      totalLoss.dispose();
    }

    // --- 5. Optimize ---
    if (totalLoss != null) {
      totalLoss.backward();
      optimizer.step();

      if (epoch % 10 == 0) {
        print(
          "Epoch $epoch | Loss: ${totalLoss.fetchData()[0].toStringAsFixed(6)}",
        );
      }
      totalLoss.dispose();
    }

    // --- 6. End of Epoch Cleanup ---
    for (var t in backpropTracker) {
      t.dispose();
    }
  }

  xInput.dispose();
  optimizer.dispose();
  print("Training Complete.");
}
