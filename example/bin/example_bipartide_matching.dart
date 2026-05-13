import 'dart:math' as math;
import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/utils/hungarian_algorithm.dart';
import 'package:dart_cuda/core/transformers/vision/vit_object_detector.dart';

void main() {
  print("--- 🚀 GPU ViT Object Detection (Vectorized Hungarian) ---");

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

  for (int epoch = 0; epoch <= 200; epoch++) {
    List<Tensor> tracker = [];
    optimizer.zeroGrad();

    // 1. Forward Pass
    final preds = detector.forward(xInput, tracker);
    final logits = preds['logits']!.reshape([numQueries, numClasses + 1]);
    final boxes = preds['boxes']!.reshape([numQueries, 4]);

    // 2. Hungarian Matching (CPU/GPU Hybrid)
    final List<double> flatGT = [];
    for (var obj in gtObjects)
      flatGT.addAll((obj['bbox'] as List).cast<double>());
    final gtBoxTensor = Tensor.fromList([gtObjects.length, 4], flatGT);

    final costMatrixTensor = boxes.computeCostMatrix(gtBoxTensor);
    final List<double> costs = costMatrixTensor.fetchData();

    final matrix = List.generate(
      numQueries,
      (q) => List.generate(numQueries, (g) {
        if (g < gtObjects.length)
          return (costs[q * gtObjects.length + g] * 10000).toInt();
        return 1000000;
      }),
    );

    final assignments = HungarianAlgorithm(matrix).getAssignment();

    // Cleanup temporary matching tensors
    gtBoxTensor.dispose();
    costMatrixTensor.dispose();

    // 3. Reorder Targets based on Assignments
    // This aligns GT to the Queries so we can use vectorized math
    final List<int> alignedClassIds = List.filled(
      numQueries,
      numClasses,
    ); // Default to background
    final List<double> alignedBoxesRaw = List.filled(numQueries * 4, 0.0);

    for (int q = 0; q < numQueries; q++) {
      int gtIdx = assignments[q];
      if (gtIdx != -1 && gtIdx < gtObjects.length) {
        alignedClassIds[q] = gtObjects[gtIdx]['class_id'];
        List<double> bbox = (gtObjects[gtIdx]['bbox'] as List).cast<double>();
        for (int i = 0; i < 4; i++) alignedBoxesRaw[q * 4 + i] = bbox[i];
      }
    }

    final alignedGtBoxes = Tensor.fromList([numQueries, 4], alignedBoxesRaw);
    tracker.add(alignedGtBoxes);

    // 4. Vectorized Loss (The "Working" Way)
    // Class Loss (Native C++ call for the whole matrix)
    final classLoss = logits.crossEntropy(alignedClassIds);
    tracker.add(classLoss);

    // Box Loss (Native C++ matrix subtraction)
    final boxDiff = (boxes - alignedGtBoxes).abs();
    tracker.add(boxDiff);

    // Combine
    final totalLoss = classLoss + boxDiff;
    tracker.add(totalLoss);

    // 5. Backprop & Step
    totalLoss.backward();
    optimizer.step();

    if (epoch % 20 == 0) {
      final lossVal = totalLoss.fetchData()[0];
      print(
        "Epoch $epoch | Loss: ${lossVal.toStringAsFixed(6)} | Match: $assignments",
      );
    }

    // 6. Final Cleanup
    final Set<int> freed = {};
    for (var t in tracker) {
      final addr = t.handle.address;
      if (addr != 0 && !freed.contains(addr)) {
        t.dispose();
        freed.add(addr);
      }
    }
  }

  xInput.dispose();
  optimizer.dispose();
  print("✅ Training Complete.");
}
