import 'dart:math' as math;
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/hungarian_algorithm.dart';
import 'package:dart_cuda/vit_object_detector.dart';

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
      'bbox': [0.10, 0.10, 0.20, 0.20],
      'class_id': 1,
    },
    {
      'bbox': [0.50, 0.50, 0.30, 0.30],
      'class_id': 2,
    },
  ];

  List<int> lastAssignments = [];

  for (int epoch = 0; epoch <= 200; epoch++) {
    List<Tensor> tracker = [];
    optimizer.zeroGrad();

    final preds = detector.forward(xInput, tracker);
    final logits = preds['logits']!.reshape([numQueries, numClasses + 1]);
    final boxes = preds['boxes']!.reshape([numQueries, 4]);

    // 1. Hungarian Matching
    final List<double> flatGT = [];
    for (var obj in gtObjects)
      flatGT.addAll((obj['bbox'] as List).cast<double>());
    final gtBoxTensor = Tensor.fromList([gtObjects.length, 4], flatGT);

    final costMatrixTensor = boxes.computeCostMatrix(gtBoxTensor);
    final List<double> costs = costMatrixTensor.fetchData();
    final assignments = HungarianAlgorithm(
      List.generate(
        numQueries,
        (q) => List.generate(numQueries, (g) {
          if (g < gtObjects.length)
            return (costs[q * gtObjects.length + g] * 10000).toInt();
          return 1000000;
        }),
      ),
    ).getAssignment();

    lastAssignments = assignments;
    gtBoxTensor.dispose();
    costMatrixTensor.dispose();

    // 2. Vectorized Loss
    final List<int> alignedClassIds = List.filled(numQueries, numClasses);
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

    final classLoss = logits.crossEntropy(alignedClassIds);
    final boxDiff = (boxes - alignedGtBoxes).abs();
    final totalLoss = classLoss + boxDiff;
    tracker.add(classLoss);
    tracker.add(boxDiff);
    tracker.add(totalLoss);

    totalLoss.backward();
    optimizer.step();

    if (epoch % 50 == 0 || epoch == 200) {
      print("\n--- Epoch $epoch ---");
      print("Loss: ${totalLoss.fetchData()[0].toStringAsFixed(6)}");
      _printDetectionTable(
        boxes.fetchData(),
        alignedBoxesRaw,
        alignedClassIds,
        numQueries,
      );
    }

    // Cleanup
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
  print("\n✅ Training Complete.");
}

void _printDetectionTable(
  List<double> pred,
  List<double> target,
  List<int> classes,
  int n,
) {
  print(
    "Query | Class | Predicted BBox             | Target BBox                | Error",
  );
  print(
    "---------------------------------------------------------------------------------",
  );
  for (int i = 0; i < n; i++) {
    final p = pred
        .sublist(i * 4, i * 4 + 4)
        .map((v) => v.toStringAsFixed(2))
        .toList();
    final t = target
        .sublist(i * 4, i * 4 + 4)
        .map((v) => v.toStringAsFixed(2))
        .toList();

    double error = 0;
    for (int j = 0; j < 4; j++)
      error += (pred[i * 4 + j] - target[i * 4 + j]).abs();

    final label = classes[i] == 5 ? "BG" : "ID:${classes[i]}";
    print("  #$i   |  $label  | $p | $t | ${error.toStringAsFixed(4)}");
  }
}
