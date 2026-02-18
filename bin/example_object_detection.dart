import 'package:dart_cuda/gpu_tensor.dart';
import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/vit_object_detector.dart';

void main() {
  print("--- 🚀 GPU ViT-based Multi-Object Detection (Chain-Linked) ---");

  // --- 1. Configuration ---
  const imageSize = 32;
  const patchSize = 8;
  const numChannels = 3;
  const embedSize = 64;
  const numClasses = 5;
  const numQueries = 3;
  const double lr = 0.001;

  final detector = ViTObjectDetector(
    imageSize: imageSize,
    patchSize: patchSize,
    numChannels: numChannels,
    embedSize: embedSize,
    numLayers: 2,
    numHeads: 4,
    numClasses: numClasses,
    numQueries: numQueries,
  );

  final optimizer = Adam(detector.parameters(), lr: lr);

  // --- 2. Data Preparation ---
  final numPatches = (imageSize ~/ patchSize) * (imageSize ~/ patchSize);
  final patchPixels = patchSize * patchSize * numChannels;

  // Input Tensor
  final xInput = Tensor.random([numPatches, patchPixels]);

  // Ground Truth Tensors
  final List<int> gtClassIdList = [1, 2, numClasses]; // 5 is background
  final List<double> gtBoxesRaw = [
    0.1,
    0.1,
    0.2,
    0.2,
    0.5,
    0.5,
    0.3,
    0.3,
    0.0,
    0.0,
    0.0,
    0.0,
  ];
  final gtBoxes = Tensor.fromList([numQueries, 4], gtBoxesRaw);

  // --- 3. Training Loop ---
  print("Training for 200 epochs... (Watch the loss drop now!)");

  for (int epoch = 0; epoch <= 200; epoch++) {
    // Only track tensors that are unique math results
    List<Tensor> tracker = [];
    optimizer.zeroGrad();

    // 1. Forward Pass
    final predictions = detector.forward(xInput, tracker);

    // 2. RESHAPE
    // IMPORTANT: Do NOT add these to the tracker.
    // They share handles with the tensors already inside the tracker (via forward).
    final logits = predictions['logits']!.reshape([numQueries, numClasses + 1]);
    final boxes = predictions['boxes']!.reshape([numQueries, 4]);

    // 3. GPU-Native Loss Calculation
    // Every operation below creates a NEW handle that MUST be disposed.
    final classLoss = logits.crossEntropy(gtClassIdList);
    tracker.add(classLoss);

    final diff = boxes - gtBoxes;
    tracker.add(diff);

    final squaredDiff = diff * diff;
    tracker.add(squaredDiff);

    // Create the scalar multiplier
    final scale = Tensor.fill(squaredDiff.shape, 0.25);
    tracker.add(scale);

    final boxLoss = squaredDiff * scale;
    tracker.add(boxLoss);

    final totalLoss = classLoss + boxLoss;
    tracker.add(totalLoss);

    // 4. Backward & Step
    totalLoss.backward();
    optimizer.step();

    if (epoch % 20 == 0) {
      final lossVal = totalLoss.fetchData()[0];
      print("Epoch $epoch | GPU Loss: ${lossVal.toStringAsFixed(6)}");
    }

    // 5. SAFE Cleanup
    // We dispose everything in the tracker.
    // Since we didn't add the reshaped 'logits' or 'boxes', we avoid the double free.
    for (var t in tracker) {
      t.dispose();
    }
  }

  print("\n✅ Training Complete. The model has now actually learned.");

  // Clean up persistent tensors
  gtBoxes.dispose();
  xInput.dispose();
  optimizer.dispose();
}
