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
    List<Tensor> tracker = [];
    optimizer.zeroGrad();

    // 1. Forward Pass
    final predictions = detector.forward(xInput, tracker);

    // 2. RESHAPE (The Game Changer)
    // This keeps the GPU handles connected to the model weights
    final logits = predictions['logits']!.reshape([numQueries, numClasses + 1]);
    final boxes = predictions['boxes']!.reshape([numQueries, 4]);
    tracker.addAll([logits, boxes]);

    // 3. GPU-Native Loss Calculation
    // Classification Loss (Batch CrossEntropy)
    final classLoss = logits.crossEntropy(gtClassIdList);
    tracker.add(classLoss);

    // Box Loss (MSE on GPU)
    final diff = boxes - gtBoxes;
    tracker.add(diff);
    final squaredDiff = diff * diff;
    tracker.add(squaredDiff);

    // Scale the box loss on the GPU (simulating a mean)
    final boxLoss = squaredDiff * Tensor.fill(squaredDiff.shape, 0.25);
    tracker.add(boxLoss);

    // Combine Losses
    final totalLoss = classLoss + boxLoss;
    tracker.add(totalLoss);

    // 4. Backward & Step
    totalLoss.backward();
    optimizer.step();

    if (epoch % 20 == 0) {
      final lossVal = totalLoss.fetchData()[0];
      print("Epoch $epoch | GPU Loss: ${lossVal.toStringAsFixed(6)}");
    }

    // 5. Cleanup intermediates
    for (var t in tracker) {
      // Avoid disposing the tensors returned directly by forward if they are params
      t.dispose();
    }
  }

  print("\n✅ Training Complete. The model has now actually learned.");

  // Clean up persistent tensors
  gtBoxes.dispose();
  xInput.dispose();
  optimizer.dispose();
}
