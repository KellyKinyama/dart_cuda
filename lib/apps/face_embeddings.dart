// file: lib/main_face_gpu.dart

import 'package:dart_cuda/network_utils.dart';

import '../adam.dart';
import '../aft_vit_face_embeding.dart';
import '../gpu_tensor.dart';
import '../triplet_loss.dart';
import 'triplet_loader.dart';

Future<void> main() async {
  print("🚀 GPU Face Recognition: Stable Training Loop");

  // --- 1. Model Configuration ---
  final model = ViTFaceEmbeddingGPU(
    imageSize: 32,
    patchSize: 8,
    embedSize: 64,

    outputDim: 64,
    numLayers: 2,
  );

  final lossFn = TripletLossGPU(margin: 0.2);

  // Cache parameters to pass to the cleanup function
  final modelParams = model.parameters();
  final optimizer = Adam(modelParams, lr: 0.0001);

  final imagesFolder = 'Original Images';

  final tripletLoader = TripletLoader(imagesFolder, 16);
  List<Tensor> tracker = [];
  print("🔥 Starting GPU Training Loop...");
  // print(tripletLoader._identityMap);
  for (int epoch = 0; epoch <= 100; epoch++) {
    final faceBatch = tripletLoader.nextBatch(32);

    // --- 2. Synthetic Data Initialization ---
    final anchor = faceBatch['anchor']!;
    final positive = faceBatch['positive']!;
    final negative = faceBatch['negative']!;

    // try {

    optimizer.zeroGrad();
    lossFn.zeroGrad();

    // --- 3. Forward Pass ---
    final embA = model.getFaceEmbedding(anchor, tracker);
    final embP = model.getFaceEmbedding(positive, tracker);
    final embN = model.getFaceEmbedding(negative, tracker);

    // --- 4. Loss Calculation ---
    final totalLoss = lossFn.forward(embA, embP, embN, tracker);

    // Fetch loss value for logging and stability check
    final lossVal = totalLoss.fetchData()[0];

    // --- 5. Backpropagation & Optimization ---
    // Stability Tip: Only backprop if loss is > 0
    // if (lossVal > 0) {
    totalLoss.backward();
    optimizer.step();
    // }

    if (epoch % 10 == 0) {
      print("Epoch $epoch | Triplet Loss: ${lossVal.toStringAsFixed(6)}");
      await saveModuleBinary(model, 'face_net.bin');
    }

    // --- 6. Safe Memory Cleanup ---
    // We pass both the tracker and the model parameters to ensure safety
    // _safeCleanup(tracker, totalLoss, modelParams);
    // totalLoss.dispose();
    // anchor.dispose();
    // positive.dispose();
    // negative.dispose();
    // optimizer.dispose();
  }
  // } catch (e) {
  //   print("Caught Exception: $e");
  // } finally {
  // --- 7. Final Resource Release ---
  // for (var fb in faceBatch.values) {
  //   fb.dispose();
  // }
  // _safeCleanup(tracker, totalLoss, modelParams);
  print("✅ Training Complete and Resources Freed.");
  // for (var fb in tracker) {
  //   fb.dispose();
  // }
  // }
}

/// Prevents Double-Disposal and Parameter Deletion
void _safeCleanup(
  List<Tensor> tracker,
  Tensor lossTensor,
  List<Tensor> params,
) {
  final freedAddresses = <int>{};

  // 1. Map current parameter addresses to protect them from disposal
  final paramAddresses = params.map((p) => p.handle.address).toSet();

  // 2. Clean the tracker
  for (var t in tracker) {
    final addr = t.handle.address;
    if (addr != 0 &&
        !freedAddresses.contains(addr) &&
        !paramAddresses.contains(addr) &&
        t.isView != true) {
      t.dispose();
      freedAddresses.add(addr);
    }
  }

  // 3. Explicitly clean the loss tensor if not already handled
  final lossAddr = lossTensor.handle.address;
  if (lossAddr != 0 &&
      !freedAddresses.contains(lossAddr) &&
      !paramAddresses.contains(lossAddr)) {
    lossTensor.dispose();
  }
}
