import '../adam.dart';
import '../aft_vit_face_embeding.dart';
import '../gpu_tensor.dart';
import '../triplet_loss.dart';
import 'triplet_loader2.dart';

Future<void> main() async {
  print("🚀 GPU Face Recognition: Stable Training Loop");

  // --- 1. Model Configuration ---
  final int imgSize = 32;
  final int batchSize = 32;
  final int features = imgSize * imgSize * 3;

  final model = ViTFaceEmbeddingGPU(
    imageSize: imgSize,
    patchSize: 8,
    embedSize: 64,
    outputDim: 64,
    numLayers: 2,
  );

  final lossFn = TripletLossGPU(margin: 0.2);
  final modelParams = model.parameters();
  final optimizer = Adam(modelParams, lr: 0.0001);

  // Load dataset into RAM (Float32List cached on CPU)
  final tripletLoader = TripletLoader('Original Images', imgSize, 5);

  print("🔥 Starting GPU Training Loop...");

  for (int epoch = 0; epoch <= 1000; epoch++) {
    List<Tensor> tracker = [];

    // 2. Fetch from RAM (CPU)
    final faceBatch = tripletLoader.nextBatch(batchSize);

    // 3. UPLOAD TO GPU TENSORS
    // We create these manually so we can dispose them manually
    final anchor = Tensor.fromList([batchSize, features], faceBatch['anchor']!);
    final positive = Tensor.fromList([
      batchSize,
      features,
    ], faceBatch['positive']!);
    final negative = Tensor.fromList([
      batchSize,
      features,
    ], faceBatch['negative']!);

    optimizer.zeroGrad();

    // 4. Forward Pass (Generates 64-dim embeddings)
    final embA = model.getFaceEmbedding(anchor, tracker);
    final embP = model.getFaceEmbedding(positive, tracker);
    final embN = model.getFaceEmbedding(negative, tracker);

    // 5. Loss Calculation
    final totalLoss = lossFn.forward(embA, embP, embN, tracker);
    final lossVal = totalLoss.fetchData()[0];

    // 6. Backward Pass
    totalLoss.backward();
    optimizer.step();

    if (epoch % 10 == 0) {
      print("Epoch $epoch | Triplet Loss: ${lossVal.toStringAsFixed(6)}");
    }

    // --- 7. CRITICAL CLEANUP (Small GPU Protection) ---
    // Cleanup the intermediate ViT tensors (attention maps, projections, etc.)
    _safeCleanup(tracker, totalLoss, modelParams);

    // Explicitly dispose of the input batch tensors to free VRAM immediately
    // anchor.dispose();
    // positive.dispose();
    // negative.dispose();

    tracker.clear();
  }

  print("✅ Training Complete.");
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
