import 'gpu_tensor.dart';
import 'nn.dart';

class MLP extends Module {
  final List<Layer> layers = [];

  MLP(int nin, List<int> nouts) {
    List<int> dims = [nin, ...nouts];
    for (int i = 0; i < nouts.length; i++) {
      layers.add(Layer(dims[i], dims[i + 1], useGelu: i != nouts.length - 1));
    }
  }

  Tensor forward(Tensor x, List<Tensor> tracker) {
    Tensor current = x;
    for (var layer in layers) {
      current = layer.forward(current, tracker);
    }
    return current;
  }

  @override
  List<Tensor> parameters() => layers.expand((l) => l.parameters()).toList();
}

void main() {
  // 1. Setup Network
  final model = MLP(2, [4, 1]);
  final double learningRate = 0.5; // High LR for XOR

  // 2. Data (XOR)
  final xData = Tensor.fromList([4, 2], [0,0, 0,1, 1,0, 1,1]);
  final target = Tensor.fromList([4, 1], [0, 1, 1, 0]);

  

  print('Starting GPU MLP Training...');

  for (int epoch = 0; epoch <= 1000; epoch++) {
    // This list tracks every tensor created in this iteration
    List<Tensor> tracker = [];

    // Forward
    final pred = model.forward(xData, tracker);
    
    // Loss = (pred - target)^2
    final diff = pred - target;
    final loss = diff.pow(2.0);
    tracker.addAll([diff, loss]);

    // Backward
    model.zeroGrad();
    loss.backward();

    // Update (On GPU)
    model.step(learningRate);

    if (epoch % 100 == 0) {
      print("Epoch $epoch, Loss: ${loss.data[0].toStringAsFixed(6)}");
    }

    // --- CRITICAL: GPU MEMORY CLEANUP ---
    // Dispose all intermediate tensors created this epoch
    for (var t in tracker) {
      t.dispose();
    }
  }

  // Final Test
  List<Tensor> dummy = [];
  final finalPred = model.forward(xData, dummy);
  print("\nFinal Results:");
  print(finalPred.data);
  
  // Final cleanup
  finalPred.dispose();
  xData.dispose();
  target.dispose();
}