import 'dart:math' as math;
import 'package:dart_cuda/adam.dart';
import 'gpu_tensor.dart';
import 'nn.dart';

class AFTAttention extends Module {
  final Layer queryLayer, keyLayer, valueLayer;
  final Tensor posBias;
  final bool masked;

  AFTAttention(
    int embedSize,
    int headSize,
    int maxSeqLen, {
    this.masked = false,
  }) : queryLayer = Layer(embedSize, headSize, useGelu: false),
       keyLayer = Layer(embedSize, headSize, useGelu: false),
       valueLayer = Layer(embedSize, headSize, useGelu: false),
       posBias = Tensor.fromList([
         maxSeqLen,
         maxSeqLen,
       ], List.generate(maxSeqLen * maxSeqLen, (_) => 0.01));

  Tensor forward(Tensor x, List<Tensor> tracker) {
    // --- DEBUG: Input Check ---
    _debugTensor("Input X", x);

    final q = queryLayer.forward(x, tracker);
    final k = keyLayer.forward(x, tracker);
    final v = valueLayer.forward(x, tracker);

    // --- DEBUG: Projection Check ---
    // If these are huge (> 20), AFT will likely explode
    _debugTensor("Query (Q)", q);
    _debugTensor("Key (K)", k);

    final out = Tensor.aft(q, k, v, posBias, masked);

    // --- DEBUG: Output Check ---
    _debugTensor("AFT Output", out);

    tracker.add(out);
    return out;
  }

  /// Helper to print Tensor stats to catch NaNs early
  void _debugTensor(String name, Tensor t) {
    final data = t.fetchData();
    if (data.isEmpty) return;

    double maxVal = -double.infinity;
    double sum = 0;
    bool hasNaN = false;

    for (var v in data) {
      if (v.isNaN) hasNaN = true;
      if (v.abs() > maxVal) maxVal = v.abs();
      sum += v;
    }
    double mean = sum / data.length;

    // print(
    //   '  [DEBUG $name] MaxAbs: ${maxVal.toStringAsFixed(4)} | Mean: ${mean.toStringAsFixed(4)} ${hasNaN ? "🚨 NAN DETECTED!" : ""}',
    // );
  }

  @override
  List<Tensor> parameters() => [
    ...queryLayer.parameters(),
    ...keyLayer.parameters(),
    ...valueLayer.parameters(),
    posBias,
  ];
}

void main() {
  const int dim = 16;
  const int seqLen = 10;
  const int currentT = 5;
  const double lr = 0.001; // Reduced for stability

  // 1. Initialize
  final aft = AFTAttention(dim, 4, seqLen, masked: false);

  // --- CRITICAL: Stabilize Weights ---
  // AFT is extremely sensitive to large weights.
  // We force everything to small values to prevent the Epoch 0/1 explosion.
  for (var p in aft.parameters()) {
    final data = p.fetchData();
    final rand = math.Random();
    for (int i = 0; i < data.length; i++) {
      data[i] = (rand.nextDouble() * 2 - 1) * 0.02;
    }
    p.data = data;
  }

  final x = Tensor.random([currentT, dim]);
  final target = Tensor.fill([
    currentT,
    dim,
  ], 0.1); // Small target for stability
  List<Tensor> tracker = [];

  // Use tighter gradClip in Adam
  final optimizer = Adam(aft.parameters(), lr: lr, gradClip: 0.1);

  // print('--- AFT Debug Training Run ---');

  for (int step = 0; step < 10; step++) {
    // print('\nSTEP $step');
    optimizer.zeroGrad();

    final output = aft.forward(x, tracker);
    final loss = output.mseLoss(target);

    final lVal = loss.fetchData()[0];
    // print('Current Loss: ${lVal.toStringAsFixed(6)}');

    if (lVal.isNaN) {
      print("🛑 Training aborted: Loss is NaN.");
      break;
    }

    loss.backward();

    // DEBUG: Check Gradients
    _checkGradients(aft.parameters());

    optimizer.step();

    // Memory Cleanup
    for (var t in tracker) t.dispose();
    tracker.clear();
    loss.dispose();
    output.dispose();
  }
}

void _checkGradients(List<Tensor> params) {
  double maxGrad = 0;
  for (var p in params) {
    if (p.grad != null) {
      for (var g in p.grad!) {
        if (g.abs() > maxGrad) maxGrad = g.abs();
      }
    }
  }
  // print('  [GRAD CHECK] Max Gradient Magnitude: ${maxGrad.toStringAsFixed(6)}');
}
