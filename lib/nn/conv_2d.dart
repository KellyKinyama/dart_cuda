import '../gpu_tensor.dart';
import '../nn.dart';

class Conv2d {
  final int inChannels;
  final int outChannels;
  final int kernelSize;
  final int stride;
  final int padding;

  late Tensor weight;
  late Tensor bias;

  Conv2d({
    required this.inChannels,
    required this.outChannels,
    required this.kernelSize,
    this.stride = 1,
    this.padding = 0,
  }) {
    int patchSize = inChannels * kernelSize * kernelSize;

    // Using fromList for initial weights (Xavier-like initialization)
    final initialWeights = List.generate(outChannels * patchSize, (i) => 0.1);
    weight = Tensor.fromList([outChannels, patchSize], initialWeights);

    // Using fromList for bias [1, outChannels]
    final initialBias = List.generate(outChannels, (i) => 0.0);
    bias = Tensor.fromList([1, outChannels], initialBias);
  }

  Tensor forward(Tensor input, List<Tensor> tracker) {
    int hIn = input.shape[1];
    int wIn = input.shape[2];
    int hOut = ((hIn + 2 * padding - kernelSize) ~/ stride) + 1;
    int wOut = ((wIn + 2 * padding - kernelSize) ~/ stride) + 1;

    int patchSize = inChannels * kernelSize * kernelSize;
    int numOutputs = hOut * wOut;

    // 1. Create ColBuffer. It is [patchSize, numOutputs].
    Tensor colBuffer = Tensor.fill([patchSize, numOutputs], 0.0);
    tracker.add(colBuffer);

    // 2. im2col
    engine.im2col(
      input.handle,
      inChannels,
      hIn,
      wIn,
      kernelSize,
      kernelSize,
      padding,
      padding,
      stride,
      stride,
      colBuffer.handle,
    );

    // 3. MatMul
    // weight is [outChannels, patchSize], colBuffer is [patchSize, numOutputs]
    // result is [outChannels, numOutputs]
    Tensor res = weight.matmul(colBuffer);
    tracker.add(res);

    // 4. Bias Addition
    // This produces a new Tensor.
    // IMPORTANT: Since you can't reassign shape later, the bias addition
    // must return a Tensor that we can interpret as [outChannels, hOut, wOut].
    Tensor output = res + bias;
    tracker.add(output);

    return output;
  }

  List<Tensor> parameters() => [weight, bias];

  void dispose() {
    weight.dispose();
    bias.dispose();
  }
}

void main() {
  print("🚀 Starting Final GPU Conv2d Validation...");

  // 1. Setup Input: 1 channel, 5x5 image, all 1s
  // Ensure the shape matches [Channels, Height, Width]
  final input = Tensor.fromList([1, 5, 5], List.generate(25, (i) => 1.0));

  // 2. Setup Conv: 3x3 kernel, 1 in-channel, 1 out-channel (to keep math simple)
  final convLayer = Conv2d(
    inChannels: 1,
    outChannels: 1,
    kernelSize: 3,
    stride: 1,
    padding: 1,
  );

  // 3. Manually set weights/bias to 1.0 for predictable verification
  // Weight shape: [OutChannels, InChannels * K * K] -> [1, 9]
  convLayer.weight = Tensor.fromList([1, 9], List.generate(9, (i) => 1.0));
  convLayer.bias = Tensor.fromList([1, 1], [0.0]);

  List<Tensor> tracker = [];

  try {
    // 4. Run Forward Pass
    // 'output' will have shape [1, 25] or [1, 5, 5] depending on your matmul/+ implementation
    final output = convLayer.forward(input, tracker);

    // 5. Fetch Data from GPU to CPU
    final data = output.fetchData();

    print("--- Results ---");
    print("Top-Left Pixel Result: ${data[0]}");

    // 6. Manually print the 5x5 grid from the flat data
    // For a 3x3 kernel of 1s on an image of 1s:
    // Corner pixel (with padding) should be 4.0
    // Edge pixel should be 6.0
    // Center pixel should be 9.0
    print("Sample Output Matrix (5x5):");
    for (int i = 0; i < 5; i++) {
      // sublist(start, end) takes the row from the flattened array
      print(data.sublist(i * 5, (i + 1) * 5));
    }

    if (data[0] > 0) {
      print("✅ SUCCESS: Conv2d is producing non-zero data!");
    } else {
      print("❌ STILL ZERO: Check the C++ wrapper for pointer casting.");
    }
  } catch (e) {
    print("Caught Error: $e");
  } finally {
    // 7. Cleanup
    input.dispose();
    convLayer.dispose();
    for (var t in tracker) t.dispose();
  }
}
