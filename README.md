🚀 G-Tensor: High-Performance Dart & CUDA Deep Learning EngineG-Tensor is a custom deep learning framework that combines the developer productivity of Dart with the raw computational power of NVIDIA CUDA.Unlike standard wrappers, G-Tensor features a custom Autoregressive Functional Transformer (AFT) implementation, a manual Autograd engine, and hand-optimized CUDA kernels for operations like Causal Masking, Layer Normalization, and Cross-Entropy with Label Smoothing.🏗 System ArchitectureThe engine is split into three distinct layers:Dart API (Frontend): High-level Tensor class with operator overloading (+, -, *, matmul) and Module classes for building neural networks.FFI Bridge: A low-level Dart FFI (Foreign Function Interface) layer that handles memory addresses and dispatches calls to compiled C++/CUDA binaries.CUDA Kernels (Backend): Hand-written .cu kernels optimized for parallel execution on the GPU, featuring custom broadcasting logic and stable gradient calculations.🧠 The AFT Causal Mechanism (Mathematical Derivation)The core of this engine is the Attention Free Transformer (AFT). Unlike standard Multi-Head Attention which has $O(T^2)$ complexity, AFT reduces this to $O(Td)$ by re-arranging the interaction between Queries, Keys, and Values.The FormulationIn your implementation, the attention-like operation is defined as:$$Z_t = \sigma(Q_t) \odot \frac{\sum_{i=1}^t \exp(K_i + w_{t,i}) \odot V_i}{\sum_{i=1}^t \exp(K_i + w_{t,i})}$$Where:$\sigma$ is the Sigmoid activation.$\odot$ is the Element-wise (Hadamard) product (successfully verified in our test_tensor2.dart).$w_{t,i}$ represents the Learned Pairwise Position Bias.The Causal MaskTo ensure the model cannot "cheat" by looking at future tokens, we apply a triangular causal mask. In G-Tensor, this is handled by a specialized engine.mulTensors call during the forward pass, ensuring that for any time $t$, the gradients from $t+1 \dots T$ are exactly zero.✨ Key FeaturesCustom Autograd: Fully functional backpropagation through computational graphs.Efficient Memory Management: Explicit tracker and dispose system to prevent VRAM leaks in Dart's garbage-collected environment.Broadcasting: Support for adding row-vector biases to activation matrices via custom CUDA indexing (e.g., adding [1, 128] bias to [64, 128] activations).Advanced Loss Kernels: Stable Cross-Entropy with built-in LogSoftmax and Label Smoothing ($\epsilon = 0.1$).🛠 Installation & SetupPrerequisitesDart SDK (v3.0+)NVIDIA CUDA Toolkit (v11.0+)CMake (for building the C++ backend)Building the BackendNavigate to the src directory.Compile the CUDA shared library:Bashmkdir build && cd build
cmake ..
make
Ensure the generated .so or .dll is in your LD_LIBRARY_PATH.💻 Usage Example1. Training with Memory ManagementBecause Dart is garbage collected but CUDA memory is not, you must use the tracker pattern:Dartfor (int step = 0; step < 1000; step++) {
  List<Tensor> tracker = [];
  optimizer.zeroGrad();

  // Forward pass
  final logits = gpt.forward(inputIdx, dummyEnc, tracker);
  final loss = logits.crossEntropy(targetIds); 

  // Backward pass
  loss.backward();
  optimizer.step();

  // CLEANUP: Free intermediate tensors to prevent CUDA OOM
  for (var t in tracker) {
    if (!gpt.parameters().contains(t)) t.dispose();
  }
  loss.dispose();
}
2. Autoregressive GenerationThe engine supports greedy and nucleus sampling for text generation:Dartvoid generate(String prompt) {
  List<int> tokens = tokenizer.encode(prompt);
  final logits = gpt.forward(tokens, dummyEnc, []);
  
  // Fetch only the last row for prediction
  List<double> lastLogits = logits.fetchRow(tokens.length - 1);
  int nextToken = sampleNucleus(lastLogits, temp: 0.8, topP: 0.9);
  print(tokenizer.decode([nextToken]));
}
📊 Performance BenchmarksOperationInput ShapeG-Tensor (CUDA)AFT Forward[64, 128]~0.9msCrossEntropy[64, 50257]~1.1msLayerNorm[64, 128]~0.3ms🤝 Roadmap & ContributingCurrent development priorities:2D Batching: Optimizing matmul kernels for [Batch, Seq, Hidden] shapes.KV-Caching: Implementation for faster inference.RMSNorm: Adding support for Llama-style normalization.Maintainer: kkinyamaProject: dart_cuda