import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

typedef _C_create =
    ffi.Pointer<ffi.Void> Function(
      ffi.Int32,
      ffi.Int32,
      ffi.Pointer<ffi.Float>,
    );
typedef _D_create =
    ffi.Pointer<ffi.Void> Function(int, int, ffi.Pointer<ffi.Float>);
typedef _C_destroy = ffi.Void Function(ffi.Pointer<ffi.Void>);
typedef _D_destroy = void Function(ffi.Pointer<ffi.Void>);
typedef _C_copy =
    ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);
typedef _D_copy = void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);
typedef _C_op1 = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);
typedef _D_op1 = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);
typedef _C_op2 =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
typedef _D_op2 =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
typedef _C_pow =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Float);
typedef _D_pow = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, double);
typedef _C_step = ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Float);
typedef _D_step = void Function(ffi.Pointer<ffi.Void>, double);
typedef _C_aft =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Bool,
    );
typedef _D_aft =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      bool,
    );

typedef _C_aft_cross =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
typedef _D_aft_cross =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );

typedef _C_concat =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Pointer<ffi.Void>>,
      ffi.Int32,
    );
typedef _D_concat =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Pointer<ffi.Void>>, int);

typedef _C_layernorm =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Float,
    );
typedef _D_layernorm =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      double,
    );

// 1. Define the types
typedef _C_embedding =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Int32>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Int32,
      ffi.Int32,
    );
typedef _D_embedding =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Int32>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      int,
      int,
    );

typedef _C_loss =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Int32>,
      ffi.Int32,
      ffi.Int32,
    );
typedef _D_loss =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Int32>,
      int,
      int,
    );

typedef _C_to_host =
    ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);
typedef _D_to_host =
    void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);

typedef _C_adam_step =
    ffi.Void Function(
      ffi.Pointer<ffi.Void> p, // Parameters
      ffi.Pointer<ffi.Void> m, // 1st Moment
      ffi.Pointer<ffi.Void> v, // 2nd Moment
      ffi.Int32 t, // Timestep
      ffi.Float lr, // Learning Rate
      ffi.Float b1, // Beta 1
      ffi.Float b2, // Beta 2
      ffi.Float eps, // Epsilon
    );

typedef _C_clip = ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Float);
typedef _D_clip = void Function(ffi.Pointer<ffi.Void>, double);

typedef _C_set_data =
    ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);
typedef _D_set_data =
    void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>);
// 2. Define the Dart-style signature
typedef _D_adam_step =
    void Function(
      ffi.Pointer<ffi.Void> p,
      ffi.Pointer<ffi.Void> m,
      ffi.Pointer<ffi.Void> v,
      int t,
      double lr,
      double b1,
      double b2,
      double eps,
    );

typedef _C_slice =
    ffi.Pointer<ffi.Void> Function(
      ffi.Pointer<ffi.Void>,
      ffi.Int32, // startRow
      ffi.Int32, // numRows
    );
typedef _D_slice =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, int, int);

typedef UnaryOpFn = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);
typedef UnaryOpDart = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);

// Define the C signatures
typedef NativeComputeCost =
    ffi.Void Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );
// Define the Dart signatures
typedef DartComputeCost =
    void Function(
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
      ffi.Pointer<ffi.Void>,
    );

// Unary operations for reductions (collapsing a tensor to a 1x1 scalar)
typedef _C_reduce = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);
typedef _D_reduce = ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>);
typedef _XavierInitC =
    ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Int32, ffi.Int32, ffi.Int32);
typedef _XavierInitDart = void Function(ffi.Pointer<ffi.Void>, int, int, int);

typedef _ZeroInitC = ffi.Void Function(ffi.Pointer<ffi.Void>);
typedef _ZeroInitDart = void Function(ffi.Pointer<ffi.Void>);

class CudaEngine {
  late ffi.DynamicLibrary _lib;
  late _D_create createTensor;
  late _D_destroy destroyTensor;
  late _D_copy getTensorData;
  late _D_copy getTensorGrad;
  late _D_destroy backward;
  late _D_destroy zeroGrad;
  late _D_step tensorStep;
  late _D_op2 addTensors;
  late _D_op2 subTensors;
  late _D_op2 mulTensors;
  late _D_op2 divTensors;
  late _D_op2 matmulTensors;
  late _D_pow powTensor;
  late _D_op1 reluTensor;
  late _D_op1 tanhTensor;
  late _D_op1 sigmoidTensor;
  late _D_op1 logTensor;
  late _D_aft aftForward;
  late _D_aft_cross aftCrossForward;
  late _D_concat concatTensors;
  late _D_layernorm layernormForward;
  late _D_op1 geluTensor;
  late _D_embedding embeddingForward;
  late _D_loss crossEntropyLoss; // Add this line
  late _D_to_host tensorToHost;
  late _D_adam_step adamStep;
  late _D_clip clipGradients;
  late _D_set_data setTensorData;
  late _D_slice sliceTensor;
  late UnaryOpDart abs_tensor;
  late UnaryOpDart softmax_forward;
  late DartComputeCost _computeCostMatrix; // <--- The internal definition
  late _D_reduce sumTensor;
  late _D_reduce meanTensor;
  late _ZeroInitDart _tensorZeroInit;
  late _XavierInitDart _tensorXavierInit;

  CudaEngine() {
    _lib = ffi.DynamicLibrary.open('${Directory.current.path}/libmatmul_v2.so');
    createTensor = _lib.lookupFunction<_C_create, _D_create>('create_tensor');
    destroyTensor = _lib.lookupFunction<_C_destroy, _D_destroy>(
      'destroy_tensor',
    );
    getTensorData = _lib.lookupFunction<_C_copy, _D_copy>('get_tensor_data');
    getTensorGrad = _lib.lookupFunction<_C_copy, _D_copy>('get_tensor_grad');
    backward = _lib.lookupFunction<_C_destroy, _D_destroy>('backward');
    zeroGrad = _lib.lookupFunction<_C_destroy, _D_destroy>('zero_grad');
    tensorStep = _lib.lookupFunction<_C_step, _D_step>('tensor_step');
    addTensors = _lib.lookupFunction<_C_op2, _D_op2>('add_tensors');
    subTensors = _lib.lookupFunction<_C_op2, _D_op2>('sub_tensors');
    mulTensors = _lib.lookupFunction<_C_op2, _D_op2>('mul_tensors');
    divTensors = _lib.lookupFunction<_C_op2, _D_op2>('div_tensors');
    matmulTensors = _lib.lookupFunction<_C_op2, _D_op2>('matmul_tensors');
    powTensor = _lib.lookupFunction<_C_pow, _D_pow>('pow_tensor');
    reluTensor = _lib.lookupFunction<_C_op1, _D_op1>('relu_tensor');
    tanhTensor = _lib.lookupFunction<_C_op1, _D_op1>('tanh_tensor');
    sigmoidTensor = _lib.lookupFunction<_C_op1, _D_op1>('sigmoid_tensor');
    logTensor = _lib.lookupFunction<_C_op1, _D_op1>('log_tensor');
    aftForward = _lib.lookupFunction<_C_aft, _D_aft>('aft_forward');
    aftCrossForward = _lib.lookupFunction<_C_aft_cross, _D_aft_cross>(
      'aft_cross_forward',
    );
    concatTensors = _lib.lookupFunction<_C_concat, _D_concat>(
      'concat_tensors_gpu',
    );

    layernormForward = _lib.lookupFunction<_C_layernorm, _D_layernorm>(
      'layernorm_forward',
    );

    geluTensor = _lib.lookupFunction<_C_op1, _D_op1>('gelu_tensor');
    embeddingForward = _lib.lookupFunction<_C_embedding, _D_embedding>(
      'embedding_forward',
    );
    crossEntropyLoss = _lib.lookupFunction<_C_loss, _D_loss>(
      'cross_entropy_loss',
    );

    tensorToHost = _lib.lookupFunction<_C_to_host, _D_to_host>(
      'tensor_to_host',
    );

    adamStep = _lib.lookupFunction<_C_adam_step, _D_adam_step>('adam_step');

    // Ensure you also have zeroGrad defined
    zeroGrad = _lib
        .lookupFunction<
          ffi.Void Function(ffi.Pointer<ffi.Void>),
          void Function(ffi.Pointer<ffi.Void>)
        >('zero_grad');

    clipGradients = _lib.lookupFunction<_C_clip, _D_clip>('clip_gradients');

    setTensorData = _lib.lookupFunction<_C_set_data, _D_set_data>(
      'set_tensor_data',
    );

    sliceTensor = _lib.lookupFunction<_C_slice, _D_slice>('slice_tensor');

    abs_tensor = _lib.lookupFunction<UnaryOpFn, UnaryOpDart>('abs_tensor');

    softmax_forward = _lib.lookupFunction<UnaryOpFn, UnaryOpDart>(
      'softmax_forward',
    );

    _computeCostMatrix = _lib
        .lookup<ffi.NativeFunction<NativeComputeCost>>('compute_cost_matrix')
        .asFunction<DartComputeCost>();

    sumTensor = _lib.lookupFunction<_C_reduce, _D_reduce>('sum_tensor');

    // Mean reduction: returns a 1x1 Tensor pointer
    meanTensor = _lib.lookupFunction<_C_reduce, _D_reduce>('mean_tensor');

    _tensorXavierInit = _lib.lookupFunction<_XavierInitC, _XavierInitDart>(
      'tensor_xavier_init',
    );

    _tensorZeroInit = _lib.lookupFunction<_ZeroInitC, _ZeroInitDart>(
      'tensor_zero_init',
    );
    // Ensure your existing slice and unary ops are there
    // sliceTensor = _lib.lookupFunction<_C_slice, _D_slice>('slice_tensor');
    // abs_tensor = _lib.lookupFunction<UnaryOpFn, UnaryOpDart>('abs_tensor');
    // softmax_forward = _lib.lookupFunction<UnaryOpFn, UnaryOpDart>('softmax_forward');
  }

  void computeCostMatrix(
    ffi.Pointer<ffi.Void> pb,
    ffi.Pointer<ffi.Void> gb,
    ffi.Pointer<ffi.Void> cm,
  ) {
    _computeCostMatrix(pb, gb, cm);
  }
}

final engine = CudaEngine();

class Tensor {
  final ffi.Pointer<ffi.Void> _handle;
  final List<int> shape;
  final int length;

  bool? isView; // New field

  Tensor._raw(this._handle, this.shape, {this.isView = false})
    : length = shape.reduce((a, b) => a * b);

  // Manual Dispose
  // void dispose() => engine.destroyTensor(_handle);

  ffi.Pointer<ffi.Void> get handle => _handle;

  Tensor gelu() => Tensor._raw(engine.geluTensor(_handle), shape);

  factory Tensor.fill(List<int> shape, double val) {
    final ptr = calloc<ffi.Float>(shape.reduce((a, b) => a * b));
    for (int i = 0; i < shape.reduce((a, b) => a * b); i++) ptr[i] = val;
    final h = engine.createTensor(
      shape[0],
      shape.length > 1 ? shape[1] : 1,
      ptr,
    );
    calloc.free(ptr);
    return Tensor._raw(h, shape);
  }

  factory Tensor.fromList(List<int> shape, List<double> vals) {
    final ptr = calloc<ffi.Float>(vals.length);
    for (int i = 0; i < vals.length; i++) ptr[i] = vals[i];
    final h = engine.createTensor(
      shape[0],
      shape.length > 1 ? shape[1] : 1,
      ptr,
    );
    calloc.free(ptr);
    return Tensor._raw(h, shape);
  }

  List<double> get data {
    final ptr = calloc<ffi.Float>(length);
    engine.getTensorData(_handle, ptr);
    final l = ptr.asTypedList(length).toList();
    calloc.free(ptr);
    return l;
  }

  void zeroGrad() => engine.zeroGrad(_handle);
  void step(double lr) => engine.tensorStep(_handle, lr);
  void backward() => engine.backward(_handle);

  Tensor sum() {
    final h = engine.sumTensor(this.handle);
    return Tensor._raw(h, [1, 1]);
  }

  /// Collapses the entire tensor into a 1x1 Tensor containing the mean
  Tensor mean() {
    final h = engine.meanTensor(this.handle);
    return Tensor._raw(h, [1, 1]);
  }

  // void xavier(int nIn, int nOut) {
  //   // Use current time as seed so every training run is unique
  //   int seed = DateTime.now().millisecondsSinceEpoch;

  //   // FFI call to the C++ wrapper
  // engine.tensorXavierInit(
  //     this.handle, // Assuming this is your Pointer to the C++ Tensor object
  //     nIn,
  //     nOut,
  //     seed,
  //   );
  // }

  // Tensor operator +(Tensor o) =>
  //     Tensor._raw(engine.addTensors(_handle, o._handle), shape);
  // Tensor operator -(Tensor o) =>
  //     Tensor._raw(engine.subTensors(_handle, o._handle), shape);
  // Tensor operator *(Tensor o) =>
  //     Tensor._raw(engine.mulTensors(_handle, o._handle), shape);
  Tensor matmul(Tensor o) => Tensor._raw(
    engine.matmulTensors(_handle, o._handle),
    [shape[0], o.shape[1]],
  );
  Tensor sigmoid() => Tensor._raw(engine.sigmoidTensor(_handle), shape);
  Tensor pow(double e) => Tensor._raw(engine.powTensor(_handle, e), shape);
  Tensor log() => Tensor._raw(engine.logTensor(_handle), shape);

  Tensor _scalarOp(
    dynamic other,
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Void>)
    opFunc,
  ) {
    if (other is Tensor) {
      return Tensor._raw(opFunc(_handle, other._handle), shape);
    } else if (other is double || other is int) {
      // Create a temporary 1x1 tensor for the scalar value
      final tempScalar = Tensor.fill([1], (other as num).toDouble());
      final resultHandle = opFunc(_handle, tempScalar.handle);
      tempScalar.dispose(); // Immediately free the temporary scalar handle
      return Tensor._raw(resultHandle, shape);
    }
    throw ArgumentError(
      "Operation not supported for type ${other.runtimeType}",
    );
  }

  // Fixed Operators
  Tensor operator +(dynamic o) => _scalarOp(o, engine.addTensors);
  Tensor operator -(dynamic o) => _scalarOp(o, engine.subTensors);
  Tensor operator *(dynamic o) => _scalarOp(o, engine.mulTensors);

  // Cross Entropy Implementation
  Tensor computeCrossEntropy(Tensor pred, Tensor target, List<Tensor> tracker) {
    // 1. logPred = log(pred)
    final logPred = pred.log();
    // 2. product = target * log(pred)
    final product = target * logPred;

    // 3. Apply negative sign: loss = product * -1
    // We'll create a constant tensor for -1.0
    final negOne = Tensor.fill(product.shape, -1.0);
    final loss = product * negOne;

    // Track all temporaries for manual disposal later
    tracker.addAll([logPred, product, negOne]);

    return loss;
  }

  Tensor abs() {
    final h = engine.abs_tensor(_handle);
    return Tensor._raw(h, shape);
  }

  Tensor softmax() {
    // Assuming your engine has a softmaxForward that handles [T, V] or [1, V]
    final h = engine.softmax_forward(_handle);
    return Tensor._raw(h, shape);
  }

  // Inside Tensor class
  static Tensor aft(Tensor q, Tensor k, Tensor v, Tensor wb, bool masked) {
    return Tensor._raw(
      engine.aftForward(q._handle, k._handle, v._handle, wb._handle, masked),
      q.shape,
    );
  }

  // Inside Tensor class
  static Tensor aftCross(Tensor q, Tensor k, Tensor v, Tensor wb) {
    return Tensor._raw(
      engine.aftCrossForward(q._handle, k._handle, v._handle, wb._handle),
      q.shape,
    );
  }

  // Inside Tensor class
  static Tensor concat(List<Tensor> tensors) {
    final ptr = calloc<ffi.Pointer<ffi.Void>>(tensors.length);
    for (int i = 0; i < tensors.length; i++) {
      ptr[i] = tensors[i]._handle;
    }
    final h = engine.concatTensors(ptr, tensors.length);
    calloc.free(ptr);
    return Tensor._raw(h, [
      tensors[0].shape[0],
      tensors[0].shape[1] * tensors.length,
    ]);
  }

  // In Tensor class
  static Tensor layerNorm(Tensor x, Tensor gamma, Tensor beta, double eps) {
    return Tensor._raw(
      engine.layernormForward(x._handle, gamma._handle, beta._handle, eps),
      x.shape,
    );
  }

  // 3. In Tensor class
  static Tensor embeddings(List<int> idx, Tensor wte, Tensor wpe) {
    final T = idx.length;
    final D = wte.shape[1];

    // Allocate memory for the indices that C can read
    final ptr = calloc<ffi.Int32>(T);
    for (int i = 0; i < T; i++) ptr[i] = idx[i];

    final handle = engine.embeddingForward(ptr, wte._handle, wpe._handle, T, D);

    // We can free the host pointer immediately because C++ copied it to d_indices
    calloc.free(ptr);

    return Tensor._raw(handle, [T, D]);
  }

  static Tensor zeros(List<int> shape) {
    final int rows = shape[0];
    final int cols = shape.length > 1 ? shape[1] : 1;
    final int size = rows * cols;

    // Allocate native memory filled with 0.0
    final ffi.Pointer<ffi.Float> buffer = calloc<ffi.Float>(size);
    // calloc usually zeros memory, but let's be explicit
    for (int i = 0; i < size; i++) {
      buffer[i] = 0.0;
    }

    final handle = engine.createTensor(rows, cols, buffer);
    calloc.free(buffer);

    return Tensor._raw(handle, shape);
  }

  static Tensor random(List<int> shape, {double scale = 0.005}) {
    final int rows = shape[0];
    final int cols = shape.length > 1 ? shape[1] : 1;
    final int size = rows * cols;

    final math.Random rng = math.Random();

    // 1. Allocate native memory (Pointer<Float>)
    final ffi.Pointer<ffi.Float> nativeBuffer = calloc<ffi.Float>(size);

    // 2. Fill the native buffer with random values
    for (int i = 0; i < size; i++) {
      // Approximating a small normal distribution
      nativeBuffer[i] = (rng.nextDouble() * 2 - 1) * scale;
    }

    // 3. Pass the pointer to C++
    // createTensor should be: Pointer<Void> Function(Int32, Int32, Pointer<Float>)
    final handle = engine.createTensor(rows, cols, nativeBuffer);

    // 4. IMPORTANT: Free the native memory after the GPU has copied it
    calloc.free(nativeBuffer);

    return Tensor._raw(handle, shape);
  }

  // Inside Tensor class
  // Inside the Tensor class
  Tensor crossEntropy(List<int> targets) {
    // logits are [T, V]
    final int T = shape[0];
    final int V = shape[1];

    if (targets.length != T) {
      throw ArgumentError(
        "Target length ${targets.length} must match Logits T: $T",
      );
    }

    // Allocate native memory for targets
    final ptr = calloc<ffi.Int32>(T);
    for (int i = 0; i < T; i++) {
      ptr[i] = targets[i];
    }

    // Call the engine
    final handle = engine.crossEntropyLoss(_handle, ptr, T, V);

    // Clean up the host pointer (C++ has already copied it to GPU)
    calloc.free(ptr);

    return Tensor._raw(handle, [1, 1]); // Returns a scalar loss
  }

  // Pulls ALL data from GPU to CPU
  Float32List fetchData() {
    final size = shape.reduce((a, b) => a * b);
    final buffer = Float32List(size);
    final ptr = calloc<ffi.Float>(size);

    // Your C++ engine must have a tensor_to_host function
    engine.tensorToHost(_handle, ptr);

    for (int i = 0; i < size; i++) buffer[i] = ptr[i];
    calloc.free(ptr);
    return buffer;
  }

  // Pulls only one row (for sampling)
  List<double> fetchRow(int row) {
    final cols = shape[1];
    final allData = fetchData();
    return allData.sublist(row * cols, (row + 1) * cols);
  }

  set data(List<double> newData) {
    if (newData.length != length) {
      throw ArgumentError(
        "Data length ${newData.length} mismatch with Tensor length $length",
      );
    }

    // Allocate native memory to hold the new data
    final ptr = calloc<ffi.Float>(length);

    // Fill the native buffer
    for (int i = 0; i < length; i++) {
      ptr[i] = newData[i];
    }

    // 1. You need this FFI call in your engine:
    // engine.setTensorData(handle, ptr);
    // It should perform: cudaMemcpy(t->data_gpu, ptr, size, cudaMemcpyHostToDevice);
    engine.setTensorData(_handle, ptr);

    // Clean up host memory
    calloc.free(ptr);
  }

  Tensor getRow(int row) {
    // 1. Safety check to prevent C++ segmentation faults
    if (row < 0 || row >= shape[0]) throw RangeError("Row index out of bounds");

    // 2. Call the engine. This returns a NEW C++ Tensor* with its own cudaMalloc'd memory.
    final handle = engine.sliceTensor(_handle, row, 1);

    // 3. We pass isView: false (default) because this handle is UNIQUE and
    // must be destroyed to free the memory allocated by slice_tensor.
    return Tensor._raw(handle, [1, shape[1]], isView: false);
  }

  Tensor reshape(List<int> newShape) {
    // Use the SAME handle, but mark it as a view so it's never double-freed
    return Tensor._raw(this._handle, newShape, isView: true);
  }

  Tensor computeCostMatrix(Tensor gtBoxes) {
    // 1. Validation
    if (shape[1] != 4 || gtBoxes.shape[1] != 4) {
      throw ArgumentError("Both tensors must have 4 columns (x, y, w, h)");
    }

    // 2. Prepare the destination tensor on the GPU
    final int numQueries = shape[0];
    final int numGT = gtBoxes.shape[0];
    final costMatrix = Tensor.fill([numQueries, numGT], 0.0);

    // 3. Call the engine using the internal handles
    // 'gpu' here refers to your CudaEngine singleton/instance
    engine.computeCostMatrix(this.handle, gtBoxes.handle, costMatrix.handle);

    return costMatrix;
  }

  Tensor slice(int startRow, int rowCount) {
    // 1. Safety check
    if (startRow < 0 || (startRow + rowCount) > shape[0]) {
      throw RangeError("Slice indices out of bounds for shape $shape");
    }

    // 2. Call the engine (same one used by getRow)
    // startRow: where to begin, rowCount: how many rows to take
    final handle = engine.sliceTensor(_handle, startRow, rowCount);

    // 3. Return as a new Tensor (usually a unique handle in C++)
    return Tensor._raw(handle, [rowCount, shape[1]], isView: false);
  }

  bool _isDisposed = false;

  void dispose() {
    // Treat null as false (not a view, so it should be destroyed)
    if (_isDisposed || (isView ?? false) || _handle.address == 0) {
      return;
    }

    engine.destroyTensor(_handle);
    _isDisposed = true;
  }

  // final bool isView; // New field

  // Tensor._raw(this._handle, this.shape, {this.isView = false});

  // Tensor reshape(List<int> newShape) {
  //   // Mark this one as a view so it doesn't double-free
  //   return Tensor._raw(this._handle, newShape, isView: true);
  // }
}

void main() {
  final x = Tensor.fill([1, 1], 0.5);
  final y = x.sigmoid() + x.pow(2);
  y.backward();
  print('Forward: ${y.data}');

  final tracker = <Tensor>[];
  final pred = Tensor.fromList([1, 2], [0.1, 0.9]); // 90% confident in class 2
  final target = Tensor.fromList([1, 2], [0.0, 1.0]); // Class 2 is the truth

  final loss = pred.computeCrossEntropy(pred, target, tracker);

  print('Loss components: ${loss.data}');
  // Should show a small positive value for the correct class

  // Cleanup
  loss.dispose();
  for (var t in tracker) t.dispose();
  pred.dispose();
  target.dispose();
}
