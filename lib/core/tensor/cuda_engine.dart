// CUDA engine FFI bindings.
//
// Holds the `CudaEngine` class that wraps `libmat_mul.so` (compiled from
// `native/src/engine.cu`) and a process-wide `engine` singleton. Imported
// (and re-exported) by `gpu_tensor.dart`; users normally just
// `import 'package:dart_cuda/core/tensor/gpu_tensor.dart';` and access
// `engine.xxx` directly.

import 'dart:ffi' as ffi;
import 'dart:io';

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

// --- Add these Typedefs at the top level ---
typedef _C_l2norm =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, ffi.Float);
typedef _D_l2norm =
    ffi.Pointer<ffi.Void> Function(ffi.Pointer<ffi.Void>, double);

// C Signature (Matching the C++ wrapper)
typedef _C_im2col =
    ffi.Void Function(
      ffi.Pointer<ffi.Void> input,
      ffi.Int32 channels,
      ffi.Int32 height,
      ffi.Int32 width,
      ffi.Int32 kh,
      ffi.Int32 kw,
      ffi.Int32 ph,
      ffi.Int32 pw,
      ffi.Int32 sh,
      ffi.Int32 sw,
      ffi.Pointer<ffi.Void> output,
    );

// Dart Signature
typedef _D_im2col =
    void Function(
      ffi.Pointer<ffi.Void> input,
      int channels,
      int height,
      int width,
      int kH,
      int kW,
      int pH,
      int pW,
      int sH,
      int sW,
      ffi.Pointer<ffi.Void> output,
    );

typedef MatMulFunc =
    ffi.Void Function(
      ffi.Pointer<ffi.Float> a,
      ffi.Pointer<ffi.Float> b,
      ffi.Pointer<ffi.Float> c,
      ffi.Int32 M,
      ffi.Int32 N,
      ffi.Int32 K,
    );

// This is the Dart-friendly version of the function.
typedef MatMul =
    void Function(
      ffi.Pointer<ffi.Float> a,
      ffi.Pointer<ffi.Float> b,
      ffi.Pointer<ffi.Float> c,
      int M,
      int N,
      int K,
    );

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
  late _D_op2 addTensorScalar;
  late _D_op2 subTensorScalar;
  late _D_op2 mulTensorScalar;
  late _D_op2 divTensorScalar;
  late _D_op2 matmulTensors;
  late _D_pow powTensor;
  late _D_op1 reluTensor;
  late _D_op1 tanhTensor;
  late _D_op1 sigmoidTensor;
  late _D_op1 logTensor;
  late _D_aft aftForward;
  late _D_aft_cross aftCrossForward;
  late _D_concat concatTensors;
  late _D_concat concatTensorsAxis0;
  late _D_layernorm layernormForward;
  late _D_op1 geluTensor;
  late _D_embedding embeddingForward;
  late _D_loss crossEntropyLoss; // Add this line
  late _D_to_host tensorToHost;
  late _D_adam_step adamStep;
  late _D_adam_step sdgStep;
  late _D_clip clipGradients;
  late _D_set_data setTensorData;
  late _D_slice sliceTensor;
  late UnaryOpDart abs_tensor;
  late UnaryOpDart softmax_forward;
  late DartComputeCost _computeCostMatrix; // <--- The internal definition
  late _D_reduce sumTensor;
  late _D_reduce meanTensor;
  // late _ZeroInitDart _tensorZeroInit;
  // late _XavierInitDart _tensorXavierInit;
  // late _D_l2norm l2Normalize; // Add this
  late _D_l2norm layerNorm;
  late _D_im2col im2col;
  late _D_im2col col2im;

  late final matMulCuda;

  CudaEngine() {
    _lib = ffi.DynamicLibrary.open(
      '${Directory.current.path}/native/lib/libmat_mul.so',
    );
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
    addTensorScalar = _lib.lookupFunction<_C_op2, _D_op2>('add_tensor_scalar');
    subTensorScalar = _lib.lookupFunction<_C_op2, _D_op2>('sub_tensor_scalar');
    mulTensorScalar = _lib.lookupFunction<_C_op2, _D_op2>('mul_tensor_scalar');
    divTensorScalar = _lib.lookupFunction<_C_op2, _D_op2>('div_tensor_scalar');
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
    concatTensorsAxis0 = _lib.lookupFunction<_C_concat, _D_concat>(
      'concat_tensors_axis0_gpu',
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
    sdgStep = _lib.lookupFunction<_C_adam_step, _D_adam_step>('sdg_step');
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

    // _tensorXavierInit = _lib.lookupFunction<_XavierInitC, _XavierInitDart>(
    //   'tensor_xavier_init',
    // );

    // _tensorZeroInit = _lib.lookupFunction<_ZeroInitC, _ZeroInitDart>(
    //   'tensor_zero_init',
    // );

    // l2Normalize = _lib.lookupFunction<_C_l2norm, _D_l2norm>(
    //   'l2_normalize_tensor',
    // );

    layerNorm = _lib.lookupFunction<_C_l2norm, _D_l2norm>('layer_norm_tensor');
    im2col = _lib.lookupFunction<_C_im2col, _D_im2col>('im2col_cuda');
    col2im = _lib.lookupFunction<_C_im2col, _D_im2col>('col2im_cuda');

    // matMulCuda = _lib.lookupFunction<MatMulFunc, MatMul>(
    //   'matrix_multiply_cuda',
    // );
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
