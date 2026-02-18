import 'dart:ffi' as ffi;
import 'dart:io';
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

  CudaEngine() {
    _lib = ffi.DynamicLibrary.open(Directory.current.path + '/libmatmul.so');
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
  }
}

final engine = CudaEngine();

class Tensor {
  final ffi.Pointer<ffi.Void> _handle;
  final List<int> shape;
  final int length;

  Tensor._raw(this._handle, this.shape)
    : length = shape.reduce((a, b) => a * b);

  // Manual Dispose
  void dispose() => engine.destroyTensor(_handle);

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

  Tensor operator +(Tensor o) =>
      Tensor._raw(engine.addTensors(_handle, o._handle), shape);
  Tensor operator -(Tensor o) =>
      Tensor._raw(engine.subTensors(_handle, o._handle), shape);
  Tensor operator *(Tensor o) =>
      Tensor._raw(engine.mulTensors(_handle, o._handle), shape);
  Tensor matmul(Tensor o) => Tensor._raw(
    engine.matmulTensors(_handle, o._handle),
    [shape[0], o.shape[1]],
  );
  Tensor sigmoid() => Tensor._raw(engine.sigmoidTensor(_handle), shape);
  Tensor pow(double e) => Tensor._raw(engine.powTensor(_handle, e), shape);
  Tensor log() => Tensor._raw(engine.logTensor(_handle), shape);

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
