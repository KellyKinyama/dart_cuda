import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'dart:typed_data';

// --- FFI Function Definitions ---
typedef CreateTensorNative = Pointer<Void> Function(Int32 rows, Int32 cols, Pointer<Float> data);
typedef CreateTensor = Pointer<Void> Function(int rows, int cols, Pointer<Float> data);
typedef DestroyTensorNative = Void Function(Pointer<Void> handle);
typedef DestroyTensor = void Function(Pointer<Void> handle);
typedef GetTensorDataNative = Void Function(Pointer<Void> handle, Pointer<Float> cpuBuffer);
typedef GetTensorData = void Function(Pointer<Void> handle, Pointer<Float> cpuBuffer);
typedef GetTensorGradNative = Void Function(Pointer<Void> handle, Pointer<Float> cpuBuffer);
typedef GetTensorGrad = void Function(Pointer<Void> handle, Pointer<Float> cpuBuffer);
typedef BinaryOpNative = Pointer<Void> Function(Pointer<Void> a, Pointer<Void> b);
typedef BinaryOp = Pointer<Void> Function(Pointer<Void> a, Pointer<Void> b);
typedef UnaryOpNative = Pointer<Void> Function(Pointer<Void> a);
typedef UnaryOp = Pointer<Void> Function(Pointer<Void> a);
typedef PowTensorNative = Pointer<Void> Function(Pointer<Void> a, Float exp);
typedef PowTensor = Pointer<Void> Function(Pointer<Void> a, double exp);
typedef BackwardNative = Void Function(Pointer<Void> handle);
typedef Backward = void Function(Pointer<Void> handle);


// --- Dart Tensor Class ---
class Tensor {
  late Pointer<Void> _handle;
  final int rows;
  final int cols;

  // --- Static FFI Lookups ---
  static final _create = _dylib.lookupFunction<CreateTensorNative, CreateTensor>('create_tensor');
  static final _destroy = _dylib.lookupFunction<DestroyTensorNative, DestroyTensor>('destroy_tensor');
  static final _getData = _dylib.lookupFunction<GetTensorDataNative, GetTensorData>('get_tensor_data');
  static final _getGrad = _dylib.lookupFunction<GetTensorGradNative, GetTensorGrad>('get_tensor_grad');
  static final _add = _dylib.lookupFunction<BinaryOpNative, BinaryOp>('add_tensors');
  static final _sub = _dylib.lookupFunction<BinaryOpNative, BinaryOp>('sub_tensors');
  static final _mul = _dylib.lookupFunction<BinaryOpNative, BinaryOp>('mul_tensors');
  static final _div = _dylib.lookupFunction<BinaryOpNative, BinaryOp>('div_tensors');
  static final _neg = _dylib.lookupFunction<UnaryOpNative, UnaryOp>('neg_tensor');
  static final _pow = _dylib.lookupFunction<PowTensorNative, PowTensor>('pow_tensor');
  static final _relu = _dylib.lookupFunction<UnaryOpNative, UnaryOp>('relu_tensor');
  static final _tanh = _dylib.lookupFunction<UnaryOpNative, UnaryOp>('tanh_tensor');
  static final _exp = _dylib.lookupFunction<UnaryOpNative, UnaryOp>('exp_tensor');
  static final _log = _dylib.lookupFunction<UnaryOpNative, UnaryOp>('log_tensor');
  static final _backward = _dylib.lookupFunction<BackwardNative, Backward>('backward');

  // --- Constructors and Methods ---
  Tensor._fromHandle(this._handle, this.rows, this.cols);

  Tensor(Float32List data, {required this.rows, required this.cols}) {
    final pData = calloc<Float>(data.length);
    pData.asTypedList(data.length).setAll(0, data);
    _handle = _create(rows, cols, pData);
    calloc.free(pData);
  }

  void dispose() => _destroy(_handle);

  Float32List getData() {
    final pData = calloc<Float>(rows * cols);
    _getData(_handle, pData);
    final data = Float32List.fromList(pData.asTypedList(rows * cols));
    calloc.free(pData);
    return data;
  }

  Float32List getGrad() {
    final pGrad = calloc<Float>(rows * cols);
    _getGrad(_handle, pGrad);
    final grad = Float32List.fromList(pGrad.asTypedList(rows * cols));
    calloc.free(pGrad);
    return grad;
  }

  Tensor operator +(Tensor other) => Tensor._fromHandle(_add(_handle, other._handle), rows, cols);
  Tensor operator -(Tensor other) => Tensor._fromHandle(_sub(_handle, other._handle), rows, cols);
  Tensor operator *(Tensor other) => Tensor._fromHandle(_mul(_handle, other._handle), rows, cols);
  Tensor operator /(Tensor other) => Tensor._fromHandle(_div(_handle, other._handle), rows, cols);
  Tensor operator -() => Tensor._fromHandle(_neg(_handle), rows, cols);
  
  Tensor pow(double exponent) => Tensor._fromHandle(_pow(_handle, exponent), rows, cols);
  Tensor relu() => Tensor._fromHandle(_relu(_handle), rows, cols);
  Tensor tanh() => Tensor._fromHandle(_tanh(_handle), rows, cols);
  Tensor exp() => Tensor._fromHandle(_exp(_handle), rows, cols);
  Tensor log() => Tensor._fromHandle(_log(_handle), rows, cols);

  void backward() => _backward(_handle);

  @override
  String toString() => 'Tensor(data: ${getData()})';
}

final _dylib = DynamicLibrary.open('./libengine.so');

void main() {
  print('--- Comprehensive Test of a Neuron ---');
  
  // Inputs
  final x1 = Tensor(Float32List.fromList([2.0]), rows:1, cols:1);
  final x2 = Tensor(Float32List.fromList([3.0]), rows:1, cols:1);
  // Weights
  final w1 = Tensor(Float32List.fromList([-0.5]), rows:1, cols:1);
  final w2 = Tensor(Float32List.fromList([0.8]), rows:1, cols:1);
  // Bias
  final b = Tensor(Float32List.fromList([1.5]), rows:1, cols:1);

  // --- Forward Pass: o = tanh( (x1*w1 + x2*w2) + b ) ---
  final x1w1 = x1 * w1;
  final x2w2 = x2 * w2;
  final sum = x1w1 + x2w2;
  final n = sum + b;
  final o = n.tanh();
  
  print('Inputs: x1=${x1.getData()}, x2=${x2.getData()}, w1=${w1.getData()}, w2=${w2.getData()}, b=${b.getData()}');
  print('Output o: $o');

  // --- Backward Pass ---
  o.backward();
  print('\n--- Backward Pass Complete ---');
  
  print('Gradient of w1: ${w1.getGrad()}');
  print('Gradient of w2: ${w2.getGrad()}');
  print('Gradient of x1: ${x1.getGrad()}');
  print('Gradient of x2: ${x2.getGrad()}');
  print('Gradient of b: ${b.getGrad()}');

  // --- Cleanup ---
  x1.dispose(); x2.dispose(); w1.dispose(); w2.dispose(); b.dispose();
  x1w1.dispose(); x2w2.dispose(); sum.dispose(); n.dispose(); o.dispose();
  print('\n--- Test Finished ---');
}
