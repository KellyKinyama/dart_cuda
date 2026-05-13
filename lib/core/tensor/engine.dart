import 'dart:ffi';
import 'package:ffi/ffi.dart';

// 1. Define the FFI function signatures to match the C/C++ wrapper.

// This is the native C function signature.
typedef MatMulFunc =
    Void Function(
      Pointer<Float> a,
      Pointer<Float> b,
      Pointer<Float> c,
      Int32 M,
      Int32 N,
      Int32 K,
    );

// This is the Dart-friendly version of the function.
typedef MatMul =
    void Function(
      Pointer<Float> a,
      Pointer<Float> b,
      Pointer<Float> c,
      int M,
      int N,
      int K,
    );

class CudaEngine {
  late final DynamicLibrary _lib;
  late final matMulCuda;

  CudaEngine() {
    final DynamicLibrary _lib = DynamicLibrary.open('./dart_cuda.so');
    matMulCuda = _lib.lookupFunction<MatMulFunc, MatMul>(
      'matrix_multiply_cuda',
    );
  }
}

final engine = CudaEngine();
