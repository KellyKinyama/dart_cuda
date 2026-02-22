library dart_cuda;

import 'dart:typed_data';
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import 'engine.dart';

part 'mat_mul.dart';
// part 'engine.dart';

class Tensor {
  List<int> shape;

  late Float32List data; // = hostA.asTypedList(M * K);
  late Float32List grad; // = hostA.asTypedList(M * K);

  void Function()? _backward;

  // 2. Fix the setter to ensure it's hitting the private field
  set onBackward(void Function() func) {
    _backward = func;
  }

  void _runBackward() {
    if (_backward != null) _backward!();
  }

  Tensor(
    this.shape,
    List<double> data,
    //   {
    //   required this.rows,
    //   required this.cols,
    // }
  ) {
    this.data = Float32List.fromList(data);
    grad = Float32List(this.data.length);
  }
}
