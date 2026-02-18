import 'dart:math' as math;
import 'dart:io';
import 'dart:typed_data';
import 'dart:ffi' as ffi;

import 'package:dart_cuda/adam.dart';
import 'package:dart_cuda/aft_transformer_decoder.dart';
import 'package:dart_cuda/gpu_tensor.dart';

// --- BINARY PERSISTENCE HELPERS ---

/// Pulls weights from GPU and writes them to a raw binary file
Future<void> saveModuleBinary(TransformerDecoder model, String filePath) async {
  final List<Tensor> parameters = model.parameters();
  final BytesBuilder builder = BytesBuilder();

  print('📦 Pulling trained weights from GPU VRAM...');
  for (var p in parameters) {
    // p.data (getter) triggers engine.getTensorData -> cudaMemcpyDeviceToHost
    final floatList = Float32List.fromList(p.data);
    builder.add(floatList.buffer.asUint8List());
  }

  await File(filePath).writeAsBytes(builder.toBytes());
  print('✅ Binary weights saved to: $filePath');
}

/// Reads a binary file and pushes the weights into GPU VRAM
Future<bool> loadModuleBinary(TransformerDecoder model, String filePath) async {
  final file = File(filePath);
  if (!await file.exists()) return false;

  final Uint8List allBytes = await file.readAsBytes();
  final Float32List allFloats = allBytes.buffer.asFloat32List();
  final List<Tensor> params = model.parameters();

  // Safety Check
  final int totalExpected = params.fold(0, (sum, p) => sum + p.length);
  if (allFloats.length != totalExpected) {
    print(
      '⚠️ Mismatch! Model needs $totalExpected floats, file has ${allFloats.length}',
    );
    return false;
  }

  print('🚀 Injecting weights into GPU VRAM...');
  int offset = 0;
  for (var p in params) {
    final int len = p.length;
    // p.data (setter) triggers engine.setTensorData -> cudaMemcpyHostToDevice
    p.data = allFloats.sublist(offset, offset + len).toList();
    offset += len;
  }
  return true;
}

// --- MAIN EXECUTION ---

void main() async {
  print("🚀 Starting GPU-Accelerated AFT-GPT Test Drive...");

  const int vocabSize = 25;
  const int bigSize = 128;
  const int blockSize = 16;
  const String weightPath = 'aft_model.bin';

  final stoi = {
    "hello": 0,
    "world": 1,
    "the": 2,
    "quick": 3,
    "brown": 4,
    "fox": 5,
    ".": 6,
    "<start>": 7,
    "jumps": 8,
    "over": 9,
    "lazy": 10,
    "dog": 11,
  };
  final itos = stoi.map((k, v) => MapEntry(v, k));

  final gpt = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: bigSize,
    encoderEmbedSize: bigSize,
    numLayers: 4,
    numHeads: 4,
    blockSize: blockSize,
  );

  final optimizer = Adam(gpt.parameters(), lr: 0.001);
  final dummyEnc = Tensor.zeros([1, bigSize]);

  // 1. Try to restore previous state
  bool isLoaded = await loadModuleBinary(gpt, weightPath);

  if (!isLoaded) {
    print("No existing weights found. Starting fresh training...");
    final dataset = [
      [7, 0, 1, 6], // <start> hello world .
      [7, 2, 3, 4, 5, 8, 9, 2, 10, 11, 6], // <start> the quick brown fox .
    ];

    for (int epoch = 0; epoch <= 500; epoch++) {
      double epochLoss = 0;

      for (var seq in dataset) {
        List<Tensor> tracker = [];
        optimizer.zeroGrad();

        final x = seq.sublist(0, seq.length - 1);
        final y = seq.sublist(1);

        final logits = gpt.forward(x, dummyEnc, tracker);
        final loss = logits.crossEntropy(y);

        loss.backward();
        optimizer.step();

        epochLoss += loss.fetchData()[0];

        for (var t in tracker) t.dispose();
        loss.dispose();
      }

      if (epoch % 50 == 0) {
        print(
          "Epoch $epoch | GPU Loss: ${(epochLoss / dataset.length).toStringAsFixed(6)}",
        );
      }
    }

    // 2. Save the weights so we don't have to train again
    await saveModuleBinary(gpt, weightPath);
  } else {
    print("✨ Weights successfully restored. Skipping training loop.");
  }

  print("\n--- Model Ready. Sampling from GPU ---");
  generate(
    gpt,
    [stoi["<start>"]!],
    stoi["."]!,
    itos,
    vocabSize,
    blockSize,
    dummyEnc,
  );
}

// --- SAMPLING LOGIC ---

void generate(
  TransformerDecoder model,
  List<int> gen,
  int endId,
  Map<int, String> itos,
  int vocabSize,
  int blockSize,
  Tensor dummyEnc,
) {
  for (int i = 0; i < 15; i++) {
    List<Tensor> tracker = [];
    List<int> context = gen.length > blockSize
        ? gen.sublist(gen.length - blockSize)
        : gen;

    final logits = model.forward(context, dummyEnc, tracker);
    List<double> lastLogits = logits.fetchRow(context.length - 1);

    int nextId = sampleLocal(lastLogits, 0.1);
    gen.add(nextId);

    print("Next -> ${itos[nextId] ?? '[UNK]'}");

    for (var t in tracker) t.dispose();
    logits.dispose();

    if (nextId == endId) break;
  }
  print("\nFinal Sequence: ${gen.map((id) => itos[id] ?? '[UNK]').join(' ')}");
}

int sampleLocal(List<double> row, double temp) {
  double maxL = row.reduce(math.max);
  List<double> exps = row.map((v) => math.exp((v - maxL) / temp)).toList();
  double sumExp = exps.reduce((a, b) => a + b);
  double r = math.Random().nextDouble() * sumExp;
  double cumulative = 0;
  for (int i = 0; i < exps.length; i++) {
    cumulative += exps[i];
    if (r <= cumulative) return i;
  }
  return 0;
}
