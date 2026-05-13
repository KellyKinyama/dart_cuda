// file: lib/mu_zero/deepseek_aft_decoder.dart
//
// DeepSeek-style Mixture-of-Experts decoder built on top of this codebase's
// GPU `Tensor` autograd and AFT attention blocks. Mirrors the architectural
// shape of `TransformerDecoder`/`TransformerDecoderBlock` but replaces the
// dense FeedForward with a routed MoE FFN containing both routed experts
// (sparse, top-K) and shared experts (always-on), as in DeepSeekMoE / V3.
//
// Routing is differentiable through the router weights via softmax scores,
// while the discrete top-K selection is a CPU-side mask. An aux-loss-free
// load balancing bias (DeepSeek-V3 style) is updated by `updateRoutingBias`.

import 'dart:math' as math;

import 'package:dart_cuda/core/attention/aft_multi_head_attention.dart';
import 'package:dart_cuda/core/attention/aft_multi_head_cross_attention.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/layers/layer_norm.dart';
import 'package:dart_cuda/core/layers/nn.dart';

/// A simple GELU-FFN expert: D -> H (GELU) -> D.
class Expert extends Module {
  final Layer w1;
  final Layer w2;

  Expert(int dim, int hiddenSize)
    : w1 = Layer(dim, hiddenSize, useGelu: true),
      w2 = Layer(hiddenSize, dim, useGelu: false);

  Tensor forward(Tensor x, List<Tensor> tracker) {
    final h = w1.forward(x, tracker);
    return w2.forward(h, tracker);
  }

  @override
  List<Tensor> parameters() => [...w1.parameters(), ...w2.parameters()];
}

/// DeepSeek-style MoE FFN: top-K routed experts + always-on shared experts.
///
/// Forward steps for an input `x` of shape `[T, D]`:
///   1. Router logits `g = x @ Wg`              -> `[T, E]`
///   2. Softmax scores `s = softmax(g)`         -> `[T, E]`  (differentiable)
///   3. CPU-side: pick top-K indices per token using `s + routingBias`
///      (bias is non-differentiable; trained only by `updateRoutingBias`).
///   4. Build a `[T, E]` 0/1 selection mask and multiply elementwise:
///      `m = s * mask`. Gradients flow into Wg through the kept entries.
///   5. For each routed expert `e`, broadcast its column of `m` over `D` via
///      a precomputed `[E, D]` selector matrix and multiply by `expert_e(x)`.
///   6. Add outputs from the shared experts (each computed on full `x`).
class MoEFeedForward extends Module {
  final int embedSize;
  final int numRoutedExperts;
  final int numSharedExperts;
  final int topK;
  final int expertHiddenSize;

  final Tensor gateW; // [embedSize, numRoutedExperts]
  final List<Expert> routedExperts;
  final List<Expert> sharedExperts;

  /// DeepSeek-V3 aux-loss-free routing bias (CPU, non-differentiable).
  final List<double> routingBias;

  /// How many tokens picked each routed expert since last bias update.
  final List<int> _expertLoad;

  /// Bias update step size (DeepSeek-V3 calls this `gamma`).
  final double biasUpdateRate;

  MoEFeedForward({
    required this.embedSize,
    required this.numRoutedExperts,
    required this.numSharedExperts,
    required this.topK,
    required this.expertHiddenSize,
    this.biasUpdateRate = 0.001,
  }) : gateW = Tensor.random([embedSize, numRoutedExperts]),
       routedExperts = List.generate(
         numRoutedExperts,
         (_) => Expert(embedSize, expertHiddenSize),
       ),
       sharedExperts = List.generate(
         numSharedExperts,
         (_) => Expert(embedSize, expertHiddenSize),
       ),
       routingBias = List<double>.filled(numRoutedExperts, 0.0),
       _expertLoad = List<int>.filled(numRoutedExperts, 0);

  /// Build the constant `[E, D]` selector matrix for expert `e`:
  /// row `e` = ones, all other rows = zeros. When right-multiplied by a
  /// `[T, E]` weight matrix, this extracts column `e` and broadcasts it
  /// across `D`.
  Tensor _buildSelector(int e) {
    final vals = List<double>.filled(numRoutedExperts * embedSize, 0.0);
    for (int d = 0; d < embedSize; d++) {
      vals[e * embedSize + d] = 1.0;
    }
    return Tensor.fromList([numRoutedExperts, embedSize], vals);
  }

  Tensor forward(Tensor x, List<Tensor> tracker) {
    final int T = x.shape[0];
    final int E = numRoutedExperts;

    // 1. Router logits + differentiable softmax scores.
    final gateLogits = x.matmul(gateW);
    final gateScores = gateLogits.softmax();
    tracker.addAll([gateLogits, gateScores]);

    // 2. CPU top-K selection with routing bias (load balancing).
    final scoresCpu = gateScores.fetchData();
    final maskVals = List<double>.filled(T * E, 0.0);
    for (int t = 0; t < T; t++) {
      final indexed = List<MapEntry<int, double>>.generate(
        E,
        (e) => MapEntry(e, scoresCpu[t * E + e] + routingBias[e]),
      );
      indexed.sort((a, b) => b.value.compareTo(a.value));
      final k = topK < E ? topK : E;
      for (int i = 0; i < k; i++) {
        final e = indexed[i].key;
        maskVals[t * E + e] = 1.0;
        _expertLoad[e]++;
      }
    }

    final mask = Tensor.fromList([T, E], maskVals);
    final maskedScores =
        gateScores * mask; // [T, E], differentiable in gateScores
    tracker.addAll([mask, maskedScores]);

    // 3. Combine routed experts via selector-matmul broadcast.
    Tensor? acc;
    for (int e = 0; e < E; e++) {
      final selector = _buildSelector(e); // [E, D]
      final colBroadcast = maskedScores.matmul(selector); // [T, D]
      final expertOut = routedExperts[e].forward(x, tracker); // [T, D]
      final weighted = colBroadcast * expertOut; // [T, D]
      tracker.addAll([selector, colBroadcast, weighted]);
      if (acc == null) {
        acc = weighted;
      } else {
        final next = acc + weighted;
        tracker.add(next);
        acc = next;
      }
    }

    // 4. Add always-on shared experts.
    for (final se in sharedExperts) {
      final out = se.forward(x, tracker);
      if (acc == null) {
        acc = out;
      } else {
        final next = acc + out;
        tracker.add(next);
        acc = next;
      }
    }

    return acc!;
  }

  /// DeepSeek-V3 aux-loss-free balancing: shift bias up for under-loaded
  /// experts and down for over-loaded ones, then reset the load counters.
  /// Call this once per epoch (or every N steps).
  void updateRoutingBias() {
    final total = _expertLoad.fold<int>(0, (a, b) => a + b);
    if (total == 0) return;
    final mean = total / numRoutedExperts;
    if (mean == 0) return;
    for (int e = 0; e < numRoutedExperts; e++) {
      final delta = (mean - _expertLoad[e]) / mean;
      routingBias[e] += biasUpdateRate * delta;
      _expertLoad[e] = 0;
    }
  }

  /// Read-only snapshot of the current per-expert token counts (debug only).
  List<int> get expertLoad => List<int>.unmodifiable(_expertLoad);

  @override
  List<Tensor> parameters() => [
    gateW,
    ...routedExperts.expand((e) => e.parameters()),
    ...sharedExperts.expand((e) => e.parameters()),
  ];
}

/// One DeepSeek-AFT decoder block: AFT self-attn + AFT cross-attn + MoE FFN,
/// each with Pre-LayerNorm and a residual connection.
class DeepSeekAFTDecoderBlock extends Module {
  final MultiHeadAFT selfAttention;
  final MultiHeadAFTCross crossAttention;
  final MoEFeedForward moe;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;
  final int embedSize;

  DeepSeekAFTDecoderBlock(
    this.embedSize,
    int numHeads,
    int encoderEmbedSize,
    int maxSeqLen, {
    required int numRoutedExperts,
    required int numSharedExperts,
    required int topK,
    required int expertHiddenSize,
  }) : selfAttention = MultiHeadAFT(
         numHeads,
         embedSize,
         maxSeqLen,
         masked: true,
       ),
       crossAttention = MultiHeadAFTCross(
         numHeads,
         embedSize,
         encoderEmbedSize,
         maxSeqLen,
         maxSeqLen,
       ),
       moe = MoEFeedForward(
         embedSize: embedSize,
         numRoutedExperts: numRoutedExperts,
         numSharedExperts: numSharedExperts,
         topK: topK,
         expertHiddenSize: expertHiddenSize,
       ),
       ln1 = LayerNorm(embedSize),
       ln2 = LayerNorm(embedSize),
       ln3 = LayerNorm(embedSize) {
    _initializeWeights();
  }

  /// Same conservative init as `TransformerDecoderBlock` for AFT stability.
  void _initializeWeights() {
    final rand = math.Random();
    for (final p in parameters()) {
      final data = p.fetchData();
      for (int i = 0; i < data.length; i++) {
        data[i] = (rand.nextDouble() * 2 - 1) * 0.02;
      }
      p.data = data;
    }
  }

  Tensor forward(Tensor xDec, Tensor xEnc, List<Tensor> tracker) {
    // 1. Masked self-attention (Pre-LN).
    final xNorm1 = ln1.forward(xDec, tracker);
    final selfAttnOut = selfAttention.forward(xNorm1, tracker);
    final xRes1 = xDec + selfAttnOut;
    tracker.add(xRes1);

    // 2. Cross-attention (Pre-LN).
    final xNorm2 = ln2.forward(xRes1, tracker);
    final crossAttnOut = crossAttention.forward(xNorm2, xEnc, tracker);
    final xRes2 = xRes1 + crossAttnOut;
    tracker.add(xRes2);

    // 3. MoE FFN (Pre-LN).
    final xNorm3 = ln3.forward(xRes2, tracker);
    final moeOut = moe.forward(xNorm3, tracker);
    final out = xRes2 + moeOut;
    tracker.add(out);

    return out;
  }

  @override
  List<Tensor> parameters() => [
    ...selfAttention.parameters(),
    ...crossAttention.parameters(),
    ...moe.parameters(),
    ...ln1.parameters(),
    ...ln2.parameters(),
    ...ln3.parameters(),
  ];
}

/// Top-level DeepSeek-style decoder: token + position embeddings,
/// a stack of [DeepSeekAFTDecoderBlock]s, final LayerNorm, and an LM head.
class DeepSeekAFTDecoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;
  final int encoderEmbedSize;

  final Tensor wte; // [vocabSize, embedSize]
  final Tensor wpe; // [blockSize, embedSize]
  final List<DeepSeekAFTDecoderBlock> blocks;
  final LayerNorm finalLayerNorm;
  final Layer lmHead;

  DeepSeekAFTDecoder({
    required this.vocabSize,
    this.embedSize = 64,
    this.blockSize = 16,
    this.numLayers = 2,
    this.numHeads = 4,
    int? encoderEmbedSize,
    int numRoutedExperts = 4,
    int numSharedExperts = 1,
    int topK = 2,
    int expertHiddenSize = 64,
  }) : encoderEmbedSize = encoderEmbedSize ?? embedSize,
       wte = Tensor.random([vocabSize, embedSize]),
       wpe = Tensor.random([blockSize, embedSize]),
       blocks = List.generate(
         numLayers,
         (_) => DeepSeekAFTDecoderBlock(
           embedSize,
           numHeads,
           encoderEmbedSize ?? embedSize,
           blockSize,
           numRoutedExperts: numRoutedExperts,
           numSharedExperts: numSharedExperts,
           topK: topK,
           expertHiddenSize: expertHiddenSize,
         ),
       ),
       finalLayerNorm = LayerNorm(embedSize),
       lmHead = Layer(embedSize, vocabSize, useGelu: false) {
    // Small embedding init for stability.
    final rawWte = wte.fetchData();
    final rand = math.Random();
    for (int i = 0; i < rawWte.length; i++) {
      rawWte[i] = (rand.nextDouble() * 2 - 1) * 0.02;
    }
    wte.data = rawWte;

    // Zero the LM head bias to avoid index-collapse early in training.
    final params = lmHead.parameters();
    if (params.length > 1) {
      final bias = params[1];
      bias.data = List<double>.filled(bias.length, 0.0);
    }
  }

  Tensor forward(List<int> idx, Tensor encoderOutput, List<Tensor> tracker) {
    final int T = idx.length;
    if (T > blockSize) {
      throw ArgumentError(
        'Sequence length $T exceeds max block size $blockSize',
      );
    }
    Tensor x = Tensor.embeddings(idx, wte, wpe);
    tracker.add(x);
    for (final block in blocks) {
      x = block.forward(x, encoderOutput, tracker);
    }
    final xNorm = finalLayerNorm.forward(x, tracker);
    return lmHead.forward(xNorm, tracker);
  }

  /// Update the aux-loss-free routing bias on every MoE block. Call once per
  /// epoch (or every N steps) to rebalance expert load.
  void updateRoutingBias() {
    for (final block in blocks) {
      block.moe.updateRoutingBias();
    }
  }

  @override
  List<Tensor> parameters() => [
    wte,
    wpe,
    ...blocks.expand((b) => b.parameters()),
    ...finalLayerNorm.parameters(),
    ...lmHead.parameters(),
  ];
}
