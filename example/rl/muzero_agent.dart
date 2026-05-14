// Generic MuZero-style agent over vector observations and a discrete
// action space. Three networks:
//
//   h(o)      : observation -> latent state s            (representation)
//   g(s, a)   : (state, one-hot action) -> (s', r)       (dynamics + reward)
//   f(s)      : state -> (policy_logits, value)          (prediction)
//
// All three are small MLPs built from `lib/core/layers/mlp.dart`. We
// concatenate the action one-hot onto the latent in dynamics; rewards and
// values are scalar heads.

import 'package:dart_cuda/core/layers/mlp.dart';
import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

class MuZeroAgent extends Module {
  final int obsSize;
  final int actionCount;
  final int latentSize;

  final MLP _h;
  final MLP _gState;
  final MLP _gReward;
  final MLP _fPolicy;
  final MLP _fValue;

  MuZeroAgent({
    required this.obsSize,
    required this.actionCount,
    this.latentSize = 32,
    int hidden = 64,
  }) : _h = MLP(obsSize, [hidden, latentSize]),
       _gState = MLP(latentSize + actionCount, [hidden, latentSize]),
       _gReward = MLP(latentSize + actionCount, [hidden, 1]),
       _fPolicy = MLP(latentSize, [hidden, actionCount]),
       _fValue = MLP(latentSize, [hidden, 1]);

  Tensor representation(List<double> obs, List<Tensor> tracker) {
    final x = Tensor.fromList([1, obsSize], obs);
    tracker.add(x);
    return _h.forward(x, tracker);
  }

  ({Tensor nextState, Tensor reward}) dynamics(
    Tensor state,
    int action,
    List<Tensor> tracker,
  ) {
    final aOneHot = List<double>.filled(actionCount, 0.0);
    aOneHot[action] = 1.0;
    final aT = Tensor.fromList([1, actionCount], aOneHot);
    tracker.add(aT);
    final concat = Tensor.concat([state, aT]);
    tracker.add(concat);
    final next = _gState.forward(concat, tracker);
    final reward = _gReward.forward(concat, tracker);
    return (nextState: next, reward: reward);
  }

  ({Tensor policy, Tensor value}) prediction(
    Tensor state,
    List<Tensor> tracker,
  ) {
    final p = _fPolicy.forward(state, tracker);
    final v = _fValue.forward(state, tracker);
    return (policy: p, value: v);
  }

  @override
  List<Tensor> parameters() => [
    ..._h.parameters(),
    ..._gState.parameters(),
    ..._gReward.parameters(),
    ..._fPolicy.parameters(),
    ..._fValue.parameters(),
  ];
}
