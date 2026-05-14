// Replay buffer + n-step return computation + MuZero-style K-step
// unrolled training step. Designed to be small and reusable across the
// `train_grid_world.dart` / `train_cart_pole.dart` scripts.

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_cuda/core/optimizers/adam.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

import 'envs/env.dart';
import 'muzero_agent.dart';

class Transition {
  final List<double> obs;
  final int action;
  final double reward;
  final bool done;
  Transition(this.obs, this.action, this.reward, this.done);
}

class Trajectory {
  final List<Transition> steps = [];
  double returnSum = 0;
  void add(Transition t) {
    steps.add(t);
    returnSum += t.reward;
  }

  int get length => steps.length;
}

class ReplayBuffer {
  final int capacity;
  final List<Trajectory> _episodes = [];
  final math.Random _rng;
  ReplayBuffer({this.capacity = 200, int? seed}) : _rng = math.Random(seed);

  void add(Trajectory t) {
    _episodes.add(t);
    if (_episodes.length > capacity) _episodes.removeAt(0);
  }

  bool get isReady => _episodes.isNotEmpty;
  int get size => _episodes.length;

  /// Sample a (trajectory, start-index) pair such that at least one step
  /// is available from `start`.
  ({Trajectory traj, int start}) sample() {
    final t = _episodes[_rng.nextInt(_episodes.length)];
    final start = _rng.nextInt(t.length);
    return (traj: t, start: start);
  }
}

/// n-step return G_t = sum_{i=0..n-1} γ^i r_{t+i} + γ^n * V_bootstrap.
/// `vBootstrap` may be null (e.g. terminal); treated as 0 in that case.
double nStepReturn(
  List<Transition> steps,
  int t,
  int n,
  double gamma,
  double? vBootstrap,
) {
  double g = 0.0;
  double discount = 1.0;
  bool truncated = false;
  for (int i = 0; i < n; i++) {
    final idx = t + i;
    if (idx >= steps.length) {
      truncated = true;
      break;
    }
    g += discount * steps[idx].reward;
    discount *= gamma;
    if (steps[idx].done) {
      truncated = true;
      break;
    }
  }
  if (!truncated && vBootstrap != null) {
    g += discount * vBootstrap;
  }
  return g;
}

/// One MuZero training step: sample a (trajectory, start), run h() on the
/// observation at `start`, then unroll `unrollSteps` of g(), supervising
/// each step's policy / reward / value heads against targets computed
/// from the trajectory.
({double total, double policy, double value, double reward}) trainStep({
  required MuZeroAgent agent,
  required Adam optimizer,
  required ReplayBuffer buffer,
  required int unrollSteps,
  required int nStep,
  required double gamma,
  required double valueWeight,
  required double rewardWeight,
}) {
  optimizer.zeroGrad();
  final tracker = <Tensor>[];

  final sample = buffer.sample();
  final traj = sample.traj;
  int t = sample.start;

  // 1. Initial state from observation at t.
  Tensor state = agent.representation(traj.steps[t].obs, tracker);

  Tensor? totalLoss;
  double pSum = 0, vSum = 0, rSum = 0;
  int unrolled = 0;

  for (int k = 0; k <= unrollSteps; k++) {
    if (t + k >= traj.length) break;

    // Prediction at this step.
    final pred = agent.prediction(state, tracker);
    final tr = traj.steps[t + k];

    // Policy target = observed action (one-hot via cross-entropy index).
    final pLoss = pred.policy.crossEntropy([tr.action]);
    tracker.add(pLoss);
    pSum += pLoss.fetchData()[0];

    // Value target = n-step return from t+k.
    final vTarget = nStepReturn(traj.steps, t + k, nStep, gamma, 0.0);
    final vTargetT = Tensor.fromList([1, 1], [vTarget]);
    tracker.add(vTargetT);
    final vLoss = pred.value.mseLoss(vTargetT);
    tracker.add(vLoss);
    vSum += vLoss.fetchData()[0];

    final scaledV = vLoss * Tensor.fill([1, 1], valueWeight);
    tracker.add(scaledV);

    Tensor stepLoss = pLoss + scaledV;
    tracker.add(stepLoss);

    // Dynamics step (don't run on the last unrolled step).
    if (k < unrollSteps && t + k + 1 < traj.length) {
      final dyn = agent.dynamics(state, tr.action, tracker);
      // Reward target = observed reward at t+k.
      final rTargetT = Tensor.fromList([1, 1], [tr.reward]);
      tracker.add(rTargetT);
      final rLoss = dyn.reward.mseLoss(rTargetT);
      tracker.add(rLoss);
      rSum += rLoss.fetchData()[0];
      final scaledR = rLoss * Tensor.fill([1, 1], rewardWeight);
      tracker.add(scaledR);
      stepLoss = stepLoss + scaledR;
      tracker.add(stepLoss);

      state = dyn.nextState;
    }

    totalLoss = totalLoss == null ? stepLoss : totalLoss + stepLoss;
    if (totalLoss != stepLoss) tracker.add(totalLoss);
    unrolled += 1;
  }

  if (totalLoss == null) {
    for (final t in tracker) {
      t.dispose();
    }
    return (total: 0.0, policy: 0.0, value: 0.0, reward: 0.0);
  }

  final totalVal = totalLoss.fetchData()[0];
  totalLoss.backward();
  optimizer.step();

  // Cleanup all per-step tensors (params are excluded by virtue of not
  // being added to the tracker).
  final paramAddrs = agent.parameters().map((p) => p.handle.address).toSet();
  final freed = <int>{};
  for (final t in tracker) {
    final a = t.handle.address;
    if (a == 0 || freed.contains(a) || paramAddrs.contains(a)) continue;
    if (t.isView == true) continue;
    t.dispose();
    freed.add(a);
  }

  return (
    total: totalVal,
    policy: pSum / unrolled,
    value: vSum / unrolled,
    reward: unrolled > 1 ? rSum / (unrolled - 1) : 0.0,
  );
}

/// Greedy / epsilon-greedy action over the agent's predicted policy.
int selectAction(
  MuZeroAgent agent,
  List<double> obs,
  double epsilon,
  math.Random rng,
) {
  if (rng.nextDouble() < epsilon) {
    return rng.nextInt(agent.actionCount);
  }
  final tracker = <Tensor>[];
  final s = agent.representation(obs, tracker);
  final pred = agent.prediction(s, tracker);
  final logits = pred.policy.fetchData();
  // argmax
  int best = 0;
  double bestV = logits[0];
  for (int i = 1; i < logits.length; i++) {
    if (logits[i] > bestV) {
      bestV = logits[i];
      best = i;
    }
  }
  for (final t in tracker) {
    t.dispose();
  }
  return best;
}

/// Run one episode collecting transitions. Returns the trajectory and
/// the total undiscounted return. If [liveLabel] is non-null, renders
/// the env in-place each step (ANSI clear + redraw) with [liveDelayMs]
/// between frames.
Future<Trajectory> rolloutEpisode({
  required Env env,
  required MuZeroAgent agent,
  required double epsilon,
  required math.Random rng,
  int? maxSteps,
  String? liveLabel,
  int liveDelayMs = 30,
}) async {
  final traj = Trajectory();
  List<double> obs = env.reset();
  final live = liveLabel != null;
  if (live) stdout.write(Ansi.hideCursor);
  try {
    for (int t = 0; ; t++) {
      if (maxSteps != null && t >= maxSteps) break;
      if (live) {
        stdout.write(Ansi.clearScreen);
        stdout.writeln('=== $liveLabel  step $t ===');
        stdout.write(env.render());
        await stdout.flush();
      }
      final a = selectAction(agent, obs, epsilon, rng);
      final res = env.step(a);
      traj.add(Transition(obs, a, res.reward, res.done));
      obs = res.observation;
      if (live) await Future<void>.delayed(Duration(milliseconds: liveDelayMs));
      if (res.done) {
        if (live) {
          stdout.write(Ansi.clearScreen);
          stdout.writeln('=== $liveLabel  step ${t + 1} (terminal) ===');
          stdout.write(env.render());
          await stdout.flush();
        }
        break;
      }
    }
  } finally {
    if (live) stdout.write(Ansi.showCursor);
  }
  return traj;
}
