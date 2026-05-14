// MuZero-style RL on CartPole. Same agent / training loop as the
// GridWorld example, with a longer horizon and a continuous-state env.
//
// Run:
//   dart run example/rl/train_cart_pole.dart
//   dart run example/rl/train_cart_pole.dart --episodes=300 --eval-every=50

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_cuda/core/optimizers/adam.dart';

import 'envs/env.dart';
import 'envs/cart_pole.dart';
import 'muzero_agent.dart';
import 'training.dart';

void main(List<String> args) async {
  final flags = _parseFlags(args);
  final seed = flags.intOr('seed', 11);
  final episodes = flags.intOr('episodes', 200);
  final trainStepsPerEpisode = flags.intOr('train-steps', 8);
  final unroll = flags.intOr('unroll', 3);
  final nStep = flags.intOr('nstep', 10);
  final gamma = flags.doubleOr('gamma', 0.99);
  final lr = flags.doubleOr('lr', 1e-3);
  final epsilonStart = flags.doubleOr('epsilon-start', 1.0);
  final epsilonEnd = flags.doubleOr('epsilon-end', 0.05);
  final evalEvery = flags.intOr('eval-every', 50);
  final render = !flags.has('no-render');

  print('--- MuZero RL: CartPole ---');
  print(
    'episodes=$episodes  train_steps/ep=$trainStepsPerEpisode  '
    'unroll=$unroll  n-step=$nStep  γ=$gamma  lr=$lr',
  );

  final env = CartPole(maxSteps: 200, seed: seed);
  final agent = MuZeroAgent(
    obsSize: env.observationSize,
    actionCount: env.actionCount,
    latentSize: 32,
    hidden: 64,
  );
  final optimizer = Adam(agent.parameters(), lr: lr);
  final buffer = ReplayBuffer(capacity: 200, seed: seed + 1);
  final rng = math.Random(seed + 2);

  final returns = <double>[];
  for (int ep = 1; ep <= episodes; ep++) {
    final eps = _linearAnneal(ep, episodes, epsilonStart, epsilonEnd);
    final traj = rolloutEpisode(
      env: env,
      agent: agent,
      epsilon: eps,
      rng: rng,
    );
    buffer.add(traj);
    returns.add(traj.returnSum);

    double pSum = 0, vSum = 0, rSum = 0;
    int steps = 0;
    if (buffer.isReady) {
      for (int s = 0; s < trainStepsPerEpisode; s++) {
        final loss = trainStep(
          agent: agent,
          optimizer: optimizer,
          buffer: buffer,
          unrollSteps: unroll,
          nStep: nStep,
          gamma: gamma,
          valueWeight: 0.5,
          rewardWeight: 0.25,
        );
        pSum += loss.policy;
        vSum += loss.value;
        rSum += loss.reward;
        steps += 1;
      }
    }

    if (ep % 5 == 0 || ep == episodes) {
      final last20 =
          returns.sublist(math.max(0, returns.length - 20)).fold<double>(
                0,
                (a, b) => a + b,
              ) /
          math.min(20, returns.length);
      print(
        'ep $ep  ε=${eps.toStringAsFixed(2)}  '
        'len=${traj.length}  return=${traj.returnSum.toStringAsFixed(0)}  '
        '(avg20=${last20.toStringAsFixed(1)})  '
        'p=${(pSum / math.max(1, steps)).toStringAsFixed(3)} '
        'v=${(vSum / math.max(1, steps)).toStringAsFixed(3)} '
        'r=${(rSum / math.max(1, steps)).toStringAsFixed(3)}',
      );
    }

    if (render && ep % evalEvery == 0) {
      await _visualRollout(env, agent, label: 'ep $ep (greedy)');
    }
  }

  print('\n--- Final greedy rollout ---');
  if (render) await _visualRollout(env, agent, label: 'final');
}

Future<void> _visualRollout(
  Env env,
  MuZeroAgent agent, {
  required String label,
}) async {
  final rng = math.Random(0);
  var obs = env.reset();
  stdout.write(Ansi.hideCursor);
  try {
    for (int step = 0; step < 200; step++) {
      stdout.write(Ansi.clearScreen);
      stdout.writeln('=== $label  step $step ===');
      stdout.write(env.render());
      stdout.flush();
      final a = selectAction(agent, obs, 0.0, rng);
      final res = env.step(a);
      obs = res.observation;
      await Future<void>.delayed(const Duration(milliseconds: 30));
      if (res.done) {
        stdout.write(Ansi.clearScreen);
        stdout.writeln('=== $label  step ${step + 1} (terminal) ===');
        stdout.write(env.render());
        stdout.flush();
        break;
      }
    }
  } finally {
    stdout.write(Ansi.showCursor);
  }
  await Future<void>.delayed(const Duration(milliseconds: 500));
}

double _linearAnneal(int step, int total, double a, double b) {
  if (step >= total) return b;
  return a + (b - a) * (step / total);
}

class _Flags {
  final Map<String, String> _v = {};
  final Set<String> _bool = {};
  _Flags(List<String> args) {
    for (final a in args) {
      if (a.startsWith('--')) {
        final eq = a.indexOf('=');
        if (eq < 0) {
          _bool.add(a.substring(2));
        } else {
          _v[a.substring(2, eq)] = a.substring(eq + 1);
        }
      }
    }
  }
  bool has(String k) => _bool.contains(k);
  int intOr(String k, int d) => int.tryParse(_v[k] ?? '') ?? d;
  double doubleOr(String k, double d) => double.tryParse(_v[k] ?? '') ?? d;
}

_Flags _parseFlags(List<String> args) => _Flags(args);
