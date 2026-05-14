// MuZero-style RL on a small grid world. Trains a tiny agent end-to-end
// with policy / value / reward heads and an unrolled K-step dynamics
// loss. After training, runs a visual rollout that re-renders the env
// in-place using ANSI escape codes.
//
// Run:
//   dart run example/rl/train_grid_world.dart
// or with overrides:
//   dart run example/rl/train_grid_world.dart \
//     --episodes=200 --train-steps=4 --unroll=3 --eval-every=25
//
// All flags: --episodes --train-steps --unroll --nstep --gamma --lr
//            --epsilon-start --epsilon-end --eval-every --seed --no-render

import 'dart:io';
import 'dart:math' as math;

import 'package:dart_cuda/core/optimizers/adam.dart';

import 'envs/env.dart';
import 'envs/grid_world.dart';
import 'muzero_agent.dart';
import 'training.dart';

void main(List<String> args) async {
  final flags = _parseFlags(args);
  final seed = flags.intOr('seed', 7);
  final episodes = flags.intOr('episodes', 150);
  final trainStepsPerEpisode = flags.intOr('train-steps', 4);
  final unroll = flags.intOr('unroll', 3);
  final nStep = flags.intOr('nstep', 5);
  final gamma = flags.doubleOr('gamma', 0.97);
  final lr = flags.doubleOr('lr', 1e-3);
  final epsilonStart = flags.doubleOr('epsilon-start', 1.0);
  final epsilonEnd = flags.doubleOr('epsilon-end', 0.05);
  final evalEvery = flags.intOr('eval-every', 25);
  final render = !flags.has('no-render');
  final renderTrain = flags.has('render-train');
  final frameMs = flags.intOr('frame-ms', 80);
  final playEpisodes = flags.intOr('play', 3);

  print('--- MuZero RL: GridWorld ---');
  print(
    'episodes=$episodes  train_steps/ep=$trainStepsPerEpisode  '
    'unroll=$unroll  n-step=$nStep  γ=$gamma  lr=$lr',
  );

  final env = GridWorld(width: 6, height: 6, seed: seed);
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
    final traj = await rolloutEpisode(
      env: env,
      agent: agent,
      epsilon: eps,
      rng: rng,
      liveLabel: renderTrain ? 'ep $ep  ε=${eps.toStringAsFixed(2)}' : null,
      liveDelayMs: frameMs,
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
          valueWeight: 0.25,
          rewardWeight: 0.5,
        );
        pSum += loss.policy;
        vSum += loss.value;
        rSum += loss.reward;
        steps += 1;
      }
    }

    if (ep % 5 == 0 || ep == episodes) {
      final last20 =
          returns
              .sublist(math.max(0, returns.length - 20))
              .fold<double>(0, (a, b) => a + b) /
          math.min(20, returns.length);
      print(
        'ep $ep  ε=${eps.toStringAsFixed(2)}  '
        'len=${traj.length}  return=${traj.returnSum.toStringAsFixed(2)}  '
        '(avg20=${last20.toStringAsFixed(2)})  '
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
  if (render && playEpisodes > 0) {
    for (int i = 1; i <= playEpisodes; i++) {
      final playEnv = GridWorld(width: 6, height: 6, seed: seed + 100 + i);
      final playRng = math.Random(seed + 200 + i);
      final traj = await rolloutEpisode(
        env: playEnv,
        agent: agent,
        epsilon: 0.0,
        rng: playRng,
        liveLabel: 'play $i/$playEpisodes',
        liveDelayMs: frameMs,
      );
      print(
        'play $i  len=${traj.length}  '
        'return=${traj.returnSum.toStringAsFixed(2)}',
      );
      await Future<void>.delayed(const Duration(milliseconds: 600));
    }
  }
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
    for (int step = 0; step < 60; step++) {
      stdout.write(Ansi.clearScreen);
      stdout.writeln('=== $label  step $step ===');
      stdout.write(env.render());
      await stdout.flush();
      final a = selectAction(agent, obs, 0.0, rng);
      final res = env.step(a);
      obs = res.observation;
      await Future<void>.delayed(const Duration(milliseconds: 80));
      if (res.done) {
        stdout.write(Ansi.clearScreen);
        stdout.writeln('=== $label  step ${step + 1} (terminal) ===');
        stdout.write(env.render());
        await stdout.flush();
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
