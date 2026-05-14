// 2D grid world: agent starts at the top-left, must reach the goal at the
// bottom-right while avoiding walls. Reward is -0.01 per step, -1 for
// hitting a wall, +1 for reaching the goal. Episode ends on goal or after
// `maxSteps` steps.
//
// Observation: one-hot of agent (x,y) position concatenated with a flat
// wall map. Small enough that a tiny MLP can learn it quickly.
//
// Actions: 0=up, 1=right, 2=down, 3=left.

import 'dart:math' as math;

import 'env.dart';

class GridWorld extends Env {
  final int width;
  final int height;
  final int maxSteps;
  final List<List<bool>> walls;
  final math.Random _rng;

  int _x = 0;
  int _y = 0;
  int _steps = 0;
  late final int _goalX;
  late final int _goalY;
  bool _done = false;
  bool _hitWall = false;

  GridWorld({
    this.width = 6,
    this.height = 6,
    this.maxSteps = 80,
    int? seed,
    double wallProb = 0.18,
  }) : walls = List.generate(6, (_) => List.filled(6, false)),
       _rng = math.Random(seed ?? 0) {
    // Generate a random wall layout that leaves start/goal reachable.
    // (We don't run a full BFS check — `wallProb` is small enough that the
    // agent can usually find a path; truly blocked layouts just produce a
    // hard episode and are still useful training signal.)
    _goalX = width - 1;
    _goalY = height - 1;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final atStart = (x == 0 && y == 0);
        final atGoal = (x == _goalX && y == _goalY);
        walls[y][x] = !atStart && !atGoal && _rng.nextDouble() < wallProb;
      }
    }
  }

  @override
  String get name => 'GridWorld(${width}x$height)';

  @override
  int get actionCount => 4;

  @override
  int get observationSize => width * height + width * height;

  @override
  List<double> reset() {
    _x = 0;
    _y = 0;
    _steps = 0;
    _done = false;
    _hitWall = false;
    return _obs();
  }

  @override
  StepResult step(int action) {
    if (_done) {
      return StepResult(_obs(), 0.0, true);
    }
    int nx = _x;
    int ny = _y;
    switch (action) {
      case 0:
        ny -= 1;
        break; // up
      case 1:
        nx += 1;
        break; // right
      case 2:
        ny += 1;
        break; // down
      case 3:
        nx -= 1;
        break; // left
    }
    double r = -0.01;
    _hitWall = false;
    if (nx < 0 || nx >= width || ny < 0 || ny >= height || walls[ny][nx]) {
      // Bounced; stay in place.
      r = -0.1;
      _hitWall = true;
    } else {
      _x = nx;
      _y = ny;
    }
    _steps += 1;
    if (_x == _goalX && _y == _goalY) {
      r = 1.0;
      _done = true;
    } else if (_steps >= maxSteps) {
      _done = true;
    }
    return StepResult(_obs(), r, _done);
  }

  List<double> _obs() {
    final out = List<double>.filled(observationSize, 0.0);
    out[_y * width + _x] = 1.0;
    final wallOffset = width * height;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        out[wallOffset + y * width + x] = walls[y][x] ? 1.0 : 0.0;
      }
    }
    return out;
  }

  @override
  String render() {
    final sb = StringBuffer();
    sb.writeln('╔${'═' * (width * 2)}╗');
    for (int y = 0; y < height; y++) {
      sb.write('║');
      for (int x = 0; x < width; x++) {
        if (x == _x && y == _y) {
          sb.write(Ansi.fg(220, '◉ '));
        } else if (x == _goalX && y == _goalY) {
          sb.write(Ansi.fg(46, '★ '));
        } else if (walls[y][x]) {
          sb.write(Ansi.fg(240, '▓▓'));
        } else {
          sb.write('  ');
        }
      }
      sb.writeln('║');
    }
    sb.writeln('╚${'═' * (width * 2)}╝');
    sb.writeln(
      'pos=($_x,$_y)  steps=$_steps  done=$_done  ${_hitWall ? "(bumped wall)" : ""}',
    );
    return sb.toString();
  }
}
