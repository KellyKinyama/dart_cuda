// Classic CartPole physics, ported from the OpenAI Gym implementation.
// State: cart position x, cart velocity, pole angle theta, pole angular
// velocity. Two discrete actions (0=push left, 1=push right). Reward of
// +1 per surviving step. Episode terminates when |x| > xThreshold or
// |theta| > thetaThreshold or after maxSteps.

import 'dart:math' as math;

import 'env.dart';

class CartPole extends Env {
  static const double gravity = 9.8;
  static const double massCart = 1.0;
  static const double massPole = 0.1;
  static const double totalMass = massCart + massPole;
  static const double poleLength = 0.5; // half-length
  static const double poleMassLength = massPole * poleLength;
  static const double forceMag = 10.0;
  static const double tau = 0.02; // seconds per step

  static const double thetaThreshold = 12 * math.pi / 180;
  static const double xThreshold = 2.4;

  final int maxSteps;
  final math.Random _rng;

  double _x = 0;
  double _xDot = 0;
  double _theta = 0;
  double _thetaDot = 0;
  int _steps = 0;
  bool _done = false;

  CartPole({this.maxSteps = 500, int? seed}) : _rng = math.Random(seed ?? 0);

  @override
  String get name => 'CartPole';

  @override
  int get actionCount => 2;

  @override
  int get observationSize => 4;

  @override
  List<double> reset() {
    _x = (_rng.nextDouble() - 0.5) * 0.1;
    _xDot = (_rng.nextDouble() - 0.5) * 0.1;
    _theta = (_rng.nextDouble() - 0.5) * 0.1;
    _thetaDot = (_rng.nextDouble() - 0.5) * 0.1;
    _steps = 0;
    _done = false;
    return _obs();
  }

  @override
  StepResult step(int action) {
    if (_done) return StepResult(_obs(), 0.0, true);
    final force = action == 1 ? forceMag : -forceMag;
    final cosTheta = math.cos(_theta);
    final sinTheta = math.sin(_theta);

    final temp =
        (force + poleMassLength * _thetaDot * _thetaDot * sinTheta) / totalMass;
    final thetaAcc =
        (gravity * sinTheta - cosTheta * temp) /
        (poleLength * (4.0 / 3.0 - massPole * cosTheta * cosTheta / totalMass));
    final xAcc = temp - poleMassLength * thetaAcc * cosTheta / totalMass;

    _x += tau * _xDot;
    _xDot += tau * xAcc;
    _theta += tau * _thetaDot;
    _thetaDot += tau * thetaAcc;
    _steps += 1;

    final terminated =
        _x.abs() > xThreshold ||
        _theta.abs() > thetaThreshold ||
        _steps >= maxSteps;
    _done = terminated;
    return StepResult(_obs(), 1.0, _done);
  }

  List<double> _obs() => [_x, _xDot, _theta, _thetaDot];

  @override
  String render() {
    const width = 60;
    final cartCol = (((_x + xThreshold) / (2 * xThreshold)) * (width - 1))
        .round()
        .clamp(0, width - 1);

    final sb = StringBuffer();
    // 5-row pole visualization above the track.
    final poleRows = 5;
    for (int row = 0; row < poleRows; row++) {
      final line = List<String>.filled(width, ' ');
      // Pole tilts based on theta. Compute column offset from the cart at
      // each row (top of pole = larger offset).
      final dy = poleRows - row; // row 0 is the top
      final poleCol = cartCol + (dy * math.sin(_theta) * 4).round();
      if (poleCol >= 0 && poleCol < width) {
        line[poleCol] = Ansi.fg(208, '│');
      }
      sb.writeln(line.join());
    }
    // Cart row.
    final cartLine = List<String>.filled(width, ' ');
    for (int dx = -2; dx <= 2; dx++) {
      final c = cartCol + dx;
      if (c >= 0 && c < width) cartLine[c] = Ansi.fg(33, '█');
    }
    sb.writeln(cartLine.join());
    // Track.
    sb.writeln('─' * width);
    sb.writeln(
      'x=${_x.toStringAsFixed(2)}  θ=${(_theta * 180 / math.pi).toStringAsFixed(1)}°  '
      'steps=$_steps  done=$_done',
    );
    return sb.toString();
  }
}
