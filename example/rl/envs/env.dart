// Minimal generic environment interface used by the MuZero-style agents
// in this folder. Observations are raw `List<double>` so they can be fed
// directly into the agent's representation network without going through
// any image/text tokenizer.

class StepResult {
  final List<double> observation;
  final double reward;
  final bool done;
  StepResult(this.observation, this.reward, this.done);
}

abstract class Env {
  /// Length of the observation vector returned by `reset()` / `step()`.
  int get observationSize;

  /// Number of discrete actions. Actions are integers in `[0, actionCount)`.
  int get actionCount;

  /// Human-readable name for logs / window titles.
  String get name;

  /// Reset to a starting state. Returns the initial observation.
  List<double> reset();

  /// Apply `action` and return the resulting transition.
  StepResult step(int action);

  /// Render the current state to a string. Implementations should produce
  /// terminal-friendly output (ANSI is fine). Used by the training scripts
  /// for live visualisation.
  String render();
}

/// ANSI helpers for live rendering in a terminal. No-op safe to call even
/// if the output is being piped to a file (the codes are just bytes).
class Ansi {
  static const String clearScreen = '\x1B[2J\x1B[H';
  static const String hideCursor = '\x1B[?25l';
  static const String showCursor = '\x1B[?25h';
  static String moveTo(int row, int col) => '\x1B[$row;${col}H';

  static String fg(int code, String s) => '\x1B[38;5;${code}m$s\x1B[0m';
  static String bg(int code, String s) => '\x1B[48;5;${code}m$s\x1B[0m';
}
