nvcc --shared -o native/lib/libmat_mul.so native/src/engine.cu -Xcompiler -fPIC

<!-- nvcc --shared -o native/lib/dart_cuda.so native/src/dart_cuda.cu -Xcompiler -fPIC -->



nvcc --shared -o native/lib/libmat_mul.so native/src/engine_v2.cu -Xcompiler -fPIC

native/src/engine_v2.cu


## MuZero chess

Train MuZero vs Stockfish (fresh run, save checkpoint each iter):

```bash
dart run example/tool/muzero_vs_stockfish_train.dart example/tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=10 \
  --sf-movetime=1000 --sf-skill=20 \
 --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1
```

Resume training from checkpoint:

```bash
dart run example/tool/muzero_vs_stockfish_train.dart example/tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=40 \
  --sf-movetime=100 --sf-skill=0 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1
```

Play a UCI match between the MuZero engine and Stockfish:

```bash
dart run example/tool/uci_match.dart \
  "dart run example/mu_zero/muzero_chess_uci.dart --no-train" \
  example/tools/stockfish --movetime=300 --maxply=80
```

Run the MuZero engine standalone (UCI on stdio, e.g. for a chess GUI):

```bash
dart run example/mu_zero/muzero_chess_uci.dart            # trains then enters UCI loop
dart run example/mu_zero/muzero_chess_uci.dart --no-train  # quick handshake test
```

Enable Zobrist-keyed PUCT MCTS at play time (works in any UCI GUI / match):

```
setoption name MctsSims value 64
```

Train with MCTS exploration for MuZero's self-play moves (targets still
Stockfish's move — distillation):

```bash
dart run example/tool/muzero_vs_stockfish_train.dart example/tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=80 \
  --sf-movetime=300 --sf-skill=20 \
  --mcts-sims=32 --mcts-temp=1.0 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1
```

Self-play demo (train then play one greedy game):

```bash
dart run lib/mu_zero/muzero_chess_player.dart
```

Build Stockfish (Linux) from bundled source:

```bash
cd example/tools/sf/stockfish/src && make -j$(nproc) build ARCH=x86-64-avx2
cp stockfish ../../../stockfish && chmod +x ../../../stockfish
```
dart run example/tool/muzero_vs_stockfish_train.dart example/tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=80 \
  --sf-movetime=300 --sf-skill=20 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1


  dart run example/tool/muzero_vs_stockfish_train.dart example/tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=80 \
  --sf-movetime=300 --sf-skill=20 \
  --mcts-sims=32 --mcts-temp=1.0 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1

  
  dart run example/tool/muzero_vs_stockfish_train.dart example/tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=80 \
  --sf-movetime=300 --sf-skill=20 \
  --mcts-sims=3 --mcts-temp=1.0 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1


## MuZero AlphaZero self-play training

Drop-in replacement for `muzero_vs_stockfish_train.dart` that uses pure
self-play (both sides MCTS, no external engine) and trains policy +
value heads on the resulting (move, game-outcome) pairs. Same checkpoint
format as the Stockfish trainer (`_AgentModule` wraps decoder + value
head), so `muzero_chess.bin` round-trips between the two trainers.

### Quick smoke run (fresh, no checkpoint)

```bash
dart run example/tool/muzero_alphazero_train.dart \
  --iters=2 --games=2 --epochs=1 --maxply=20 \
  --mcts-sims=8 --temperature=1.0 --temp-moves=10 \
  --save=muzero_chess.bin --save-every=1
```

### Equivalent of the Stockfish command on the same checkpoint

```bash
dart run example/tool/muzero_alphazero_train.dart \
  --iters=3 --games=2 --epochs=2 --maxply=10 \
  --mcts-sims=32 --temperature=1.0 --temp-moves=15 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1
```

### Resume training from checkpoint (longer games, more search)

```bash
dart run example/tool/muzero_alphazero_train.dart \
  --iters=10 --games=4 --epochs=2 --maxply=80 \
  --mcts-sims=64 --cpuct=1.4 --temperature=1.0 --temp-moves=20 \
  --dirichlet-alpha=0.3 --dirichlet-eps=0.25 \
  --replay-size=20000 --replay-batch=512 \
  --lr=5e-4 --value-weight=0.5 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1
```

`--replay-size` keeps a FIFO buffer of `(history, sampled-move,
value-target)` triples across iterations; each training epoch then
samples `--replay-batch` of them (without replacement). 0 disables the
buffer (legacy: train only on the freshest pairs).

`--dirichlet-alpha=0.3 --dirichlet-eps=0.25` mixes symmetric
Dirichlet(α) noise into the **root** priors before each MCTS search:
`P_i ← (1 − ε)·P_i + ε·η_i`. These are the AlphaZero chess defaults and
keep self-play from collapsing to the same lines. Set `--dirichlet-eps=0`
to disable. Noise is applied **idempotently** (re-mixed from a stored
clean snapshot each turn) so subtree reuse never compounds it.

**Subtree reuse** is on by default: one `ZobristMcts` instance is kept
per game so the transposition table — and therefore the search subtree
below the move just played — is preserved as next move's root
expansion. Pass `--no-subtree-reuse` to fall back to the legacy
fresh-MCTS-per-move behavior. The win grows with `--mcts-sims`: at
≤16 sims it's invisible, at 64–256 sims it can roughly halve search
wall-clock per game.

### Eval gate (best-checkpoint ladder)

Periodically play the freshly-trained agent against a saved "best"
checkpoint; only promote on a strong-enough win, otherwise roll back.
This prevents a bad iteration from quietly overwriting a good model.

```bash
dart run example/tool/muzero_alphazero_train.dart \
  --iters=20 --games=4 --epochs=2 --maxply=80 \
  --mcts-sims=64 --temperature=1.0 --temp-moves=20 \
  --dirichlet-alpha=0.3 --dirichlet-eps=0.25 \
  --replay-size=20000 --replay-batch=512 \
  --eval-every=2 --eval-games=8 --eval-threshold=0.55 \
  --eval-sims=32 --eval-maxply=80 \
  --load=muzero_chess.bin --save=muzero_chess.bin --best=muzero_chess.best.bin
```

What the gate does at the end of each `--eval-every` iteration:

1. If `<best>` doesn't exist yet: snapshot the current `<save>` to
   `<best>` and continue.
2. Otherwise load `<best>` into a separate agent and play `--eval-games`
   games (alternating colors) at `--eval-sims` sims/move, greedy, **no
   Dirichlet noise**.
3. If candidate's score `(W + 0.5·D + 0.5·unfinished) / games ≥
   --eval-threshold`: copy `<save>` over `<best>` (PROMOTED).
4. Otherwise: load `<best>` back into the live training agent and
   overwrite `<save>` with `<best>` (REJECTED, rolled back). Adam
   moments are preserved — only the network weights revert.

Defaults: `--eval-every=0` (disabled), `--eval-games=4`,
`--eval-threshold=0.55`, `--eval-maxply=60`, `--eval-sims=16`,
`--best=<save>.best`. The gate is a no-op without `--save`.

### Watch the games as they play

UCI move list with MCTS visit count and prior for each chosen move:

```bash
dart run example/tool/muzero_alphazero_train.dart \
  --iters=3 --games=2 --epochs=2 --maxply=10 \
  --mcts-sims=32 --temperature=1.0 --temp-moves=15 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1 \
  --show-moves
```

Per-move output looks like:

```
  game 1/2 (self-play) ...
      1.W e2e4   (visits=18/32, prior=0.142)
      1.B e7e5   (visits=15/32, prior=0.121)
      2.W g1f3   (visits=12/32, prior=0.098)
      ...
    fen: <final FEN>
10 plies, unfinished (maxply, z=0)
```

Add `--show-board` to also print the ASCII board after every move
(implies `--show-moves`).

### Flags

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--iters=N` | 3 | Outer iterations: each plays N games then trains. |
| `--games=N` | 2 | Self-play games per iteration. |
| `--epochs=N` | 2 | Training passes over the iteration's pairs. |
| `--maxply=N` | 40 | Cap plies per game (hits → outcome `z=0`). |
| `--mcts-sims=N` | 32 | PUCT simulations per move. |
| `--cpuct=F` | 1.4 | PUCT exploration constant. |
| `--temperature=F` | 1.0 | Visit-distribution sampling temperature. |
| `--temp-moves=N` | 15 | After this many plies, switch to greedy (most-visited). |
| `--dirichlet-alpha=F` | 0.3 | Symmetric Dirichlet(α) for root prior noise. |
| `--dirichlet-eps=F` | 0.25 | Mix weight of root noise (0 = disabled). |
| `--replay-size=N` | 0 | FIFO replay buffer capacity (0 = disabled, fresh pairs only). |
| `--replay-batch=N` | 0 | Pairs sampled per epoch from the buffer (0 = use all). |
| `--no-subtree-reuse` | off | Use fresh MCTS per move (legacy); default reuses the per-game tree. |
| `--eval-every=N` | 0 | Run candidate-vs-best gate every N iters (0 = disabled). |
| `--eval-games=N` | 4 | Eval games per gate (alternating colors). |
| `--eval-threshold=F` | 0.55 | Min candidate score to be promoted to new best. |
| `--eval-maxply=N` | 60 | Maxply per eval game. |
| `--eval-sims=N` | 16 | MCTS sims/move during eval (no Dirichlet, greedy). |
| `--best=PATH` | `<save>.best` | Path to the best-checkpoint file. |
| `--lr=F` | 1e-3 | Base LR for Adam (warmup-cosine schedule). |
| `--value-weight=F` | 0.5 | Weight of value-MSE relative to policy-CE. |
| `--seed=N` | 42 | RNG seed. |
| `--load=PATH` | – | Load checkpoint before training. |
| `--save=PATH` | – | Save checkpoint after each `--save-every` iters. |
| `--save-every=N` | 1 | Save every N iters (always saves the last iter). |
| `--show-moves` | off | Print every move with MCTS stats. |
| `--show-board` | off | Also print ASCII board after each move (implies `--show-moves`). |

### Notes

- Both sides are played by the model via PUCT MCTS (`ZobristMcts`)
  using its own policy priors and value head as leaf evaluations.
- Move sampling uses visit-count distribution `^ (1/T)` for the first
  `--temp-moves` plies, then switches to greedy (most-visited).
- Value targets are the actual game outcome `z` from the side-to-move's
  POV at that step (±1 for checkmate, 0 for draw / maxply cutoff).
- Policy targets are the sampled move id (cross-entropy). This is a
  hard-target proxy for the full visit distribution and matches the
  existing `Tensor.crossEntropy(List<int>)` API.
- The move vocabulary is built once from the bundled PGN dataset, so the
  policy head can address all common moves from step 0; vocab is **not**
  expanded during self-play.
- A new `ZobristMcts` instance is reused **per game** by default so the
  search tree below the just-played move carries over to the next
  search (subtree reuse). Pass `--no-subtree-reuse` to revert to the
  legacy per-move construction. The table is dropped at game end to
  bound memory.

## MuZero overfit + next-move LM examples

Both examples below auto-resolve the bundled Stockfish binary at
`example/tools/stockfish` (override with `--stockfish=/path/to/stockfish`).
Build it once with the recipe in the previous section.

### 1. Overfit value head on the start position (Stockfish target)

Sanity check that representation -> value head -> autograd -> Adam is wired:

```bash
dart run example/mu_zero/overfit_chess_value_startpos.dart
```

With explicit hyperparameters:

```bash
dart run example/mu_zero/overfit_chess_value_startpos.dart \
  --epochs=300 --lr=0.05 --movetime=300 --tol=0.02 --logEvery=1
```

This now also overfits the **policy head** on Stockfish's `bestmove` for
the start position (joint MSE + cross-entropy loss). Converges in ~10-30
epochs.

Flags: `--epochs --lr --movetime --tol --stockfish`.

### 2. Train next-move policy from PGN dataset

Bundled UCI dataset (default — `lib/loaders/dataset.dart`):

```bash
dart run example/mu_zero/train_next_move_pgn.dart
```

Quick smoke run:

```bash
dart run example/mu_zero/train_next_move_pgn.dart \
  --games=10 --steps=100 --block=24 --embed=32 --layers=1 --heads=4 \
  --logEvery=20 --valEvery=50 --sampleEvery=50 --valSplit=0.3
```

Real PGN file on disk (Lichess elite, TWIC, your own export, etc.):

```bash
dart run example/mu_zero/train_next_move_pgn.dart \
  --pgn=path/to/games.pgn \
  --games=500 --steps=5000 --block=64 --embed=128 --layers=2 --lr=5e-4
```

Outputs:
- per-step train loss + EMA + token-level top-1 accuracy
- periodic token-weighted **val loss + acc** on a held-out split
- periodic **legal-move masked greedy continuation** from `<start>`

Flags: `--pgn --games --steps --block --embed --layers --heads --lr
--valSplit --logEvery --valEvery --sampleEvery --seed`.


## Vision: ViT image classifier

Train a tiny ViT (existing `ViTBackbone` + a Linear classifier head) on
any ImageFolder-style directory:

```
<root>/
  <class_a>/  *.jpg|*.png
  <class_b>/  *.jpg|*.png
  ...
```

Default root is the bundled `Original Images/` celebrity dataset
(31 classes, ~2.5k images).

Quick smoke run (4 classes × 8 images, ~10s):

```bash
dart run example/vision/train_image_classifier.dart \
  --imgSize=32 --patchSize=8 --embed=32 --layers=1 --heads=4 \
  --batch=4 --steps=40 --maxPerClass=8 --maxClasses=4 \
  --logEvery=10 --valEvery=20 --valSplit=0.25
```

Real run on 6 classes (~32s, val acc reaches 2× chance):

```bash
dart run example/vision/train_image_classifier.dart \
  --imgSize=32 --patchSize=8 --embed=64 --layers=2 --heads=4 \
  --batch=8 --steps=300 --maxPerClass=20 --maxClasses=6 --lr=1e-3 \
  --logEvery=25 --valEvery=75 --valSplit=0.25
```

Custom dataset (point at any folder-of-folders):

```bash
dart run example/vision/train_image_classifier.dart \
  --root=/path/to/my_dataset \
  --imgSize=64 --patchSize=8 --embed=128 --layers=3 --batch=16 --steps=2000
```

Outputs:
- per-step train loss + EMA + top-1 accuracy on the mini-batch
- periodic **val loss + val acc** on the held-out split
- final cleanup of all GPU tensors

Flags: `--root --imgSize --patchSize --embed --layers --heads --batch
--steps --lr --maxPerClass --maxClasses --valSplit --logEvery --valEvery
--seed`.

## Vision: face triplet metric learning

Train face-style metric embeddings with `TripletLoss` on the same
ImageFolder layout. Uses `ImageFolderLoader.sampleTriplet()` to pick
(anchor, positive, negative) per step, the existing `ViTBackbone` as
encoder, a `Linear(embed, outDim)` projection (no L2-norm: see note
below), and `TripletLossGPU(margin)`.

> **Engine limitation.** The current CUDA backward becomes unstable when
> too many ViT forward passes accumulate into a single autograd graph.
> Defaults are intentionally tiny so the demo runs to completion.
> Increasing `--steps` or `--triplets` may segfault inside
> `Tensor.backward`. Reduce them if that happens.

Default safe run (1 layer, 1 triplet/step, 5 steps, ~10s):

```bash
dart run example/vision/train_face_triplet.dart
```

With explicit hyperparameters (still conservative):

```bash
dart run example/vision/train_face_triplet.dart \
  --imgSize=32 --patchSize=8 --embed=32 --outDim=32 \
  --layers=1 --heads=4 --triplets=1 --steps=5 \
  --lr=1e-4 --margin=0.2 \
  --maxPerClass=12 --maxClasses=6 \
  --valSplit=0.25 --valPairs=60 --logEvery=1 --valEvery=5
```

Custom dataset:

```bash
dart run example/vision/train_face_triplet.dart \
  --root=/path/to/faces \
  --maxClasses=10 --maxPerClass=20
```

Outputs:
- per-step triplet loss + EMA + count of "active" triplets
  (loss > 0 = margin not yet satisfied)
- periodic **verification accuracy** on random val pairs (mean L2 distance
  for same-class vs different-class pairs, accuracy at the best L2
  threshold)

Flags: `--root --imgSize --patchSize --embed --outDim --layers --heads
--triplets --steps --lr --margin --maxPerClass --maxClasses --valSplit
--valPairs --logEvery --valEvery --seed`.