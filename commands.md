nvcc --shared -o native/lib/libmat_mul.so native/src/engine.cu -Xcompiler -fPIC

nvcc --shared -o native/lib/dart_cuda.so native/src/dart_cuda.cu -Xcompiler -fPIC



nvcc --shared -o native/lib/libmat_mul.so native/src/engine_v2.cu -Xcompiler -fPIC


## MuZero chess

Train MuZero vs Stockfish (fresh run, save checkpoint each iter):

```bash
dart run example/tool/muzero_vs_stockfish_train.dart example/tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=40 \
  --sf-movetime=100 --sf-skill=0 \
  --save=muzero_chess.bin --save-every=1
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
  --epochs=300 --lr=0.05 --movetime=300 --tol=0.02
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