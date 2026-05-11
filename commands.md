nvcc --shared -o libmat_mul.so engine.cu -Xcompiler -fPIC

nvcc --shared -o dart_cuda.so dart_cuda.cu -Xcompiler -fPIC



nvcc --shared -o libmat_mul.so engine_v2.cu -Xcompiler -fPIC


## MuZero chess

Train MuZero vs Stockfish (fresh run, save checkpoint each iter):

```bash
dart run tool/muzero_vs_stockfish_train.dart tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=40 \
  --sf-movetime=100 --sf-skill=0 \
  --save=muzero_chess.bin --save-every=1
```

Resume training from checkpoint:

```bash
dart run tool/muzero_vs_stockfish_train.dart tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=40 \
  --sf-movetime=100 --sf-skill=0 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1
```

Play a UCI match between the MuZero engine and Stockfish:

```bash
dart run tool/uci_match.dart \
  "dart run lib/mu_zero/muzero_chess_uci.dart --no-train" \
  tools/stockfish --movetime=300 --maxply=80
```

Run the MuZero engine standalone (UCI on stdio, e.g. for a chess GUI):

```bash
dart run lib/mu_zero/muzero_chess_uci.dart            # trains then enters UCI loop
dart run lib/mu_zero/muzero_chess_uci.dart --no-train  # quick handshake test
```

Self-play demo (train then play one greedy game):

```bash
dart run lib/mu_zero/muzero_chess_player.dart
```

Build Stockfish (Linux) from bundled source:

```bash
cd tools/sf/stockfish/src && make -j$(nproc) build ARCH=x86-64-avx2
cp stockfish ../../../stockfish && chmod +x ../../../stockfish
```
dart run tool/muzero_vs_stockfish_train.dart tools/stockfish \
  --iters=3 --games=2 --epochs=2 --maxply=80 \
  --sf-movetime=300 --sf-skill=20 \
  --load=muzero_chess.bin --save=muzero_chess.bin --save-every=1