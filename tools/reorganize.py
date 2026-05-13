#!/usr/bin/env python3
"""Reorganize dart_cuda library into loaders/core/examples.

Usage: python3 tools/reorganize.py <phase>
   phase: 1 (loaders), 2 (core), 3 (examples), all
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/c/Users/kkinyama/dart_cuda")
LIB = ROOT / "lib"

# (old_rel_to_root, new_rel_to_root) - all paths relative to repo root
PHASES = {
    1: [  # loaders
        ("lib/dataset/dataset.dart",       "lib/loaders/dataset.dart"),
        ("lib/dataset/chess.dart",         "lib/loaders/chess.dart"),
        ("lib/apps/images.dart",           "lib/loaders/images.dart"),
        ("lib/apps/triplet_loader.dart",   "lib/loaders/triplet_loader.dart"),
        ("lib/apps/triplet_loader2.dart",  "lib/loaders/triplet_loader2.dart"),
    ],
    2: [  # core
        # tensor
        ("lib/gpu_tensor.dart",            "lib/core/tensor/gpu_tensor.dart"),
        ("lib/core/engine.dart",           "lib/core/tensor/engine.dart"),
        ("lib/core/matrix.dart",           "lib/core/tensor/matrix.dart"),
        ("lib/core/tensor.dart",           "lib/core/tensor/tensor.dart"),
        ("lib/core/mat_mul.dart",          "lib/core/tensor/mat_mul.dart"),
        # optimizers
        ("lib/adam.dart",                                  "lib/core/optimizers/adam.dart"),
        ("lib/optimizers/cross_entropy.dart",              "lib/core/optimizers/cross_entropy.dart"),
        ("lib/optimizers/stochastic_grad_desc.dart",       "lib/core/optimizers/stochastic_grad_desc.dart"),
        # layers
        ("lib/nn.dart",            "lib/core/layers/nn.dart"),
        ("lib/layer_norm.dart",    "lib/core/layers/layer_norm.dart"),
        ("lib/feed_forward.dart",  "lib/core/layers/feed_forward.dart"),
        ("lib/mlp.dart",           "lib/core/layers/mlp.dart"),
        ("lib/mlp2.dart",          "lib/core/layers/mlp2.dart"),
        ("lib/mlp3.dart",          "lib/core/layers/mlp3.dart"),
        ("lib/nn/conv_2d.dart",    "lib/core/layers/conv_2d.dart"),
        # attention
        ("lib/aft.dart",                              "lib/core/attention/aft.dart"),
        ("lib/aft_cross_attention.dart",              "lib/core/attention/aft_cross_attention.dart"),
        ("lib/aft_multi_head_attention.dart",         "lib/core/attention/aft_multi_head_attention.dart"),
        ("lib/aft_multi_head_cross_attention.dart",   "lib/core/attention/aft_multi_head_cross_attention.dart"),
        # transformers
        ("lib/aft_transformer_encoder.dart",          "lib/core/transformers/aft_transformer_encoder.dart"),
        ("lib/aft_transformer_encoder_block.dart",    "lib/core/transformers/aft_transformer_encoder_block.dart"),
        ("lib/aft_transformer_decoder.dart",          "lib/core/transformers/aft_transformer_decoder.dart"),
        ("lib/aft_transformer_decoder_block.dart",    "lib/core/transformers/aft_transformer_decoder_block.dart"),
        ("lib/aft_text_decoder_block.dart",           "lib/core/transformers/aft_text_decoder_block.dart"),
        ("lib/aft_muzero_transformer_decoder.dart",   "lib/core/transformers/aft_muzero_transformer_decoder.dart"),
        ("lib/text_decoder.dart",                     "lib/core/transformers/text_decoder.dart"),
        ("lib/text_transformer.dart",                 "lib/core/transformers/text_transformer.dart"),
        ("lib/audio_transformer.dart",                "lib/core/transformers/audio_transformer.dart"),
        ("lib/video_transformer.dart",                "lib/core/transformers/video_transformer.dart"),
        ("lib/multi_modal_transformer.dart",          "lib/core/transformers/multi_modal_transformer.dart"),
        ("lib/multi_modal_transformer2.dart",         "lib/core/transformers/multi_modal_transformer2.dart"),
        ("lib/multi_modal_trnasformer_encoder.dart",  "lib/core/transformers/multi_modal_trnasformer_encoder.dart"),
        ("lib/aft_vit_backbone.dart",                 "lib/core/transformers/aft_vit_backbone.dart"),
        ("lib/aft_vit_face_embeding.dart",            "lib/core/transformers/aft_vit_face_embeding.dart"),
        # models
        ("lib/vit_object_detector.dart",      "lib/core/models/vit_object_detector.dart"),
        ("lib/mu_zero/deepseek_aft_decoder.dart",     "lib/core/models/mu_zero/deepseek_aft_decoder.dart"),
        ("lib/mu_zero/mu_zero_greedy_agent2.dart",    "lib/core/models/mu_zero/mu_zero_greedy_agent2.dart"),
        ("lib/mu_zero/muzero_chess_mcts.dart",        "lib/core/models/mu_zero/muzero_chess_mcts.dart"),
        ("lib/mu_zero/muzero_chess_player.dart",      "lib/core/models/mu_zero/muzero_chess_player.dart"),
        ("lib/mu_zero/muzero_greedy_agent.dart",      "lib/core/models/mu_zero/muzero_greedy_agent.dart"),
        ("lib/mu_zero/training.dart",                 "lib/core/models/mu_zero/training.dart"),
        ("lib/chess/mcts.dart",                       "lib/core/models/chess/mcts.dart"),
        ("lib/chess/uci.dart",                        "lib/core/models/chess/uci.dart"),
        # utils
        ("lib/network_utils.dart",       "lib/core/utils/network_utils.dart"),
        ("lib/persistence.dart",         "lib/core/utils/persistence.dart"),
        ("lib/hungarian_algorithm.dart", "lib/core/utils/hungarian_algorithm.dart"),
        ("lib/triplet_loss.dart",        "lib/core/utils/triplet_loss.dart"),
        ("lib/open_cv/open_cv.dart",     "lib/core/utils/open_cv.dart"),
    ],
    3: [  # examples
        ("lib/example_audio_video.dart",  "example/audio_video.dart"),
        ("lib/train_xor.dart",            "example/train_xor.dart"),
        ("lib/train_xor_2.dart",          "example/train_xor_2.dart"),
        ("lib/train_xor_3.dart",          "example/train_xor_3.dart"),
        ("lib/overfit.dart",              "example/overfit.dart"),
        ("lib/main_face_gpu.dart",        "example/main_face_gpu.dart"),
        ("lib/mlp_learn.dart",            "example/mlp_learn.dart"),
        ("lib/mu_zero/example.dart",                          "example/mu_zero/example.dart"),
        ("lib/mu_zero/example2.dart",                         "example/mu_zero/example2.dart"),
        ("lib/mu_zero/example3.dart",                         "example/mu_zero/example3.dart"),
        ("lib/mu_zero/example_deepseek_aft_training.dart",    "example/mu_zero/deepseek_aft_training.dart"),
        ("lib/mu_zero/example_deepseek_shakespeare.dart",     "example/mu_zero/deepseek_shakespeare.dart"),
        ("lib/mu_zero/example_deepseek_muzero_shakespeare.dart","example/mu_zero/deepseek_muzero_shakespeare.dart"),
        ("lib/mu_zero/shakespear_example.dart",               "example/mu_zero/shakespear.dart"),
        ("lib/linformer/chess_gpt_example.dart",              "example/chess_gpt.dart"),
        ("lib/apps/face_embeddings.dart",                     "example/face_embeddings.dart"),
        ("lib/apps/face_training.dart",                        "example/face_training.dart"),
        # bin/ -> example/bin/
        ("bin/chess_gpu_train.dart",            "example/bin/chess_gpu_train.dart"),
        ("bin/chess_gpu_train2.dart",           "example/bin/chess_gpu_train2.dart"),
        ("bin/dart_cuda.dart",                  "example/bin/dart_cuda.dart"),
        ("bin/example.dart",                    "example/bin/example.dart"),
        ("bin/example2.dart",                   "example/bin/example2.dart"),
        ("bin/example_bipartide_matching.dart", "example/bin/example_bipartide_matching.dart"),
        ("bin/example_bipartide_matching2.dart","example/bin/example_bipartide_matching2.dart"),
        ("bin/example_bipartide_matching3.dart","example/bin/example_bipartide_matching3.dart"),
        ("bin/example_multi_modal.dart",        "example/bin/example_multi_modal.dart"),
        ("bin/example_object_detection.dart",   "example/bin/example_object_detection.dart"),
        ("bin/main.dart",                       "example/bin/main.dart"),
        ("bin/shakespear.dart",                 "example/bin/shakespear.dart"),
        ("bin/shakespear2.dart",                "example/bin/shakespear2.dart"),
        ("bin/shakespear3.dart",                "example/bin/shakespear3.dart"),
        ("bin/shakespear4.dart",                "example/bin/shakespear4.dart"),
        ("bin/test_run_multi_modal.dart",       "example/bin/test_run_multi_modal.dart"),
        ("bin/test_run_multi_modal_full.dart",  "example/bin/test_run_multi_modal_full.dart"),
    ],
}


def run(cmd, check=True, cwd=None):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd or ROOT, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout)
    if r.stderr.strip():
        print(r.stderr, file=sys.stderr)
    if check and r.returncode != 0:
        raise SystemExit(f"Failed: {cmd}")
    return r


def git_mv(old, new):
    new_dir = (ROOT / new).parent
    new_dir.mkdir(parents=True, exist_ok=True)
    # Use git mv if file is tracked, else plain mv
    r = subprocess.run(["git", "ls-files", "--error-unmatch", old],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        run(f"git mv '{old}' '{new}'")
    else:
        run(f"mv '{old}' '{new}'")


def build_package_path(rel):
    """lib/foo/bar.dart -> foo/bar.dart"""
    assert rel.startswith("lib/")
    return rel[4:]


def collect_all_dart_files():
    out = []
    for base in [ROOT / "lib", ROOT / "bin", ROOT / "tool", ROOT / "test", ROOT / "example"]:
        if not base.exists():
            continue
        for p in base.rglob("*.dart"):
            out.append(p)
    return out


def rewrite_imports(mapping):
    """mapping: dict old_rel -> new_rel (for lib/* files only).
    Rewrites:
      package:dart_cuda/<old_pkg_path> -> package:dart_cuda/<new_pkg_path>
    Also rewrites relative imports inside lib/ to use package: form so they
    survive directory moves cleanly.
    """
    # Build pkg path mapping (only lib/ entries produce package: imports)
    pkg_map = {}
    for old, new in mapping.items():
        if old.startswith("lib/") and new.startswith("lib/"):
            pkg_map[build_package_path(old)] = build_package_path(new)

    files = collect_all_dart_files()
    changed = 0
    for f in files:
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        orig = text
        # 1. Direct package: rewrites
        for old_pkg, new_pkg in pkg_map.items():
            text = re.sub(
                r"(['\"])package:dart_cuda/" + re.escape(old_pkg) + r"(['\"])",
                lambda m, np=new_pkg: f"{m.group(1)}package:dart_cuda/{np}{m.group(2)}",
                text,
            )
        if text != orig:
            f.write_text(text)
            changed += 1
    print(f"Rewrote package: imports in {changed} file(s).")


def normalize_relative_to_package(files):
    """Convert all relative imports in lib/ .dart files to package: imports.
    This is a one-time migration that lets later moves only need to update
    package: paths.
    """
    changed = 0
    for f in files:
        if "/lib/" not in str(f).replace(os.sep, "/"):
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        orig = text
        # Find import 'foo.dart' or import '../foo.dart' (relative, no scheme)
        # and rewrite to package:dart_cuda/<resolved>
        def repl(m):
            quote = m.group(1)
            target = m.group(2)
            if target.startswith("dart:") or target.startswith("package:"):
                return m.group(0)
            # Resolve relative to file's directory, must end up under lib/
            base = f.parent
            resolved = (base / target).resolve()
            try:
                rel = resolved.relative_to(LIB)
            except ValueError:
                return m.group(0)
            return f"import {quote}package:dart_cuda/{rel.as_posix()}{quote}"

        text = re.sub(
            r"import\s+(['\"])([^'\"]+\.dart)\1",
            repl,
            text,
        )
        # Same for export
        def repl_export(m):
            quote = m.group(1)
            target = m.group(2)
            if target.startswith("dart:") or target.startswith("package:"):
                return m.group(0)
            base = f.parent
            resolved = (base / target).resolve()
            try:
                rel = resolved.relative_to(LIB)
            except ValueError:
                return m.group(0)
            return f"export {quote}package:dart_cuda/{rel.as_posix()}{quote}"

        text = re.sub(
            r"export\s+(['\"])([^'\"]+\.dart)\1",
            repl_export,
            text,
        )
        if text != orig:
            f.write_text(text)
            changed += 1
    print(f"Normalized relative imports to package: in {changed} file(s).")


def do_phase(n):
    print(f"\n=== Phase {n} ===")
    moves = PHASES[n]
    # First normalize all relative imports in lib/ to package: form so we
    # only need to rewrite package: paths.
    files = collect_all_dart_files()
    normalize_relative_to_package(files)

    mapping = dict(moves)
    # Perform moves
    for old, new in moves:
        if not (ROOT / old).exists():
            print(f"skip (missing): {old}")
            continue
        git_mv(old, new)

    # Rewrite imports
    rewrite_imports(mapping)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    arg = sys.argv[1]
    if arg == "all":
        for n in (1, 2, 3):
            do_phase(n)
    else:
        do_phase(int(arg))


if __name__ == "__main__":
    main()
