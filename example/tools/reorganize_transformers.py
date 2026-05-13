#!/usr/bin/env python3
"""Phase 4: subdivide lib/core/transformers/ by architecture family."""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/c/Users/kkinyama/dart_cuda")
LIB = ROOT / "lib"

MOVES = [
    # AFT building blocks (pure attention transformer stacks)
    ("lib/core/transformers/aft_transformer_encoder.dart",        "lib/core/transformers/aft/transformer_encoder.dart"),
    ("lib/core/transformers/aft_transformer_encoder_block.dart",  "lib/core/transformers/aft/transformer_encoder_block.dart"),
    ("lib/core/transformers/aft_transformer_decoder.dart",        "lib/core/transformers/aft/transformer_decoder.dart"),
    ("lib/core/transformers/aft_transformer_decoder_block.dart",  "lib/core/transformers/aft/transformer_decoder_block.dart"),
    ("lib/core/transformers/aft_text_decoder_block.dart",         "lib/core/transformers/aft/text_decoder_block.dart"),
    ("lib/core/transformers/aft_muzero_transformer_decoder.dart", "lib/core/transformers/aft/muzero_transformer_decoder.dart"),

    # DeepSeek MoE decoder family (was misfiled under models/mu_zero)
    ("lib/core/models/mu_zero/deepseek_aft_decoder.dart",         "lib/core/transformers/deepseek/deepseek_aft_decoder.dart"),

    # Vision transformers
    ("lib/core/transformers/aft_vit_backbone.dart",               "lib/core/transformers/vision/vit_backbone.dart"),
    ("lib/core/transformers/aft_vit_face_embeding.dart",          "lib/core/transformers/vision/vit_face_embedding.dart"),
    ("lib/core/models/vit_object_detector.dart",                  "lib/core/transformers/vision/vit_object_detector.dart"),

    # Modality-specific wrappers (text / audio / video / multi-modal)
    ("lib/core/transformers/text_decoder.dart",                       "lib/core/transformers/modalities/text_decoder.dart"),
    ("lib/core/transformers/text_transformer.dart",                   "lib/core/transformers/modalities/text_transformer.dart"),
    ("lib/core/transformers/audio_transformer.dart",                  "lib/core/transformers/modalities/audio_transformer.dart"),
    ("lib/core/transformers/video_transformer.dart",                  "lib/core/transformers/modalities/video_transformer.dart"),
    ("lib/core/transformers/multi_modal_transformer.dart",            "lib/core/transformers/modalities/multi_modal_transformer.dart"),
    ("lib/core/transformers/multi_modal_transformer2.dart",           "lib/core/transformers/modalities/multi_modal_transformer2.dart"),
    ("lib/core/transformers/multi_modal_trnasformer_encoder.dart",    "lib/core/transformers/modalities/multi_modal_transformer_encoder.dart"),
]


def run(cmd, check=True):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)
    if r.stdout.strip(): print(r.stdout)
    if r.stderr.strip(): print(r.stderr, file=sys.stderr)
    if check and r.returncode != 0:
        raise SystemExit(f"Failed: {cmd}")


def git_mv(old, new):
    (ROOT / new).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "ls-files", "--error-unmatch", old],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        run(f"git mv '{old}' '{new}'")
    else:
        run(f"mv '{old}' '{new}'")


def collect_dart_files():
    out = []
    for base in [ROOT / "lib", ROOT / "example", ROOT / "tool", ROOT / "test"]:
        if base.exists():
            out.extend(base.rglob("*.dart"))
    return out


def pkg(rel):
    assert rel.startswith("lib/")
    return rel[4:]


def main():
    pkg_map = {pkg(o): pkg(n) for o, n in MOVES}

    for old, new in MOVES:
        if not (ROOT / old).exists():
            print(f"skip (missing): {old}")
            continue
        git_mv(old, new)

    files = collect_dart_files()
    changed = 0
    for f in files:
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        orig = text
        for old_pkg, new_pkg in pkg_map.items():
            text = re.sub(
                r"(['\"])package:dart_cuda/" + re.escape(old_pkg) + r"(['\"])",
                lambda m, np=new_pkg: f"{m.group(1)}package:dart_cuda/{np}{m.group(2)}",
                text,
            )
        if text != orig:
            f.write_text(text)
            changed += 1
    print(f"Rewrote imports in {changed} file(s).")


if __name__ == "__main__":
    main()
