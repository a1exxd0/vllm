# SPDX-License-Identifier: Apache-2.0
"""Build a calibration corpus (JSONL of {"text": ...}) for NVFP4-KV calibration.

Two modes:
  --from-dir PATH   recursively collect source/text files (robust, offline).
                    Best default: calibrate on code representative of your traffic.
  --from-hf NAME    stream a Hugging Face dataset (e.g. a code dataset).

The calibrator (nvfp4_qad.laguna_calibrate) packs these texts into fixed-length
blocks itself, so each line here can just be one file/document.

Examples:
    # from a local code tree (e.g. the vllm repo, or your own codebase):
    python -m nvfp4_qad.build_calib_data --from-dir ~/vllm/vllm \
        --num-docs 2000 --min-chars 800 --out data/calib.jsonl

    # from a HF code dataset (streaming, no full download):
    python -m nvfp4_qad.build_calib_data \
        --from-hf codeparrot/github-code-clean --text-field code \
        --num-docs 4000 --min-chars 800 --out data/calib.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

# Sensible code/text extensions for --from-dir mode.
DEFAULT_EXTS = (
    ".py", ".pyi", ".ipynb", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".sql", ".md", ".rst", ".txt", ".yaml", ".yml", ".toml",
)


def iter_dir(root, exts, min_chars, max_chars):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip noise
        dirnames[:] = [d for d in dirnames if d not in
                       (".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist")]
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            if len(text) < min_chars:
                continue
            yield text[:max_chars] if max_chars else text


def iter_hf(name, config, split, field, min_chars, max_chars):
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split, streaming=True)
    for ex in ds:
        text = ex.get(field) or ex.get("text") or ex.get("content") or ex.get("code")
        if not text or len(text) < min_chars:
            continue
        yield text[:max_chars] if max_chars else text


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-dir", help="recursively collect files from this directory")
    src.add_argument("--from-hf", help="Hugging Face dataset id to stream")
    ap.add_argument("--config", default=None, help="HF dataset config name")
    ap.add_argument("--split", default="train", help="HF split (default: train)")
    ap.add_argument("--text-field", default="text",
                    help="HF field holding the text (try 'code'/'content' for code datasets)")
    ap.add_argument("--exts", default=None,
                    help="comma-separated extensions for --from-dir (default: common code/text)")
    ap.add_argument("--num-docs", type=int, default=2000, help="max documents to write")
    ap.add_argument("--min-chars", type=int, default=800, help="skip docs shorter than this")
    ap.add_argument("--max-chars", type=int, default=0, help="truncate docs to this (0 = no cap)")
    ap.add_argument("--out", default="data/calib.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    if args.from_dir:
        exts = tuple(e if e.startswith(".") else "." + e
                     for e in args.exts.split(",")) if args.exts else DEFAULT_EXTS
        gen = iter_dir(args.from_dir, exts, args.min_chars, args.max_chars)
        srcdesc = f"dir:{args.from_dir}"
    else:
        gen = iter_hf(args.from_hf, args.config, args.split, args.text_field,
                      args.min_chars, args.max_chars)
        srcdesc = f"hf:{args.from_hf}"

    n, chars = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for text in gen:
            f.write(json.dumps({"text": text}) + "\n")
            n += 1
            chars += len(text)
            if n % 500 == 0:
                print(f"  {n} docs, ~{chars/1e6:.1f}M chars")
            if n >= args.num_docs:
                break

    approx_tok = chars / 4  # rough chars->tokens
    print(f"\nWrote {n} docs ({chars/1e6:.1f}M chars, ~{approx_tok/1e6:.1f}M tokens) "
          f"from {srcdesc} -> {args.out}")
    if n == 0:
        print("  WARNING: 0 docs written — check the path/field/min-chars.")
    else:
        print(f"  At seq-len 4096 that's ~{int(approx_tok/4096)} packed blocks "
              f"(plenty for --batches 64).")
        print(f"Next:\n  python -m nvfp4_qad.laguna_calibrate --model ~/models/Laguna-XS.2 "
              f"--data {args.out} --seq-len 4096 --batches 64 --out runs/laguna_kv_scales.json")


if __name__ == "__main__":
    main()
