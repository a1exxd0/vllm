# SPDX-License-Identifier: Apache-2.0
"""Abstract base class and entry point for NVFP4 attention QAD.

Subclass :class:`QADTrainer` and implement :meth:`build_adapters` to wire the
toolkit to your model's attention mechanism.  See ``README.md`` for the expected
adapter contract.

Usage (after implementing build_adapters in your subclass):
    # 0. verify the codec matches the CUDA kernel first (no training):
    python -m nvfp4_qad.parity

    # 1. then train:
    python -m nvfp4_qad.train \
        --model /path/to/model --data /path/to/longctx.jsonl \
        --stage 1 --steps 2000 --seq-len 8192 \
        --out runs/nvfp4_qad --log runs/nvfp4_qad.jsonl

    # 2. watch it live (separate shell, re-run anytime):
    python -m nvfp4_qad.dashboard runs/nvfp4_qad.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod


class QADTrainer(ABC):
    """Abstract base for model-specific NVFP4 QAD wiring.

    Subclass this and implement :meth:`build_adapters` for your architecture.
    """

    @abstractmethod
    def build_adapters(self, model, head_dim: int) -> dict:
        """Return a dict with the following keys:

            named_attn_modules : {name: attention_module}      (for calibration hooks)
            get_qkv(module, inputs, output) -> (q, k, v)       (post-RoPE Q/K, V; [B,H,S,D])
            install_student(scale_mods: {name: NVFP4FakeQuantScores})
                -> monkeypatch each attention to route scores through its scale module
            teacher_fn(batch) -> (logits, probs_list, hidden_list)   # BF16, no grad
            student_fn(batch) -> (logits, probs_list, hidden_list)   # fake-quant attn

        Q/K must be taken AFTER RoPE (vLLM caches post-RoPE K).
        ``softmax_scale`` should be ``head_dim**-0.5``.
        """


def load_jsonl(path, tokenizer, seq_len, batch_size):
    """Minimal long-context loader: pack text into seq_len blocks."""
    import torch
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            txt = obj.get("text", "")
            buf += tokenizer(txt).input_ids
            while len(buf) >= seq_len * batch_size:
                ids = torch.tensor(buf[: seq_len * batch_size]).view(batch_size, seq_len)
                buf = buf[seq_len * batch_size:]
                yield {"input_ids": ids.cuda()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--calib-batches", type=int, default=64)
    ap.add_argument("--lambda-attn", type=float, default=0.3)
    ap.add_argument("--out", default="runs/qad")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nvfp4_qad.attention import NVFP4FakeQuantScores
    from nvfp4_qad.calibration import attach_calibration_hooks, finalize_scales
    from nvfp4_qad.dashboard import TrainingLogger
    from nvfp4_qad.distill import QADConfig, build_optimizer, distill_step, export_scales

    os.makedirs(args.out, exist_ok=True)
    log = TrainingLogger(args.log or os.path.join(args.out, "train.jsonl"),
                         meta={"model": os.path.basename(args.model), "stage": args.stage})

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # Instantiate the model-specific trainer subclass.  Users must implement
    # build_adapters() — see class QADTrainer above and README for guidance.
    trainer = _get_trainer_impl()
    ad = trainer.build_adapters(model, head_dim)

    # --- Stage 0: calibrate amax -> initial scales -------------------------- #
    print("[stage0] calibrating amax(Q/K/V)...")
    accs, handles = attach_calibration_hooks(
        ad["named_attn_modules"], get_qkv=ad["get_qkv"], quantile=0.9999
    )
    scales0 = finalize_scales(
        run_calibration_loop(model, load_jsonl(args.data, tok, args.seq_len, args.batch_size),
                             (accs, handles), max_batches=args.calib_batches)
    )

    scale_mods = {
        name: NVFP4FakeQuantScores(
            k_scale_init=s.k_scale, v_scale_init=s.v_scale, q_scale_init=s.q_scale,
            softmax_scale=head_dim ** -0.5,
        ).cuda()
        for name, s in scales0.items()
    }
    ad["install_student"](scale_mods)

    # --- Stage 1/2: QAD ----------------------------------------------------- #
    cfg = QADConfig(stage=args.stage, lambda_attn=args.lambda_attn,
                    lambda_hidden=(0.1 if args.stage >= 2 else 0.0))
    opt = build_optimizer(model, scale_mods.values(), cfg)
    print(f"[stage{args.stage}] QAD for {args.steps} steps...")
    data = load_jsonl(args.data, tok, args.seq_len, args.batch_size)
    for step in range(args.steps):
        batch = next(data)
        stats = distill_step(batch, teacher_fn=ad["teacher_fn"], student_fn=ad["student_fn"],
                             scale_modules=list(scale_mods.values()), optimizer=opt, cfg=cfg,
                             logger=log, step=step, scale_names=scale_mods)
        if step % 50 == 0:
            print(f"  step {step:5d}  loss={stats['loss']:.4f}  kl={stats['kl']:.4f}  "
                  f"attn={stats['attn']:.4f}")

    # --- Export ------------------------------------------------------------- #
    out_scales = export_scales(scale_mods)
    with open(os.path.join(args.out, "kv_scales.json"), "w") as f:
        json.dump(out_scales, f, indent=2)
    log.close()
    print(f"[done] exported {len(out_scales)} scale entries to {args.out}/kv_scales.json")
    print("Serve:  vllm serve <checkpoint-with-these-scales> --kv-cache-dtype nvfp4")


def run_calibration_loop(model, data_iter, accs_handles_setup, *, max_batches):
    """Run calibration data through the model and return raw accumulators.

    Hook removal is handled by :func:`nvfp4_qad.calibration.run_calibration`.
    """
    from nvfp4_qad.calibration import run_calibration
    return run_calibration(model, data_iter, accs_handles_setup, max_batches=max_batches)


def _get_trainer_impl() -> QADTrainer:
    """Return the QADTrainer subclass to use.

    Override this in your entry point, or subclass QADTrainer and call
    main() after setting this to return your instance.
    """
    raise NotImplementedError(
        "No QADTrainer implementation found.  Subclass QADTrainer, implement "
        "build_adapters(), and call main() from your entry point.  "
        "See README for an example."
    )


if __name__ == "__main__":
    main()
