# SPDX-License-Identifier: Apache-2.0
"""Live training dashboard for NVFP4 QAD — logs your REAL run, then plots it.

This is the honest "it's working as I train it" artifact: your training loop
appends one JSONL row per step / eval, and :func:`plot_dashboard` renders
presentation figures from that real data.  Nothing here fabricates results.

Wiring (in your training loop):

    from nvfp4_qad.dashboard import TrainingLogger
    logger = TrainingLogger("runs/nvfp4_qad.jsonl", meta={"model": "my-model", "stage": 1})
    ...
    stats = distill_step(batch, ..., cfg=cfg)          # returns {loss, kl, attn, hidden}
    logger.log_step(step, stats,
                    scales={name: m.export_scales() for name, m in scale_mods.items()},
                    lr=opt.param_groups[0]["lr"])
    ...
    logger.log_eval(step, {"ruler_acc": 0.81}, context_len=131072)   # from your evals

Then, anytime (e.g. live during training):

    python -m nvfp4_qad.dashboard runs/nvfp4_qad.jsonl

writes training_loss.png / scale_evolution.png / eval_vs_step.png /
eval_vs_context.png next to the log.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingLogger:
    """Append-only JSONL logger for QAD training + eval events."""

    def __init__(self, path: str, meta: dict | None = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        self._f = open(path, "a", encoding="utf-8")
        if meta:
            self._write({"type": "meta", **meta})

    def _write(self, row: dict):
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def log_step(self, step: int, losses: dict, *, scales: dict | None = None,
                 lr: float | None = None, stage: int | None = None,
                 context_len: int | None = None):
        """One optimization step.  ``losses`` e.g. {loss, kl, attn, hidden}.

        ``scales`` is ``{layer_name: {"k_scale":..,"v_scale":..,"q_scale":..}}``.
        """
        row = {"type": "step", "step": step, **{k: float(v) for k, v in losses.items()}}
        if lr is not None:
            row["lr"] = float(lr)
        if stage is not None:
            row["stage"] = int(stage)
        if context_len is not None:
            row["context_len"] = int(context_len)
        if scales is not None:
            row["scales"] = scales
        self._write(row)

    def log_eval(self, step: int, metrics: dict, *, context_len: int | None = None):
        """An eval point (e.g. RULER accuracy, perplexity) at a given context length."""
        row = {"type": "eval", "step": step, **{k: float(v) for k, v in metrics.items()}}
        if context_len is not None:
            row["context_len"] = int(context_len)
        self._write(row)

    def close(self):
        self._f.close()


# --------------------------------------------------------------------------- #
# Plotting.
# --------------------------------------------------------------------------- #
def _read(path: str):
    steps, evals, meta = [], [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("type")
            if t == "step":
                steps.append(r)
            elif t == "eval":
                evals.append(r)
            elif t == "meta":
                meta = r
    return steps, evals, meta


def _title_suffix(meta, watermark):
    bits = [meta[k] for k in ("model", "stage") if k in meta]
    s = "  ·  ".join(str(b) for b in bits)
    return (f"  [{s}]" if s else "") + ("   (SAMPLE — placeholder data)" if watermark else "")


def plot_dashboard(path: str, out_dir: str | None = None, *, watermark: bool = False):
    """Render presentation figures from a JSONL training log.  Plots only what
    is present (loss components, per-layer scales, evals)."""
    steps, evals, meta = _read(path)
    out_dir = out_dir or os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def finish(fig, name):
        if watermark:
            fig.text(0.5, 0.5, "SAMPLE", fontsize=80, color="gray", alpha=0.12,
                     ha="center", va="center", rotation=30)
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p)
        print(f"  wrote {p}")

    # --- 1. Loss components vs step ---------------------------------------- #
    if steps:
        xs = [r["step"] for r in steps]
        comps = [k for k in ("loss", "kl", "attn", "hidden") if any(k in r for r in steps)]
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        for k in comps:
            ys = [r.get(k, float("nan")) for r in steps]
            ax.plot(xs, ys, lw=2, label=k)
        ax.set_yscale("log")
        ax.set_xlabel("step"); ax.set_ylabel("loss (log)")
        ax.set_title("QAD training loss" + _title_suffix(meta, watermark))
        ax.legend(); ax.grid(alpha=0.3, which="both")
        # shade stage transitions if logged
        stages = [(r["step"], r["stage"]) for r in steps if "stage" in r]
        if stages:
            last = stages[0][1]
            for s_step, s in stages:
                if s != last:
                    ax.axvline(s_step, color="gray", ls=":", lw=1)
                    last = s
        finish(fig, "training_loss.png")

    # --- 2. Per-layer scale evolution -------------------------------------- #
    scale_rows = [r for r in steps if "scales" in r]
    if scale_rows:
        # series[layer][which] = list of (step, value)
        series = defaultdict(lambda: defaultdict(list))
        for r in scale_rows:
            for layer, d in r["scales"].items():
                for which, val in d.items():
                    series[layer][which].append((r["step"], val))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
        for ax, which in zip(axes, ("k_scale", "v_scale", "q_scale")):
            for layer, d in series.items():
                if which in d:
                    xy = d[which]
                    ax.plot([p[0] for p in xy], [p[1] for p in xy], lw=1.2, alpha=0.8,
                            label=layer if len(series) <= 8 else None)
            ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title(which)
            ax.grid(alpha=0.3, which="both")
        if len(series) <= 8:
            axes[-1].legend(fontsize=7, loc="best")
        fig.suptitle("Per-layer learned scales over training" + _title_suffix(meta, watermark),
                     y=1.03)
        finish(fig, "scale_evolution.png")

    # --- 3. Eval vs step --------------------------------------------------- #
    if evals:
        metric_keys = sorted({k for r in evals for k in r
                              if k not in ("type", "step", "context_len")})
        # eval vs step (per metric), grouping by context_len if present
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        for mk in metric_keys:
            by_ctx = defaultdict(list)
            for r in evals:
                if mk in r:
                    by_ctx[r.get("context_len")].append((r["step"], r[mk]))
            for ctx, xy in sorted(by_ctx.items(), key=lambda kv: (kv[0] is None, kv[0])):
                xy.sort()
                lbl = mk + (f" @{ctx//1024}k" if ctx else "")
                ax.plot([p[0] for p in xy], [p[1] for p in xy], "-o", lw=2, label=lbl)
        ax.set_xlabel("step"); ax.set_ylabel("eval metric")
        ax.set_title("Eval over training" + _title_suffix(meta, watermark))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        finish(fig, "eval_vs_step.png")

        # eval vs context length (latest step per context)
        have_ctx = [r for r in evals if r.get("context_len")]
        if have_ctx:
            fig, ax = plt.subplots(figsize=(8.4, 4.6))
            for mk in metric_keys:
                latest = {}
                for r in have_ctx:
                    if mk in r:
                        c = r["context_len"]
                        if c not in latest or r["step"] >= latest[c][0]:
                            latest[c] = (r["step"], r[mk])
                if latest:
                    cs = sorted(latest)
                    ax.plot(cs, [latest[c][1] for c in cs], "-o", lw=2, label=mk)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("context length (tokens)"); ax.set_ylabel("eval metric")
            ax.set_title("Eval vs context length (latest)" + _title_suffix(meta, watermark))
            ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
            finish(fig, "eval_vs_context.png")

    if not written:
        print("  (no plottable rows yet — log some steps/evals first)")
    return written


def _write_sample(path: str):
    """Write a clearly-labeled SAMPLE log so you can preview the dashboard layout.

    This is synthetic placeholder data for *format preview only* — every figure it
    produces is watermarked SAMPLE.  Delete and replace with your real run.
    """
    import math
    import random

    random.seed(0)
    logger = TrainingLogger(path, meta={"model": "SAMPLE-placeholder", "stage": 1})
    layers = [f"model.layers.{i}.self_attn" for i in range(4)]
    k0 = {ly: 0.5 * (i + 1) for i, ly in enumerate(layers)}  # start off-target
    for step in range(0, 600, 10):
        prog = step / 600
        stage = 1 if step < 300 else 2
        base = 3.0 * math.exp(-3 * prog) + 0.15
        losses = {"loss": base + random.uniform(-0.03, 0.03),
                  "kl": 0.7 * base, "attn": 0.25 * base,
                  "hidden": (0.1 * base if stage == 2 else 0.0)}
        scales = {}
        for ly in layers:
            tgt = 0.012
            cur = tgt + (k0[ly] - tgt) * math.exp(-4 * prog)
            scales[ly] = {"k_scale": cur, "v_scale": cur * 1.1, "q_scale": cur * 0.9}
        logger.log_step(step, losses, scales=scales, lr=1e-2, stage=stage)
        if step % 100 == 0:
            for ctx in (8192, 32768, 131072):
                acc = 0.55 + 0.4 * prog - 0.06 * math.log2(ctx / 8192) / 4
                logger.log_eval(step, {"ruler_acc": max(0, min(1, acc))}, context_len=ctx)
    logger.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--sample":
        plot_dashboard(sys.argv[1], watermark=("--sample" in sys.argv))
    else:
        # No log given: write + render a watermarked SAMPLE so you see the layout.
        sample = os.path.join(os.path.dirname(__file__), "figures", "_sample_run.jsonl")
        print("No log path given — writing a SAMPLE preview (synthetic, watermarked).")
        _write_sample(sample)
        plot_dashboard(sample, watermark=True)
        print("These are SAMPLE layout previews. Point the script at your real "
              "JSONL log to get real figures:  python -m nvfp4_qad.dashboard <log>.jsonl")
