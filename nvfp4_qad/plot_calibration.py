# SPDX-License-Identifier: Apache-2.0
"""Plot a Stage-0 calibration result (runs/kv_scales.json) -> PNG.

Real data from your calibration run: per-global-layer k/v/q scales and the
observed activation amax.  Good presentation artifact.

    python -m nvfp4_qad.plot_calibration --in runs/kv_scales.json
    # writes runs/kv_scales.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_layer_value(pl: dict, i: int, key: str):
    """Fetch a per-layer value, handling both int and string JSON keys."""
    return pl[str(i)][key] if str(i) in pl else pl[i][key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/kv_scales.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        d = json.load(f)
    pl = d["per_layer"]
    layers = sorted(int(i) for i in pl)

    ks = [_get_layer_value(pl, i, "k_scale") for i in layers]
    vs = [_get_layer_value(pl, i, "v_scale") for i in layers]
    qs = [_get_layer_value(pl, i, "q_scale") for i in layers]
    ak = [_get_layer_value(pl, i, "amax_k") for i in layers]
    av = [_get_layer_value(pl, i, "amax_v") for i in layers]
    aq = [_get_layer_value(pl, i, "amax_q") for i in layers]
    x = list(range(len(layers)))
    xt = [str(i) for i in layers]
    meta = d.get("meta", {})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))

    a1.plot(x, ks, "-o", label="k_scale", color="#1f77b4")
    a1.plot(x, vs, "-o", label="v_scale", color="#2ca02c")
    a1.plot(x, qs, "-o", label="q_scale", color="#d62728")
    a1.set_yscale("log"); a1.set_xticks(x); a1.set_xticklabels(xt)
    a1.set_xlabel("attention layer index"); a1.set_ylabel("NVFP4-KV scale (= amax/448, log)")
    a1.set_title("Calibrated per-layer NVFP4-KV scales")
    a1.legend(); a1.grid(alpha=0.3, which="both")

    w = 0.27
    a2.bar([i - w for i in x], ak, w, label="amax K", color="#1f77b4")
    a2.bar(x, av, w, label="amax V", color="#2ca02c")
    a2.bar([i + w for i in x], aq, w, label="amax Q", color="#d62728")
    a2.set_yscale("log"); a2.set_xticks(x); a2.set_xticklabels(xt)
    a2.set_xlabel("attention layer index"); a2.set_ylabel("observed |activation| max (log)")
    a2.set_title("Observed Q/K/V activation ranges")
    a2.legend(); a2.grid(alpha=0.3, axis="y", which="both")

    model_name = meta.get("model", "")
    title = "NVFP4-KV Stage-0 calibration"
    if model_name:
        title = f"{model_name} — {title}"
    if meta:
        title += (f"   [{len(layers)} layers, "
                  f"{meta.get('batches', '?')}×{meta.get('seq_len', '?')} tok, "
                  f"q={meta.get('quantile', '?')}]")
    fig.suptitle(title, y=1.03, fontsize=11)

    out = args.out or os.path.splitext(args.inp)[0] + ".png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
