# SPDX-License-Identifier: Apache-2.0
"""Method-explainer figures for the NVFP4 attention QAD toolkit.

IMPORTANT: these are **not training results**.  They are exact mathematical
properties of the NVFP4 codec (:mod:`nvfp4_qad.fake_quant`) computed on synthetic
tensors -- use them to *explain how the method works*, not to claim accuracy.
For real "it's training / it's working" curves, log your actual run with
:mod:`nvfp4_qad.dashboard` and plot that.

Run:  python -m nvfp4_qad.figures        # writes PNGs to nvfp4_qad/figures/
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nvfp4_qad.fake_quant import (
    FP8_E4M3_MAX,
    NVFP4_BLOCK_SIZE,
    fake_quant_kv_nvfp4,
    init_kv_scale_from_amax,
)

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)

E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ACCENT, GOOD, BAD = "#1f77b4", "#2ca02c", "#d62728"
NOTE = "method explainer — exact codec property on synthetic data, not a training result"


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def fig_quantizer():
    """Exact NVFP4 (E2M1) quantizer transfer function for one 16-element block."""
    head = NVFP4_BLOCK_SIZE
    A = 6.0
    xs = torch.linspace(-A, A, 1201)
    ys = []
    for x in xs:
        block = torch.zeros(1, head)
        block[0, 0] = A          # fix vec_max = A so the block scale is constant
        block[0, 1] = x
        ks = init_kv_scale_from_amax(torch.tensor(A))
        ys.append(fake_quant_kv_nvfp4(block, ks)[0, 1].item())
    ys = np.array(ys)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[2, 1.1])
    ax1.plot(xs.numpy(), xs.numpy(), "--", color="gray", lw=1, label="identity (bf16)")
    ax1.plot(xs.numpy(), ys, color=ACCENT, lw=2, label="NVFP4 dequantized")
    ax1.set_title("NVFP4 quantizer transfer function (one 16-elem block)")
    ax1.set_xlabel("input value"); ax1.set_ylabel("dequantized value")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

    for v in E2M1_GRID:
        ax2.scatter([v, -v], [0, 0], color=ACCENT, zorder=3)
        ax2.annotate(f"{v:g}", (v, 0), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8)
    ax2.set_title("E2M1 magnitudes\n{0,.5,1,1.5,2,3,4,6} × sign × scale")
    ax2.set_yticks([]); ax2.set_xlabel("representable magnitudes"); ax2.grid(alpha=0.3, axis="x")
    fig.suptitle("4-bit NVFP4 (E2M1): non-uniform grid, denser near zero", y=1.04, fontsize=12)
    fig.text(0.5, -0.04, NOTE, ha="center", fontsize=8, style="italic", color="gray")
    return _save(fig, "fig1_nvfp4_quantizer.png")


def fig_scale_sweep():
    """Reconstruction error & fp8 block-scale health vs the per-tensor scale.

    Uses outlier-heavy K (a few spiking channels) -- the regime where the
    per-tensor scale actually drives fp8 block-scale saturation / underflow.
    """
    N, D = 4096, 128
    K = torch.randn(N, D)
    K = K + (torch.rand(N, D) > 0.99).float() * 25.0 * torch.sign(torch.randn(N, D))
    amax = K.abs().amax()
    base = init_kv_scale_from_amax(amax)

    mults = np.logspace(-3, 3, 60)
    mse, frac_sat, frac_uf = [], [], []
    for m in mults:
        ks = base * float(m)
        dq = fake_quant_kv_nvfp4(K, ks)
        mse.append(torch.mean((dq - K) ** 2).item())
        gs = 1.0 / ks
        vmax = K.reshape(N, D // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE).abs().amax(-1)
        bs = (gs * vmax / 6.0).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn).float()
        nz = vmax > 0
        frac_sat.append(((bs >= FP8_E4M3_MAX) & nz).float().mean().item())
        frac_uf.append(((bs == 0) & nz).float().mean().item())
    mse = np.array(mse)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.loglog(mults, mse, color=ACCENT, lw=2, label="reconstruction MSE")
    ax.axvline(1.0, color="black", ls="--", lw=1)
    ax.scatter([1.0], [mse[np.argmin(np.abs(mults - 1.0))]], color=GOOD, zorder=5,
               label="calibrated  k_scale = amax/448")
    ax.set_xlabel("k_scale  (× amax/448)"); ax.set_ylabel("MSE(dequant K, K)")
    ax.set_title("Calibration matters at the tails: reconstruction error vs scale\n"
                 "(synthetic outlier-heavy K, head_size=128)")
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    ax2.semilogx(mults, np.array(frac_sat) * 100, color=BAD, lw=1.5, ls=":",
                 label="% blocks fp8-saturated")
    ax2.semilogx(mults, np.array(frac_uf) * 100, color="#ff7f0e", lw=1.5, ls=":",
                 label="% blocks underflow→0")
    ax2.set_ylabel("% fp8 block scales clamped"); ax2.set_ylim(0, 100)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper center", fontsize=8)
    fig.text(0.5, -0.04, NOTE, ha="center", fontsize=8, style="italic", color="gray")
    return _save(fig, "fig2_quant_error_vs_scale.png")


def main():
    print("Generating NVFP4 method-explainer figures ->", OUT)
    print("(NOT training results — see nvfp4_qad.dashboard for live training curves)")
    fig_quantizer()
    fig_scale_sweep()
    print("Done.")


if __name__ == "__main__":
    main()
