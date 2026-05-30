# nvfp4_qad — Quantization-Aware Distillation for NVFP4 attention

Training-side toolkit to fine-tune a BF16 model (e.g. **Laguna XS**) so its
attention stays accurate at long (1M) context when served by this vLLM fork with
`--kv-cache-dtype nvfp4`.

## Why

This fork already serves attention as **FP8 query × NVFP4 KV** on the FlashInfer
trtllm-gen kernel (Blackwell / SM100). It folds per-layer `q_scale`/`k_scale`/
`v_scale` into the matmuls, but those default to **1.0** and the model was trained
in BF16 — nothing taught it to tolerate fp4 attention, which hurts most at long
context. QAD closes that gap: fine-tune with the fp4/fp8 attention **simulated in
the forward pass**, distilling from the BF16 teacher, and **learn/calibrate the
per-layer scales**. No new vLLM kernels are needed — you ship a checkpoint.

## What's here

| File | Role |
|---|---|
| `fake_quant.py` | Differentiable NVFP4 (K/V) + FP8 (Q) fake-quant. Forward **bit-matches** vLLM's `ref_nvfp4_quant`; STE on the fp8 block-scale and E2M1 rounding so gradients reach activations **and** the learnable scale. |
| `calibration.py` | Stage-0: hook the model, collect per-layer `amax(Q/K/V)` (post-RoPE K), init `k/v/q_scale`. |
| `attention.py` | `NVFP4FakeQuantScores` — drop-in attention score module with log-space learnable scales; optional softmax-map output for distillation. |
| `distill.py` | Staged QAD loop: KL(logits) + attention-map MSE + hidden MSE; per-stage optimizer; `export_scales`. |
| `parity.py` | The hard gate: fake-quant forward == vLLM reference (any HW) and == the real CUDA kernel (SM100). |

## The exact scheme it mirrors (do not drift)

Per 16-element block along `head_size` (`cvt_warp_fp16_to_fp4` in
`csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh`):

- E2M1 grid `{0, .5, 1, 1.5, 2, 3, 4, 6}` ±sign; block scale = `fp8_e4m3((1/k_scale)·blockAbsMax/6)`, round-tripped through fp8.
- Checkpoint convention (matches the fork's `tests/kernels/attention/test_cache.py`): `k_scale = v_scale = amax/448`, `q_scale = amax/448`; the kernel uses `global_scale = 1/k_scale`. `k_scale` cancels nominally — it only sets how the fp8 block scale lands (saturate vs underflow), which is the QAD lever.
- **K is quantized post-RoPE** (vLLM caches post-RoPE K). Quantize Q→fp8, K,V→nvfp4, then run normal BF16 attention on the dequantized tensors — **do not** re-multiply by the served `bmm1_scale`.

## Recipe

```
# 0. Calibrate (fake-quant OFF) -> initial scales
from nvfp4_qad.calibration import attach_calibration_hooks, run_calibration
accs = attach_calibration_hooks(named_attn_modules, get_qkv=my_get_qkv, quantile=0.9999)
scales0 = run_calibration(model, calib_loader, accs, max_batches=64)

# 1. Scale-only QAD (weights frozen) — often recovers most of the gap
from nvfp4_qad.attention import NVFP4FakeQuantScores
from nvfp4_qad.distill import QADConfig, build_optimizer, distill_step, export_scales
scale_mods = { name: NVFP4FakeQuantScores(
                   k_scale_init=s.k_scale, v_scale_init=s.v_scale, q_scale_init=s.q_scale,
                   softmax_scale=head_dim**-0.5)
               for name, s in scales0.items() }
# ... wire each scale_mods[name] into the student's attention (replace the score compute) ...
cfg = QADConfig(stage=1, lambda_attn=0.3)
opt = build_optimizer(student, scale_mods.values(), cfg)
for batch in loader:
    distill_step(batch, teacher_fn=teacher_fn, student_fn=student_fn,
                 scale_modules=list(scale_mods.values()), optimizer=opt, cfg=cfg)

# 2. (only if RULER gap remains) Stage-2: unfreeze LoRA on q/k/v/o_proj, set cfg.stage=2,
#    cfg.lambda_hidden>0, rebuild the optimizer.

# Export for vLLM
weights_extra = export_scales(scale_mods)   # {checkpoint_key: positive scalar}
```

**1M context**: run stages on a length curriculum (`QADConfig.length_curriculum`)
and **re-calibrate scales per length stage** — `amax(K/V)` drifts with length and
can push block scales into fp8 saturation. Apply your RoPE scaling (YaRN/NTK) and
make sure the exported HF `rope_scaling` + `max_position_embeddings` match what
vLLM re-derives at serve time.

### You must wire (model-specific)
- `get_qkv(module, inputs, output) -> (q, k, v)` returning **post-RoPE** Q/K, V.
- `teacher_fn(batch) -> (logits, probs_list, hidden_list)` (BF16, no grad).
- `student_fn(batch) -> (logits, probs_list, hidden_list)` (attention via `NVFP4FakeQuantScores`).
- Confirm the checkpoint key names your loader expects for `*.k_scale/v_scale/q_scale`
  (adjust `export_scales(key_template=...)`). **Missing or ≤0 → vLLM silently uses 1.0.**

## Verify

```
python -m nvfp4_qad.parity     # reference + gradient checks anywhere; CUDA-kernel parity on SM100
```

- **Reference parity** (any hardware): fake-quant forward == vLLM `ref_nvfp4_quant_dequant`. Locally measured max error: **0.0 (bf16)**, ~2e-7 (fp32 reciprocal noise).
- **CUDA-kernel parity** (SM100): fake-quant == real `reshape_and_cache_flash(..., "nvfp4", k_scale, v_scale)` dequantized — the train/inference-skew gate. Run on your Blackwell GPU.
- **Checkpoint-load test**: after exporting, load in vLLM and assert `layer._k/_v/_q_scale_float` equal what you exported.

## Serve

```
vllm serve <qad-checkpoint> --kv-cache-dtype nvfp4   # Blackwell SM100; FlashInfer/trtllm-gen
# do NOT pass --attention-config.disable_flashinfer_q_quantization (keeps Q in fp8)
```

## Figures & live dashboard

Two distinct things — keep them separate when presenting:

- **Method-explainer figures** (`python -m nvfp4_qad.figures` → `figures/fig1_*`, `fig2_*`):
  exact properties of the NVFP4 codec on synthetic data. They explain *how the
  quantizer behaves* (the E2M1 staircase; why scale calibration avoids the fp8
  saturation cliff). They are **not** training results and are watermarked as such.

- **Live training dashboard** (`nvfp4_qad/dashboard.py`): the honest "it's working
  as I train it" artifact. Your training loop appends one JSONL row per step/eval;
  the plotter renders real `training_loss.png`, `scale_evolution.png`,
  `eval_vs_step.png`, `eval_vs_context.png` from *your* run. Wire it via:

  ```python
  from nvfp4_qad.dashboard import TrainingLogger
  logger = TrainingLogger("runs/laguna_qad.jsonl", meta={"model": "laguna-xs", "stage": 1})
  distill_step(batch, ..., cfg=cfg, logger=logger, step=step,
               scale_names=scale_mods)          # logs losses + per-layer scales
  logger.log_eval(step, {"ruler_acc": acc}, context_len=131072)
  ```

  Then, anytime during training:  `python -m nvfp4_qad.dashboard runs/laguna_qad.jsonl`

  Run `python -m nvfp4_qad.dashboard` with no args to render a **watermarked SAMPLE**
  preview of the layout (synthetic placeholder data — not a result).

## End-to-end evals
Run **through vLLM** with `--kv-cache-dtype nvfp4`: RULER (4k→1M), needle-in-haystack
to 1M, long-doc perplexity, long-repo code completion — compare BF16 teacher /
naive-nvfp4 / nvfp4-QAD. Target: QAD closes most of the teacher-vs-naive gap at ≥128k.

## Notes / scope
- `csrc` is the source of truth; `parity.py` pins the fake-quant to it. If you bump the vLLM commit, re-run parity.
- **Q stays fp8, not fp4** — the served path uses fp8 Q deliberately; fp4 Q adds error on the most sensitive matmul (QK^T) with no KV-memory benefit. The fake-quant keeps the fp4 path available for future kernels.
