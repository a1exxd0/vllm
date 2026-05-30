# SPDX-License-Identifier: Apache-2.0
"""Scale-only gradient QAD for Laguna-XS.2 NVFP4-KV — runs on 2x A6000.

Trains ONLY the per-layer NVFP4-KV k/v scales (weights frozen) so the model's
attention is robust to 4-bit KV.  Distills the NVFP4-KV student against the
bf16-KV teacher (same weights, two forwards), minimizing KL on the logits.

Why this fits your hardware: the only trainable tensors are ~20 scalars (k/v per
global layer), there are no weight gradients or optimizer state on the 66 GB
model, and at short context the activation memory is small.  The NVFP4 KV is
injected via a DynamicCache subclass whose scales are learnable (fake-quant is
differentiable w.r.t. the scale), so no model surgery and no Q/forward patching.

Logs every step to a JSONL the dashboard plots (real training curves).

    python -m nvfp4_qad.train_laguna --model ~/models/Laguna-XS.2 \
        --init-scales runs/laguna_kv_scales.json --data data/calib.jsonl \
        --seq-len 1024 --steps 300 --lr 0.05 \
        --out runs/laguna_kv_scales_qad.json --log runs/laguna_qad.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from nvfp4_qad.dashboard import TrainingLogger
from nvfp4_qad.distill import kl_logit_loss
from nvfp4_qad.fake_quant import fake_quant_kv_nvfp4
from nvfp4_qad.laguna_calibrate import iter_token_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--init-scales", default="runs/laguna_kv_scales.json",
                    help="Stage-0 calibration JSON to initialize the scales")
    ap.add_argument("--data", required=True, help="jsonl with {'text': ...}")
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="short context fits 2x A6000; raise if you have room")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05, help="LR for the log-scales")
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--out", default="runs/laguna_kv_scales_qad.json")
    ap.add_argument("--log", default="runs/laguna_qad.jsonl")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import DynamicCache

    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                            fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # freeze: scale-only QAD

    # Learnable log-scales per global layer, initialized from calibration.
    init = json.load(open(args.init_scales, encoding="utf-8"))["per_layer"]
    dev = model.device if hasattr(model, "device") else "cuda"
    tiny = torch.finfo(torch.float32).tiny
    log_k, log_v, q_fixed = {}, {}, {}
    for s, v in init.items():
        i = int(s)
        log_k[i] = torch.nn.Parameter(
            torch.tensor(max(v["k_scale"], tiny), device=dev).log())
        log_v[i] = torch.nn.Parameter(
            torch.tensor(max(v["v_scale"], tiny), device=dev).log())
        q_fixed[i] = float(v["q_scale"])
    params = list(log_k.values()) + list(log_v.values())
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"Training {len(params)} log-scales over {len(log_k)} global layers: {sorted(log_k)}")

    class QADCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            if layer_idx in log_k:
                key_states = fake_quant_kv_nvfp4(
                    key_states, log_k[layer_idx].exp()).to(key_states.dtype)
                value_states = fake_quant_kv_nvfp4(
                    value_states, log_v[layer_idx].exp()).to(value_states.dtype)
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    logger = TrainingLogger(args.log, meta={"model": os.path.basename(args.model),
                                            "stage": "scale-only-QAD"})
    data = iter_token_blocks(args.data, tok, args.seq_len, args.steps, demo=False)

    for step in range(args.steps):
        try:
            batch = next(data)
        except StopIteration:
            data = iter_token_blocks(args.data, tok, args.seq_len, args.steps, demo=False)
            batch = next(data)
        ids = batch["input_ids"].to(model.device)

        with torch.no_grad():
            t_logits = model(input_ids=ids, use_cache=False).logits
        s_logits = model(input_ids=ids, past_key_values=QADCache(), use_cache=True).logits

        V = s_logits.shape[-1]  # per-token-averaged KL
        loss = kl_logit_loss(s_logits.reshape(-1, V), t_logits.reshape(-1, V),
                             temperature=args.kl_temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        scales = {f"model.layers.{i}.self_attn": {
            "k_scale": float(log_k[i].exp()), "v_scale": float(log_v[i].exp()),
            "q_scale": q_fixed[i]} for i in log_k}
        logger.log_step(step, {"loss": float(loss), "kl": float(loss)},
                        scales=scales, lr=args.lr, stage=1)
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  KL={float(loss):.5f}")

    logger.close()

    # Export QAD'd scales in the same format as calibration (k/v learned, q kept).
    per_layer = {}
    for i in sorted(log_k):
        per_layer[i] = {"k_scale": float(log_k[i].exp()), "v_scale": float(log_v[i].exp()),
                        "q_scale": q_fixed[i],
                        "amax_k": float(log_k[i].exp()) * 448,
                        "amax_v": float(log_v[i].exp()) * 448,
                        "amax_q": q_fixed[i] * 448}
    vllm_keys = {}
    for i, v in per_layer.items():
        b = f"model.layers.{i}.self_attn.attn"
        vllm_keys[f"{b}.k_scale"] = v["k_scale"]
        vllm_keys[f"{b}.v_scale"] = v["v_scale"]
        vllm_keys[f"{b}.q_scale"] = v["q_scale"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump({"meta": {"model": os.path.basename(args.model), "method": "scale-only-QAD",
                        "steps": args.steps, "seq_len": args.seq_len},
               "per_layer": per_layer, "vllm_keys": vllm_keys}, open(args.out, "w"), indent=2)
    print(f"\nWrote QAD'd scales -> {args.out}")
    print(f"Plot training:  python -m nvfp4_qad.dashboard {args.log}")
    print(f"Compare scales: python -m nvfp4_qad.plot_calibration --in {args.out}")


if __name__ == "__main__":
    main()
