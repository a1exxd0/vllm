# SPDX-License-Identifier: Apache-2.0
"""Stage-0 NVFP4-KV calibration for Laguna-XS.2 — runnable on 2x A6000.

Collects per-layer amax(Q/K/V) on the **global-attention** layers via forward
hooks (no surgery on poolside's custom model), turns them into per-tensor
``k/v/q`` scales (``amax/448``, the fork's convention), and writes a JSON you can
fold into the served checkpoint for ``--kv-cache-dtype nvfp4``.

Why hooks (not the KV-cache or a forward rewrite):
  * Forward hooks on ``k_norm`` / ``v_proj`` / ``q_norm`` are robust across
    transformers/cache versions and need zero changes to modeling_laguna.py.
  * K amax is captured **pre-RoPE** (k_norm output). RoPE is a per-pair rotation
    over only half the head dims (partial_rotary_factor=0.5), so per-tensor amax
    shifts negligibly — fine for scale *init*. The exact post-RoPE behavior is
    handled by the fake-quant during gradient QAD (a later, bigger-hardware step).

This is forward-only and runs under no_grad → fits a 66 GB bf16 model sharded
across 2x A6000 with device_map="auto".

Usage:
    python -m nvfp4_qad.laguna_calibrate \
        --model ~/models/Laguna-XS.2 --data longctx.jsonl \
        --seq-len 4096 --batches 32 --out runs/laguna_kv_scales.json
    # quick pipeline smoke (no data file):
    python -m nvfp4_qad.laguna_calibrate --model ~/models/Laguna-XS.2 --demo-data --batches 2
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from nvfp4_qad.calibration import AmaxAccumulator
from nvfp4_qad.fake_quant import init_kv_scale_from_amax, init_q_scale_from_amax

DEMO_TEXTS = [
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "import torch\nclass MLP(torch.nn.Module):\n    def __init__(self, d):\n        super().__init__()\n        self.fc = torch.nn.Linear(d, d)\n    def forward(self, x):\n        return self.fc(x).relu()\n",
    "The quick brown fox jumps over the lazy dog. " * 64,
    "async def fetch(url):\n    async with aiohttp.ClientSession() as s:\n        async with s.get(url) as r:\n            return await r.json()\n",
]


def iter_token_blocks(data_path, tokenizer, seq_len, batches, demo):
    """Yield {input_ids:[1,seq_len]} blocks from a jsonl (or demo texts)."""
    buf: list[int] = []
    produced = 0

    def source():
        if demo or not data_path:
            i = 0
            while True:
                yield DEMO_TEXTS[i % len(DEMO_TEXTS)]
                i += 1
        else:
            while True:  # loop the file if we need more blocks than it holds
                with open(data_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield json.loads(line).get("text", "")

    for txt in source():
        buf += tokenizer(txt).input_ids
        while len(buf) >= seq_len:
            ids = torch.tensor(buf[:seq_len], dtype=torch.long).unsqueeze(0)
            buf = buf[seq_len:]
            yield {"input_ids": ids}
            produced += 1
            if produced >= batches:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=None, help="jsonl with {'text': ...}")
    ap.add_argument("--demo-data", action="store_true",
                    help="use a few built-in texts (pipeline smoke test only)")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--quantile", type=float, default=0.9999,
                    help="amax quantile (robust to outliers); use 1.0 for true max")
    ap.add_argument("--all-layers", action="store_true",
                    help="calibrate every layer, not just global-attention ones")
    ap.add_argument("--out", default="runs/laguna_kv_scales.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    cfg = model.config

    layer_types = list(getattr(cfg, "layer_types", []))
    if args.all_layers or not layer_types:
        targets = list(range(cfg.num_hidden_layers))
        kind = "all"
    else:
        targets = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        kind = "global-attention"
    print(f"Calibrating {len(targets)} {kind} layers: {targets}")

    # Locate the decoder layers (LagunaForCausalLM -> .model.layers).
    layers = model.model.layers
    q = (None if args.quantile >= 1.0 else args.quantile)
    accs = {i: {"k": AmaxAccumulator(q), "v": AmaxAccumulator(q), "q": AmaxAccumulator(q)}
            for i in targets}
    handles = []

    def mk_hook(idx, which):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            accs[idx][which].update(t)
        return hook

    for i in targets:
        attn = layers[i].self_attn
        handles.append(attn.k_norm.register_forward_hook(mk_hook(i, "k")))  # pre-RoPE K
        handles.append(attn.q_norm.register_forward_hook(mk_hook(i, "q")))
        handles.append(attn.v_proj.register_forward_hook(mk_hook(i, "v")))

    n = 0
    with torch.no_grad():
        for batch in iter_token_blocks(args.data, tok, args.seq_len, args.batches, args.demo_data):
            ids = batch["input_ids"].to(model.device)
            model(input_ids=ids, use_cache=False)
            n += 1
            if n % 4 == 0 or n == 1:
                print(f"  calibrated on {n}/{args.batches} blocks of {args.seq_len} tokens")
    for h in handles:
        h.remove()

    # amax -> scales (k/v: amax/448 ; q: amax/448) and a vLLM-keyed export.
    per_layer, vllm_keys = {}, {}
    for i in targets:
        ka = accs[i]["k"].compute(); va = accs[i]["v"].compute(); qa = accs[i]["q"].compute()
        ks = float(init_kv_scale_from_amax(ka)); vs = float(init_kv_scale_from_amax(va))
        qs = float(init_q_scale_from_amax(qa))
        per_layer[i] = {"k_scale": ks, "v_scale": vs, "q_scale": qs,
                        "amax_k": float(ka), "amax_v": float(va), "amax_q": float(qa)}
        base = f"model.layers.{i}.self_attn.attn"  # confirm against vLLM's Laguna loader
        vllm_keys[f"{base}.k_scale"] = ks
        vllm_keys[f"{base}.v_scale"] = vs
        vllm_keys[f"{base}.q_scale"] = qs

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": {"model": os.path.basename(args.model), "layers": targets,
                            "seq_len": args.seq_len, "batches": n, "quantile": args.quantile},
                   "per_layer": per_layer, "vllm_keys": vllm_keys}, f, indent=2)
    print(f"\nWrote {len(targets)} layers of NVFP4-KV scales -> {args.out}")
    ex = next(iter(per_layer.values()))
    print(f"  example layer {targets[0]}: k_scale={ex['k_scale']:.5f} "
          f"v_scale={ex['v_scale']:.5f} q_scale={ex['q_scale']:.5f} "
          f"(amax_k={ex['amax_k']:.2f})")
    print("Next: fold vllm_keys into the served checkpoint and serve with "
          "--kv-cache-dtype nvfp4 on Blackwell; compare RULER vs the FP8-KV baseline.")


if __name__ == "__main__":
    main()
