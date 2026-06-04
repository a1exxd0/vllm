# SPDX-License-Identifier: Apache-2.0
"""Test calibrated NVFP4-KV on real text, on Ampere, via fake-quant emulation.

Generates from a prompt twice and prints both side by side:
  * BASELINE  — full bf16 KV cache (what the model normally does).
  * NVFP4-KV  — K/V fake-quantized to NVFP4 using the calibrated scales from
                Stage-0 (what Blackwell would serve).

The NVFP4 path is injected at the KV-cache level (a DynamicCache subclass that
fake-quantizes post-RoPE K and V on write), so it works with any attention
backend and needs no model surgery.  This is the PyTorch emulation validated
against vLLM's reference (0.0 err) -- not bit-exact to the CUDA kernel (that
needs Blackwell), but a faithful quality preview.

Note: this emulates NVFP4 **K/V** only; the served path also quantizes Q to fp8,
which is a smaller effect and not applied here.

    python -m nvfp4_qad.generate_test --model /path/to/model \
        --scales runs/kv_scales.json \
        --prompt "Write a Python function that reverses a linked list." \
        --max-new 200
"""

from __future__ import annotations

import argparse
import json

import torch

from nvfp4_qad.fake_quant import fake_quant_kv_nvfp4


def load_scales(path: str) -> dict[int, tuple[float, float]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    pl = d["per_layer"]
    return {int(i): (float(v["k_scale"]), float(v["v_scale"])) for i, v in pl.items()}


class NVFP4EmuCache:
    """DynamicCache subclass that fake-quantizes K/V for calibrated layers.

    Args:
        scales: ``{layer_idx: (k_scale, v_scale)}`` from a calibration JSON.
    """

    def __new__(cls, scales: dict[int, tuple[float, float]]):
        from transformers import DynamicCache

        class _NVFP4EmuCache(DynamicCache):
            _scales = scales

            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                sc = self._scales.get(layer_idx)
                if sc is not None:
                    k_scale, v_scale = sc
                    key_states = fake_quant_kv_nvfp4(
                        key_states, torch.tensor(k_scale, device=key_states.device)
                    ).to(key_states.dtype)
                    value_states = fake_quant_kv_nvfp4(
                        value_states, torch.tensor(v_scale, device=value_states.device)
                    ).to(value_states.dtype)
                return super().update(key_states, value_states, layer_idx, cache_kwargs)

        return _NVFP4EmuCache()


def generate(model, tok, prompt, max_new, cache=None, chat=True):
    if chat and hasattr(tok, "apply_chat_template") and tok.chat_template:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
        )
        ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
        ids = ids.to(model.device)
    else:
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    kw = {}
    if cache is not None:
        kw["past_key_values"] = cache
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, **kw)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scales", required=True, help="calibration JSON with per_layer k/v scales")
    ap.add_argument("--prompt", default="Write a Python function that reverses a singly linked list.")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--no-chat", action="store_true", help="raw prompt, no chat template")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                            fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    scales = load_scales(args.scales)
    print(f"Loaded NVFP4-KV scales for {len(scales)} layers: {sorted(scales)}\n")

    print("=" * 78)
    print("PROMPT:", args.prompt)
    print("=" * 78)

    base = generate(model, tok, args.prompt, args.max_new, cache=None, chat=not args.no_chat)
    print("\n--- BASELINE (bf16 KV) ---\n" + base)

    nv = generate(model, tok, args.prompt, args.max_new,
                  cache=NVFP4EmuCache(scales), chat=not args.no_chat)
    print("\n--- NVFP4-KV (calibrated, emulated) ---\n" + nv)

    print("\n" + "=" * 78)
    print("identical:" if base == nv else "differ:",
          "outputs match" if base == nv else "outputs diverge (expected to some degree under 4-bit KV)")


if __name__ == "__main__":
    main()
