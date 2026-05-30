# SPDX-License-Identifier: Apache-2.0
"""Staged quantization-aware distillation (QAD) loop + scale export.

Framework-agnostic scaffold.  You supply:
  * a frozen BF16 **teacher** (full-precision attention),
  * a **student** that is the same model with its attention score computation
    replaced by :class:`nvfp4_qad.attention.NVFP4FakeQuantScores` (one per layer),
  * thin adapters to pull (logits, per-layer attention probs, hidden states) out
    of each — model-specific; see README for wiring Laguna XS.

Stages (advance only as far as the eval gap requires):
  0. Calibrate amax -> init k/v/q_scale            (nvfp4_qad.calibration)
  1. Scale-only QAD: train k/v/q_scale, weights frozen
  2. Weight QAD: also train LoRA (or full) on q/k/v/o_proj

Loss = forward-KL(student||teacher logits)
     + lambda_attn * attention-map distillation (post-softmax probs)
     + lambda_hidden * hidden-state MSE         (matters once weights unfreeze)

Combine with 1M context extension by running stages on a length curriculum and
re-calibrating scales per length stage (amax drifts with sequence length).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class QADConfig:
    stage: int = 1                       # 1 = scale-only, 2 = +weights
    kl_temperature: float = 1.0
    lambda_attn: float = 0.3             # attention-map distillation weight
    lambda_hidden: float = 0.0          # hidden-state MSE (turn on in stage 2)
    attn_query_subsample: int = 256     # cap query positions in the attn loss at long ctx
    lr_scale: float = 1e-2              # scales are few params -> larger LR ok
    lr_weight: float = 1e-4
    grad_clip: float = 1.0
    length_curriculum: list[int] = field(default_factory=lambda: [8192, 32768, 131072, 1_000_000])


# --------------------------------------------------------------------------- #
# Loss components.
# --------------------------------------------------------------------------- #
def kl_logit_loss(student_logits, teacher_logits, *, temperature=1.0):
    """Forward KL( teacher || student ) on next-token logits (token-averaged)."""
    t = temperature
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    with torch.no_grad():
        t_p = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(s_logp, t_p, reduction="batchmean") * (t * t)


def attention_map_loss(student_probs, teacher_probs, *, subsample=256):
    """MSE between student/teacher post-softmax attention maps, per layer.

    ``*_probs`` are lists of ``[batch, heads, q, k]`` tensors (one per layer).
    Query positions are subsampled at long context to bound cost.
    """
    if not student_probs:
        return student_probs.new_zeros(()) if torch.is_tensor(student_probs) else torch.zeros(())
    total = 0.0
    for sp, tp in zip(student_probs, teacher_probs):
        q = sp.shape[-2]
        if q > subsample:
            idx = torch.randperm(q, device=sp.device)[:subsample]
            sp, tp = sp[..., idx, :], tp[..., idx, :]
        total = total + F.mse_loss(sp, tp.detach())
    return total / len(student_probs)


def hidden_state_loss(student_hidden, teacher_hidden):
    """MSE across matched hidden states (lists of ``[batch, seq, dim]``)."""
    if not student_hidden:
        return torch.zeros((), device=teacher_hidden[0].device if teacher_hidden else "cpu")
    total = 0.0
    for sh, th in zip(student_hidden, teacher_hidden):
        total = total + F.mse_loss(sh, th.detach())
    return total / len(student_hidden)


# --------------------------------------------------------------------------- #
# Optimizer construction per stage.
# --------------------------------------------------------------------------- #
def build_optimizer(student, scale_modules, cfg: QADConfig):
    """Stage-aware optimizer.

    ``scale_modules``: iterable of :class:`NVFP4FakeQuantScores` (the learnable
    k/v/q scales).  In stage 1 only these train; in stage 2 add the (LoRA or full)
    projection weights you have set ``requires_grad=True`` on.
    """
    scale_params = []
    for m in scale_modules:
        m.set_trainable(True)
        scale_params += m.scale_parameters()
    groups = [{"params": scale_params, "lr": cfg.lr_scale}]

    if cfg.stage >= 2:
        scale_param_ids = {id(p) for p in scale_params}
        weight_params = [p for _, p in student.named_parameters()
                         if p.requires_grad and id(p) not in scale_param_ids]
        if weight_params:
            groups.append({"params": weight_params, "lr": cfg.lr_weight})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)


# --------------------------------------------------------------------------- #
# One distillation step.
# --------------------------------------------------------------------------- #
def distill_step(batch, *, teacher_fn, student_fn, scale_modules, optimizer, cfg: QADConfig,
                 logger=None, step: int | None = None, scale_names: dict | None = None):
    """Run one optimization step.

    ``teacher_fn(batch) -> (logits, probs_list, hidden_list)`` under no_grad.
    ``student_fn(batch) -> (logits, probs_list, hidden_list)`` with grad.
    Returns a dict of scalar loss components.

    If ``logger`` (a :class:`nvfp4_qad.dashboard.TrainingLogger`) and ``step`` are
    given, the step is logged.  Pass ``scale_names = {name: module}`` to also log
    per-layer scales for the scale-evolution figure.
    """
    with torch.no_grad():
        t_logits, t_probs, t_hidden = teacher_fn(batch)

    s_logits, s_probs, s_hidden = student_fn(batch)

    l_kl = kl_logit_loss(s_logits, t_logits, temperature=cfg.kl_temperature)
    l_attn = attention_map_loss(s_probs, t_probs, subsample=cfg.attn_query_subsample)
    l_hidden = (hidden_state_loss(s_hidden, t_hidden)
                if cfg.lambda_hidden > 0 else s_logits.new_zeros(()))
    loss = l_kl + cfg.lambda_attn * l_attn + cfg.lambda_hidden * l_hidden

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip:
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g["params"]], cfg.grad_clip
        )
    optimizer.step()
    for m in scale_modules:
        m.clamp_scales()

    stats = {
        "loss": float(loss.detach()),
        "kl": float(l_kl.detach()),
        "attn": float(l_attn.detach()) if torch.is_tensor(l_attn) else float(l_attn),
        "hidden": float(l_hidden.detach()),
    }
    if logger is not None and step is not None:
        scales = ({name: m.export_scales() for name, m in scale_names.items()}
                  if scale_names else None)
        logger.log_step(step, stats, scales=scales, stage=cfg.stage,
                        lr=optimizer.param_groups[0]["lr"])
    return stats


# --------------------------------------------------------------------------- #
# Export the learned scales into a vLLM-loadable form.
# --------------------------------------------------------------------------- #
def export_scales(layer_name_to_scale_module: dict, *, key_template: str = "{layer}.attn") -> dict:
    """Build a ``{checkpoint_key: scalar}`` map for k/v/q scales.

    vLLM's ``BaseKVCacheMethod.process_weights_after_loading`` reads per-layer
    ``k_scale``, ``v_scale``, ``q_scale`` (positive per-tensor scalars; missing or
    <= 0 silently defaults to 1.0 -- so export **all three**).  Confirm the exact
    key names your model's weight loader expects (e.g. ``model.layers.N.self_attn``
    with ``.k_scale`` suffix) and adjust ``key_template``.
    """
    out: dict[str, float] = {}
    for layer, mod in layer_name_to_scale_module.items():
        base = key_template.format(layer=layer)
        for name, val in mod.export_scales().items():
            assert val > 0.0, f"{base}.{name} must be > 0 (got {val}); vLLM would default to 1.0"
            out[f"{base}.{name}"] = val
    return out
