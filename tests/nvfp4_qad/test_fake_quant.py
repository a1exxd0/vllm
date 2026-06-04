# SPDX-License-Identifier: Apache-2.0
"""pytest tests for nvfp4_qad: fake_quant, calibration, distill, attention."""

import pytest
import torch

from nvfp4_qad.attention import NVFP4FakeQuantScores
from nvfp4_qad.calibration import AmaxAccumulator
from nvfp4_qad.distill import (
    QADConfig,
    attention_map_loss,
    build_optimizer,
    kl_logit_loss,
)
from nvfp4_qad.fake_quant import (
    NVFP4_BLOCK_SIZE,
    fake_quant_kv_nvfp4,
    fake_quant_q_fp8,
    init_kv_scale_from_amax,
    init_q_scale_from_amax,
)
from nvfp4_qad.parity import (
    check_against_reference,
    check_device_execution,
    check_gradients,
)


# --------------------------------------------------------------------------- #
# parity: reference, device, gradients
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("head_size", [16, 64, 128, 256])
def test_reference_parity(head_size):
    err = check_against_reference(head_size=head_size)
    assert err == 0.0, (
        f"head_size={head_size}: fake-quant diverged from CPU reference (err={err})"
    )


def test_gradients_flow():
    check_gradients()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_execution():
    check_device_execution()


# --------------------------------------------------------------------------- #
# fake_quant: scale init
# --------------------------------------------------------------------------- #

def test_scale_from_amax_positive():
    for amax in [0.1, 1.0, 100.0, 1e-10]:
        ks = init_kv_scale_from_amax(amax)
        qs = init_q_scale_from_amax(amax)
        assert ks.item() > 0, f"kv scale non-positive for amax={amax}"
        assert qs.item() > 0, f"q scale non-positive for amax={amax}"


def test_scale_from_zero_amax():
    # Zero amax should clamp to float32.tiny, not zero.
    ks = init_kv_scale_from_amax(0.0)
    assert ks.item() > 0


def test_fake_quant_kv_shape():
    x = torch.randn(4, 8, 16, 128)
    ks = init_kv_scale_from_amax(x.abs().amax())
    out = fake_quant_kv_nvfp4(x, ks)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_fake_quant_kv_head_size_divisibility():
    with pytest.raises(AssertionError, match="divisible"):
        fake_quant_kv_nvfp4(torch.randn(4, 15), init_kv_scale_from_amax(1.0))


def test_fake_quant_q_shape():
    q = torch.randn(2, 4, 32, 64)
    qs = init_q_scale_from_amax(q.abs().amax())
    out = fake_quant_q_fp8(q, qs)
    assert out.shape == q.shape


# --------------------------------------------------------------------------- #
# AmaxAccumulator
# --------------------------------------------------------------------------- #

def test_amax_true_max():
    acc = AmaxAccumulator(quantile=None)
    acc.update(torch.tensor([1.0, 2.0, -5.0]))
    acc.update(torch.tensor([3.0]))
    assert acc.compute().item() == pytest.approx(5.0)


def test_amax_quantile():
    acc = AmaxAccumulator(quantile=0.5)
    data = torch.arange(1, 101, dtype=torch.float32)
    acc.update(data)
    # Median of 1..100 is ~50; allow tolerance due to reservoir sampling.
    val = acc.compute().item()
    assert 40 <= val <= 60, f"unexpected median estimate {val}"


def test_amax_all_zeros():
    acc = AmaxAccumulator(quantile=None)
    acc.update(torch.zeros(64))
    assert acc.compute().item() == pytest.approx(0.0)


def test_amax_no_data_raises():
    acc = AmaxAccumulator(quantile=None)
    with pytest.raises(AssertionError, match="no data"):
        acc.compute()


def test_amax_quantile_no_data_raises():
    acc = AmaxAccumulator(quantile=0.9)
    with pytest.raises(AssertionError, match="no data"):
        acc.compute()


# --------------------------------------------------------------------------- #
# distill: loss functions
# --------------------------------------------------------------------------- #

def test_kl_logit_loss_zero_for_same():
    logits = torch.randn(8, 32)
    loss = kl_logit_loss(logits, logits)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_attention_map_loss_empty_list():
    loss = attention_map_loss([], [])
    assert loss.item() == pytest.approx(0.0)


def test_attention_map_loss_subsample():
    # With subsample < seq, loss should still be a scalar.
    sp = [torch.rand(2, 4, 512, 512) for _ in range(3)]
    tp = [torch.rand(2, 4, 512, 512) for _ in range(3)]
    loss = attention_map_loss(sp, tp, subsample=64)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_attention_map_loss_zip_truncation():
    # zip() truncates to the shorter list — ensure no crash and non-negative.
    sp = [torch.rand(1, 2, 8, 8)]
    tp = [torch.rand(1, 2, 8, 8), torch.rand(1, 2, 8, 8)]
    loss = attention_map_loss(sp, tp)
    assert loss.item() >= 0


# --------------------------------------------------------------------------- #
# distill: build_optimizer
# --------------------------------------------------------------------------- #

def test_build_optimizer_stage1_only_scales():
    model = torch.nn.Linear(8, 8)
    for p in model.parameters():
        p.requires_grad_(False)
    scores = NVFP4FakeQuantScores(softmax_scale=0.125)
    cfg = QADConfig(stage=1)
    opt = build_optimizer(model, [scores], cfg)
    assert len(opt.param_groups) == 1
    assert len(opt.param_groups[0]["params"]) == 3  # log_k, log_v, log_q


def test_build_optimizer_stage2_includes_weights():
    model = torch.nn.Linear(8, 8)
    for p in model.parameters():
        p.requires_grad_(True)
    scores = NVFP4FakeQuantScores(softmax_scale=0.125)
    cfg = QADConfig(stage=2)
    opt = build_optimizer(model, [scores], cfg)
    assert len(opt.param_groups) == 2


# --------------------------------------------------------------------------- #
# NVFP4FakeQuantScores: export_scales invariant
# --------------------------------------------------------------------------- #

def test_export_scales_positive():
    scores = NVFP4FakeQuantScores(k_scale_init=0.01, v_scale_init=0.02, q_scale_init=0.005)
    d = scores.export_scales()
    for name, val in d.items():
        assert val > 0.0, f"{name}={val} not positive"


def test_export_scales_assert_on_zero():
    scores = NVFP4FakeQuantScores(k_scale_init=1.0)
    # Force log_k_scale to -inf so exp() returns 0.
    with torch.no_grad():
        scores.log_k_scale.fill_(-1e38)
    with pytest.raises(AssertionError, match="k_scale"):
        scores.export_scales()
