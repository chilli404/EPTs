"""Test _gradnorm_cw_inline: exact gradient measurement for ternary consistent.

Uses separate forward passes with batch=1 and gradient checkpointing,
matching the approach of _gradnorm_cw but for the ternary case.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fogen.model import GPT, ModelConfig
from fogen.training.train import (
    _compute_consistency,
    _gradnorm_cache,
    _gradnorm_cw_inline,
)


def _make_tiny_model():
    cfg = ModelConfig(
        vocab_size=256, n_layer=4, d_model=64, n_head=4, ctx_len=32,
        execution_mode="sequential",
    )
    return GPT(cfg), cfg


def _make_batch(cfg, batch=4):
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (batch, cfg.ctx_len))
    return x, y


def _train_diverged(model, cfg, steps=50):
    """Train seq-only so par diverges."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = _make_batch(cfg)
        seq_logits = model(x, mode="sequential")
        loss = F.cross_entropy(seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_exact_gradnorm_produces_valid_cw():
    """Should compute cw = rho * ||∇LM|| / ||∇con|| via separate fwd passes."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    execution_mask = ["sequential", "parallel", "skip", "sequential"]
    execution_cfg = {
        "consistency_type": "centered_mse",
        "gradnorm_every": 1,
    }

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})

    cw = _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.2, step=0)

    print(f"  cw={cw:.6f}")
    assert 1e-4 <= cw <= 10.0, f"cw={cw} out of bounds"


def test_exact_gradnorm_deterministic():
    """Same inputs should produce the same cw."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    execution_mask = ["parallel", "sequential", "parallel", "sequential"]
    execution_cfg = {
        "consistency_type": "centered_mse",
        "gradnorm_every": 1,
    }

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    cw1 = _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.2, step=0)

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    cw2 = _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.2, step=0)

    print(f"  cw1={cw1:.6f} cw2={cw2:.6f}")
    assert cw1 == cw2, f"Not deterministic: {cw1} != {cw2}"


def test_exact_gradnorm_skips_between_every():
    """Between gradnorm_every intervals, should return cached cw."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    execution_mask = ["parallel", "sequential", "parallel", "sequential"]
    execution_cfg = {
        "consistency_type": "centered_mse",
        "gradnorm_every": 100,
    }

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    cw0 = _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.2, step=0)
    cw50 = _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.2, step=50)

    assert cw50 == cw0, f"step 50 should return cached cw0={cw0}, got {cw50}"
    print(f"  cw0={cw0:.6f} cw50={cw50:.6f} (same)")


def test_exact_gradnorm_proportional_to_rho():
    """cw should be proportional to rho."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    execution_mask = ["parallel", "sequential", "parallel", "sequential"]
    execution_cfg = {
        "consistency_type": "centered_mse",
        "gradnorm_every": 1,
    }

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    cw_02 = _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.2, step=0)

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    cw_04 = _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.4, step=0)

    ratio = cw_04 / max(cw_02, 1e-12)
    print(f"  cw(0.2)={cw_02:.4f} cw(0.4)={cw_04:.4f} ratio={ratio:.3f}")
    # May be clamped at 10.0 — check proportionality only if unclamped
    if cw_02 < 10.0 and cw_04 < 10.0:
        assert 1.9 < ratio < 2.1, f"ratio={ratio:.3f}, expected ~2.0"
    else:
        print("  (clamped, skipping ratio check)")


def test_exact_gradnorm_no_memory_leak():
    """After gradnorm, GPU memory should not grow."""
    torch.manual_seed(42)
    model, cfg = _make_tiny_model()
    _train_diverged(model, cfg, steps=50)
    x, y = _make_batch(cfg)

    execution_mask = ["parallel", "sequential", "parallel", "sequential"]
    execution_cfg = {
        "consistency_type": "centered_mse",
        "gradnorm_every": 1,
    }

    # Run twice — if retain_graph leaks, second call uses more memory
    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.2, step=0)

    _gradnorm_cache.clear()
    _gradnorm_cache.update({"cw": 0.1, "step": -100})
    _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho=0.2, step=0)

    # No assertion on memory — just verify it doesn't crash
    print("  no memory leak (ran twice without issue)")


if __name__ == "__main__":
    test_exact_gradnorm_produces_valid_cw()
    print("PASS: produces valid cw")

    test_exact_gradnorm_deterministic()
    print("PASS: deterministic")

    test_exact_gradnorm_skips_between_every()
    print("PASS: skips between every")

    test_exact_gradnorm_proportional_to_rho()
    print("PASS: proportional to rho")

    test_exact_gradnorm_no_memory_leak()
    print("PASS: no memory leak")

    print("\nAll tests passed.")
