"""Gradient checkpointing must be an identity transform on gradients.

The symmetric (non-memory_efficient) consistency branch needs checkpointing to
fit 3B/7B without switching to the memory_efficient path, which hardcodes
teacher_detach=True and therefore changes the objective. These tests pin the
invariant that enabling checkpointing changes memory, not math.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from fogen.model import GPT, ModelConfig


def _make_tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=256, n_layer=4, d_model=64, n_head=4, ctx_len=32,
        execution_mode="sequential",
    )
    return GPT(cfg), cfg


def _grads_for(model, x, y, mode, gradient_checkpointing):
    model.zero_grad(set_to_none=True)
    logits = model(x, mode=mode, gradient_checkpointing=gradient_checkpointing)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    return loss.detach().clone(), {
        n: p.grad.detach().clone() for n, p in model.named_parameters()
        if p.grad is not None
    }


def test_checkpointing_preserves_gradients_sequential():
    model, cfg = _make_tiny_model()
    model.train()
    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))

    loss_off, grads_off = _grads_for(model, x, y, "sequential", False)
    loss_on, grads_on = _grads_for(model, x, y, "sequential", True)

    torch.testing.assert_close(loss_off, loss_on, rtol=1e-5, atol=1e-6)
    assert set(grads_off) == set(grads_on)
    for name in grads_off:
        torch.testing.assert_close(
            grads_off[name], grads_on[name], rtol=1e-4, atol=1e-6,
            msg=lambda m, n=name: f"gradient mismatch for {n}: {m}")


def test_checkpointing_preserves_gradients_under_execution_mask():
    """The mask path is what actually runs at 3B/7B."""
    model, cfg = _make_tiny_model(seed=3)
    model.train()
    torch.manual_seed(2)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    mask = ["sequential", "parallel", "skip", "parallel"]

    loss_off, grads_off = _grads_for(model, x, y, mask, False)
    loss_on, grads_on = _grads_for(model, x, y, mask, True)

    torch.testing.assert_close(loss_off, loss_on, rtol=1e-5, atol=1e-6)
    for name in grads_off:
        torch.testing.assert_close(
            grads_off[name], grads_on[name], rtol=1e-4, atol=1e-6,
            msg=lambda m, n=name: f"gradient mismatch for {n}: {m}")


def test_polymorphic_loss_accepts_and_uses_gradient_checkpointing():
    """The binary/poly branch needs it too, or removing memory_efficient OOMs.

    Regression guard for job 320: gradient_checkpointing was added to the ternary
    and 6-mode symmetric branches only, so a poly config setting the flag had it
    silently ignored and ran with no memory relief at all.
    """
    import inspect

    from fogen.training.train import polymorphic_loss

    sig = inspect.signature(polymorphic_loss)
    assert "gradient_checkpointing" in sig.parameters, (
        "polymorphic_loss must accept gradient_checkpointing")

    src = inspect.getsource(polymorphic_loss)
    # both forward calls in the symmetric path must receive it
    assert src.count("gradient_checkpointing=gradient_checkpointing") >= 2, (
        "both sequential and parallel forwards must pass the flag through")

    # and it must be honoured numerically
    model, cfg = _make_tiny_model(seed=5)
    model.train()
    torch.manual_seed(4)
    x = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))
    y = torch.randint(0, cfg.vocab_size, (2, cfg.ctx_len))

    model.zero_grad(set_to_none=True)
    loss_off, _ = polymorphic_loss(model, x, y, 0.5, 10.0)
    loss_off.backward()
    g_off = {n: p.grad.detach().clone() for n, p in model.named_parameters()
             if p.grad is not None}

    model.zero_grad(set_to_none=True)
    loss_on, _ = polymorphic_loss(model, x, y, 0.5, 10.0,
                                  gradient_checkpointing=True)
    loss_on.backward()
    g_on = {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.grad is not None}

    torch.testing.assert_close(loss_off.detach(), loss_on.detach(),
                               rtol=1e-5, atol=1e-6)
    for name in g_off:
        torch.testing.assert_close(
            g_off[name], g_on[name], rtol=1e-4, atol=1e-6,
            msg=lambda m, n=name: f"gradient mismatch for {n}: {m}")


def test_symmetric_branch_forwards_gradient_checkpointing_flag():
    """The symmetric consistency branch must honour the config key.

    Regression guard: before this change the branch called model(...) without
    gradient_checkpointing, so 3B ternary could only fit via memory_efficient,
    which silently switched the objective to teacher_detach.
    """
    import inspect

    from fogen.training import train as train_mod

    src = inspect.getsource(train_mod.main)
    # Locate the symmetric branch: it is the one building mask_set / masks_per_step.
    assert "mask_set" in src, "symmetric branch not found"
    marker = src.index("mask_set = [execution_mask]")
    end = src.index("execution_metrics = {", marker)
    branch = src[marker:end]
    assert "gradient_checkpointing" in branch, (
        "symmetric consistency branch must pass gradient_checkpointing to "
        "model(...) so 3B/7B can run without memory_efficient/teacher_detach")
