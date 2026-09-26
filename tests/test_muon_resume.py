"""Muon must survive a save/load round-trip with bf16 parameters.

torch.optim.Optimizer.load_state_dict casts floating-point state tensors to the
dtype of their parameter. With `bf16_params: true` that silently downcasts Muon's
fp32 momentum_buffer and fp32_copy to bf16, and the next step() dies with
    RuntimeError: expected dtype c10::BFloat16 for `end` but got dtype float
This broke `--resume` for every bf16 run (observed: 3B 6-mode at step 3000).
"""
from __future__ import annotations

import torch

from fogen.training.muon import Muon


def _bf16_matrix_params(n=2, shape=(32, 16)):
    return [torch.nn.Parameter(torch.randn(*shape, dtype=torch.bfloat16))
            for _ in range(n)]


def _step(opt, params):
    for p in params:
        p.grad = torch.randn_like(p, dtype=torch.float32).to(p.dtype)
    opt.step()


def test_muon_state_is_fp32_when_params_are_bf16():
    params = _bf16_matrix_params()
    opt = Muon(params, lr=0.01)
    _step(opt, params)
    for p in params:
        st = opt.state[p]
        assert st["momentum_buffer"].dtype is torch.float32
        assert st["fp32_copy"].dtype is torch.float32


def test_muon_resumes_after_state_dict_round_trip_bf16():
    params = _bf16_matrix_params()
    opt = Muon(params, lr=0.01)
    _step(opt, params)
    sd = opt.state_dict()

    fresh = _bf16_matrix_params()
    for f, p in zip(fresh, params):
        f.data.copy_(p.data)
    opt2 = Muon(fresh, lr=0.01)
    opt2.load_state_dict(sd)

    # fp32 invariants must survive the load, else step() raises
    for p in fresh:
        st = opt2.state[p]
        assert st["momentum_buffer"].dtype is torch.float32, (
            "momentum_buffer was downcast by load_state_dict")
        if "fp32_copy" in st:
            assert st["fp32_copy"].dtype is torch.float32, (
                "fp32_copy was downcast by load_state_dict")

    _step(opt2, fresh)  # must not raise


def test_muon_resume_fp32_params_unaffected():
    """Regression guard: the fp32 path must keep working unchanged."""
    params = [torch.nn.Parameter(torch.randn(32, 16))]
    opt = Muon(params, lr=0.01)
    _step(opt, params)
    assert "fp32_copy" not in opt.state[params[0]]
    sd = opt.state_dict()
    opt2 = Muon(params, lr=0.01)
    opt2.load_state_dict(sd)
    assert opt2.state[params[0]]["momentum_buffer"].dtype is torch.float32
    _step(opt2, params)
