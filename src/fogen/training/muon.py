"""Muon optimizer (Jordan et al. 2024): momentum + Newton-Schulz orthogonalization.

Used for 2-D matrix parameters only; embeddings/head/scalars use AdamW.
"""

import torch


@torch.no_grad()
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.04, momentum=0.95, weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    def load_state_dict(self, state_dict):
        """Restore fp32 momentum / master weights after a state-dict load.

        torch.optim.Optimizer.load_state_dict casts floating-point state tensors
        to their parameter's dtype. Under `bf16_params: true` that downcasts
        `momentum_buffer` and `fp32_copy` to bf16, and the next step() raises
        "expected dtype c10::BFloat16 for `end`". Muon requires both in fp32, so
        put them back.

        Note the master weights are bf16-rounded by this round trip. That matches
        what resume already does to the parameters themselves (checkpoints are
        saved via `.bfloat16()`), so it adds no precision loss beyond resume's.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state:
                    continue
                for key in ("momentum_buffer", "fp32_copy"):
                    buf = state.get(key)
                    if isinstance(buf, torch.Tensor) and buf.dtype is not torch.float32:
                        state[key] = buf.float()

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad, dtype=torch.float32)
                    if p.dtype != torch.float32:
                        state["fp32_copy"] = p.data.float()
                buf = state["momentum_buffer"]
                grad_f32 = p.grad.float()
                buf.lerp_(grad_f32, 1 - group["momentum"])
                g = grad_f32.lerp(buf, group["momentum"])  # nesterov
                g = newton_schulz(g, group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if "fp32_copy" in state:
                    fp32_p = state["fp32_copy"]
                    if group["weight_decay"] > 0:
                        fp32_p.mul_(1 - group["lr"] * group["weight_decay"])
                    fp32_p.add_(g, alpha=-group["lr"] * scale)
                    p.copy_(fp32_p)
                else:
                    if group["weight_decay"] > 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(g, alpha=-group["lr"] * scale)
        return loss
