"""Config-driven training with in-loop probe evals and dense checkpointing.

Usage:
  python -m fogen.training.train --config configs/v1_repro.yaml --seed 42 \
      [--out runs/v1_repro_s42] [--no-wandb]

Reproduces the v1 setup: Muon (matrices) + AdamW (embeddings), wd decaying
to 0, cosine warmdown over the final fraction of steps, bf16 autocast,
probe battery scored every probe_every steps (every step for the first 50).
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from fogen.data import ShardedLoader, load_tokenizer
from fogen.evals.scoring import aggregate, fogen_scorer, load_battery
from fogen.model import GPT, ModelConfig
from fogen.training.margin_guard import forced_choice_margin, project_gradient_
from fogen.training.muon import Muon
from fogen.training.tracking import create_tracker


def active_loader(step, loader_a, loader_b=None, switch_step=None):
    """Windowed exposure (Step-6T amendment 2026-06-11): batches for
    step < switch_step draw from loader_a, step >= switch_step from
    loader_b. Without data.phase_b in the config, loader_a serves every
    step — the pre-amendment behavior, bit-for-bit."""
    if loader_b is not None and step >= switch_step:
        return loader_b
    return loader_a


def lr_scale(step: int, total: int, warmdown_frac: float) -> float:
    start = int(total * (1 - warmdown_frac))
    if step < start:
        return 1.0
    t = (step - start) / max(1, total - start)
    return 0.5 * (1 + math.cos(math.pi * t))


def random_execution_mask(n_layers, parallel_probability, generator,
                          skip_probability=0.0):
    r = torch.rand(n_layers, generator=generator)
    modes = []
    for v in r.tolist():
        if v < skip_probability:
            modes.append("skip")
        elif v < skip_probability + parallel_probability:
            modes.append("parallel")
        else:
            modes.append("sequential")
    return modes


def curriculum_skip_probability(step, total, execution_cfg):
    """Skip probability for this step, optionally ramped from a lower start.

    Lets the model establish sequential/parallel alignment before layer removal
    is introduced. `skip_curriculum_frac` is the fraction of training over which
    the probability ramps linearly; 0 (default) disables the curriculum.
    """
    target = execution_cfg.get("skip_probability", 0.0)
    frac = execution_cfg.get("skip_curriculum_frac", 0.0)
    if frac <= 0.0:
        return target
    start = execution_cfg.get("skip_probability_start", 0.0)
    ramp_end = max(1, int(total * frac))
    if step >= ramp_end:
        return target
    return start + (target - start) * (step / ramp_end)


_DEFAULT_6MODE_PROBS = {
    "sequential": 0.3,
    "parallel": 0.3,
    "skip": 0.1,
    "reverse": 0.15,
    "attn_only": 0.075,
    "ffn_only": 0.075,
}


def random_execution_mask_6mode(n_layers, mode_probabilities, generator):
    """Sample a per-layer execution mask from 6 modes with given probabilities."""
    probs = mode_probabilities or _DEFAULT_6MODE_PROBS
    mode_names = list(probs.keys())
    cum_probs = []
    cumulative = 0.0
    for name in mode_names:
        cumulative += probs[name]
        cum_probs.append(cumulative)
    r = torch.rand(n_layers, generator=generator)
    modes = []
    for v in r.tolist():
        chosen = mode_names[-1]
        for i, threshold in enumerate(cum_probs):
            if v < threshold:
                chosen = mode_names[i]
                break
        modes.append(chosen)
    return modes


_gradnorm_cache = {"cw": 0.1, "step": -1}


def _record_gradnorm_diag(cw, norm_lm, norm_con, rho, cap):
    """Stash the quantities needed to see cap starvation.

    Gradnorm targets cw * ||grad_con|| == rho * ||grad_lm||. When the cap binds,
    the achieved ratio silently falls below rho, which is invisible if only cw is
    logged.
    """
    required = float(rho * norm_lm / max(norm_con, 1e-8))
    _gradnorm_cache["grad_norm_lm"] = float(norm_lm)
    _gradnorm_cache["grad_norm_con"] = float(norm_con)
    _gradnorm_cache["cw_required"] = required
    _gradnorm_cache["cw_ratio_achieved"] = float(cw * norm_con / max(norm_lm, 1e-12))
    _gradnorm_cache["cw_capped"] = float(required > cap)


def _gradnorm_diag(device):
    """Latest gradnorm diagnostics as tensors, for the metric logger."""
    keys = ("grad_norm_lm", "grad_norm_con", "cw_required",
            "cw_ratio_achieved", "cw_capped")
    return {k: torch.tensor(_gradnorm_cache[k], device=device)
            for k in keys if k in _gradnorm_cache}


def _gradnorm_cw_inline(model, x, y, execution_mask, execution_cfg, rho, step=0):
    """Exact gradient-normalized cw for random_mask_consistent.

    Same approach as _gradnorm_cw but for the ternary consistent branch:
    separate forward passes with batch=1 and gradient checkpointing.

      LM  = 0.5 * CE(seq) + 0.5 * CE(mask)
      con = consistency(seq_logits, mask_logits)

    Only recomputes every gradnorm_every steps (default 100).
    """
    warmup = execution_cfg.get("gradnorm_warmup_steps", 0)
    if step < warmup:
        fallback = execution_cfg.get("consistency_weight", 0.1)
        if _gradnorm_cache["step"] < 0:
            print(f"  [gradnorm-ternary] warmup until step {warmup}, using fixed cw={fallback}",
                  flush=True)
            _gradnorm_cache["step"] = 0
        _gradnorm_cache["cw"] = fallback
        return fallback
    every = execution_cfg.get("gradnorm_every", 100)
    if step > 0 and (step - _gradnorm_cache["step"]) < every:
        return _gradnorm_cache["cw"]

    print(f"  [gradnorm-ternary] computing at step {step}...", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    con_type = execution_cfg.get("consistency_type", "centered_mse")
    con_temp = execution_cfg.get("consistency_temperature", 1.0)

    def _grad_norm(loss):
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return sum(g.detach().float().norm().item() ** 2
                   for g in grads if g is not None) ** 0.5

    gn_batch = min(2, x.size(0))
    x_gn, y_gn = x[:gn_batch], y[:gn_batch]

    # LM gradient: separate fwd+bwd, then free
    print(f"  [gradnorm-ternary] LM fwd+grad (batch={gn_batch})...", flush=True)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                        enabled=x.device.type != "cpu"):
        seq_logits = model(x_gn, mode="sequential", gradient_checkpointing=True)
        mask_logits = model(x_gn, mode=execution_mask, gradient_checkpointing=True)
        lm_loss = 0.5 * F.cross_entropy(
            seq_logits.view(-1, seq_logits.size(-1)), y_gn.reshape(-1)) + \
            0.5 * F.cross_entropy(
            mask_logits.view(-1, mask_logits.size(-1)), y_gn.reshape(-1))
    norm_lm = _grad_norm(lm_loss)
    del seq_logits, mask_logits, lm_loss
    model.zero_grad(set_to_none=True)

    # Consistency gradient: separate fwd+bwd, then free
    print(f"  [gradnorm-ternary] con fwd+grad (batch={gn_batch})...", flush=True)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                        enabled=x.device.type != "cpu"):
        seq_logits = model(x_gn, mode="sequential", gradient_checkpointing=True)
        mask_logits = model(x_gn, mode=execution_mask, gradient_checkpointing=True)
        con_loss = _compute_consistency(
            seq_logits, mask_logits, con_type, temperature=con_temp)
    norm_con = _grad_norm(con_loss)
    del seq_logits, mask_logits, con_loss
    model.zero_grad(set_to_none=True)

    cap = execution_cfg.get("gradnorm_cw_max", 10.0)
    cw = float(rho * norm_lm / max(norm_con, 1e-8))
    cw = max(1e-4, min(cw, cap))
    _record_gradnorm_diag(cw, norm_lm, norm_con, rho, cap)
    # If con gradient is negligible (<1% of LM), use rho as default
    if norm_con < 0.01 * norm_lm:
        cw = _gradnorm_cache["cw"] if _gradnorm_cache.get("has_valid", False) else rho
        _gradnorm_cache["cw"] = cw
        print(f"  [gradnorm-ternary] ||∇LM||={norm_lm:.4f} ||∇con||={norm_con:.4f} (noise, using cw={cw:.4f})",
              flush=True)
    else:
        _gradnorm_cache["cw"] = cw
        _gradnorm_cache["has_valid"] = True
        print(f"  [gradnorm-ternary] ||∇LM||={norm_lm:.4f} ||∇con||={norm_con:.4f} cw={cw:.4f}",
              flush=True)
    _gradnorm_cache["step"] = step
    return cw


_losstarget_cache = {"cw": None}


def losstarget_cw(execution_cfg):
    """Current coefficient for the loss-targeting controller.

    Gradnorm targets cw*||grad_con||/||grad_lm|| = rho. Measured across runs that
    product is nearly scale-invariant (cw 10->100 all give rho_eff 0.02-0.03), so
    the ratio does not identify an operating point and the controller settles
    wherever it first satisfies the condition -- the low-cw, low-agreement end.

    The consistency loss level does discriminate (2.9e-4 -> 99.56% agreement vs
    1.6e-3 -> 98.44%), so we control on that instead. Costs no extra forward
    passes: the consistency loss is already computed by the training step.
    """
    if _losstarget_cache["cw"] is None:
        _losstarget_cache["cw"] = execution_cfg.get("consistency_weight", 1.0)
    return _losstarget_cache["cw"]


def losstarget_target(execution_cfg, step, total):
    """Target consistency loss at this step, optionally ramped.

    A constant target measured late in training is unreachable early: at step 100
    of a 6-mode run the observed consistency loss is ~100x the late-training
    value, so the controller drives cw to its ceiling during exactly the window
    where high cw damages the model. Ramping the target geometrically from a
    loose initial value tracks the natural decline instead.

    `consistency_loss_target_start` and `consistency_loss_target_frac` enable it;
    without them the target is constant (previous behaviour).
    """
    final = execution_cfg.get("consistency_loss_target")
    if final is None:
        return None
    start = execution_cfg.get("consistency_loss_target_start")
    frac = execution_cfg.get("consistency_loss_target_frac", 0.0)
    if start is None or frac <= 0.0:
        return final
    progress = min(1.0, step / max(1.0, total * frac))
    # geometric interpolation: spans orders of magnitude smoothly
    return float(start * (final / start) ** progress)


def losstarget_update(con_loss, execution_cfg, step, total=1):
    """Multiplicative update in log-space toward the target consistency loss."""
    target = losstarget_target(execution_cfg, step, total)
    if target is None:
        return None
    every = execution_cfg.get("cw_control_every", 20)
    if step % every:
        return _losstarget_cache["cw"]
    gain = execution_cfg.get("cw_control_gain", 0.5)
    max_step = execution_cfg.get("cw_control_max_step", 1.3)
    cw = _losstarget_cache["cw"]
    obs_raw = float(con_loss)
    if not (obs_raw > 1e-10):
        # step 0 (and any step where consistency was not computed) carries no
        # information; acting on it takes a spurious downward step.
        return cw

    # Smooth the observation before acting on it. A single random multi-mode mask
    # gives a very noisy consistency loss (measured: obs/target swinging 0.56-2.17
    # at 430M 6-mode), so an unsmoothed loop chases sampling noise -- 39% of
    # updates hit the rate limit and cw swung 6.8x. Same variance that broke
    # gradnorm on 6-mode. EMA in log space, since obs spans orders of magnitude.
    beta = execution_cfg.get("cw_control_ema", 0.9)
    prev = _losstarget_cache.get("obs_ema")
    if beta <= 0.0 or prev is None:
        obs = obs_raw
    else:
        obs = math.exp(beta * math.log(prev) + (1.0 - beta) * math.log(obs_raw))
    _losstarget_cache["obs_ema"] = obs
    factor = (obs / target) ** gain
    factor = min(max(factor, 1.0 / max_step), max_step)
    cw_raw = float(cw * factor)
    cap = execution_cfg.get("cw_control_max", 1e6)
    cw = max(execution_cfg.get("cw_control_min", 1e-3), min(cw_raw, cap))
    at_cap = cw_raw >= cap
    rate_limited = (factor >= max_step - 1e-9) or (factor <= 1.0 / max_step + 1e-9)
    _losstarget_cache.update({
        "cw": cw, "target": float(target), "obs": obs, "obs_raw": obs_raw,
        "obs_over_target": obs / target, "at_cap": float(at_cap),
        "rate_limited": float(rate_limited),
    })
    print(f"  [losstarget] step={step} con={obs_raw:.4e} ema={obs:.4e} "
          f"target={target:.4e} obs/tgt={obs / target:6.2f} cw={cw:.3f}"
          f"{' [CAP]' if at_cap else ''}{' [RATE-LIMITED]' if rate_limited else ''}",
          flush=True)
    return cw


def losstarget_diag(device):
    """Controller diagnostics for the metric logger.

    Without these, a converged run and a run pegged at cw_control_max look
    identical in the logs -- the same blind spot that hid gradnorm's
    cap-starvation (F23).
    """
    keys = ("target", "obs_over_target", "at_cap", "rate_limited")
    return {f"cw_{k}": torch.tensor(_losstarget_cache[k], device=device)
            for k in keys if k in _losstarget_cache}


def _gradnorm_cw(model, x, y, execution_cfg, rho, step=0):
    """Compute gradient-normalized consistency weight: cw = ρ * ||∇LM|| / ||∇con||.

    Measures gradients of exactly the same objectives used in the training step:
      LM = (1-pw)*CE_seq + pw*CE_par
      con = consistency(seq_logits, par_logits)  [symmetric, through both branches]

    Only recomputes every gradnorm_every steps (default 100) to amortize cost.
    Clamps output to [1e-4, 10.0].
    """
    warmup = execution_cfg.get("gradnorm_warmup_steps", 0)
    every = execution_cfg.get("gradnorm_every", 100)
    if step < warmup:
        fallback = execution_cfg.get("consistency_weight", 0.1)
        if _gradnorm_cache["step"] < 0:
            print(f"  [gradnorm] ramp over {warmup} steps from cw={fallback}",
                  flush=True)
            _gradnorm_cache["step"] = 0
        _gradnorm_cache["cw"] = fallback
        return fallback
    if (step - _gradnorm_cache["step"]) < every:
        return _gradnorm_cache["cw"]

    print(f"  [gradnorm] computing at step {step}...", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    pw = execution_cfg.get("parallel_weight", 0.5)
    con_type = execution_cfg.get("consistency_type", "centered_mse")
    con_temp = execution_cfg.get("consistency_temperature", 1.0)
    td = execution_cfg.get("teacher_detach", False)

    def _grad_norm(loss, retain=False):
        grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        return sum(g.detach().float().norm().item() ** 2
                   for g in grads if g is not None) ** 0.5

    # Use a small batch slice for gradient measurement to avoid OOM at 3B+
    gn_batch = min(2, x.size(0))
    x_gn, y_gn = x[:gn_batch], y[:gn_batch]

    print(f"  [gradnorm] LM fwd+grad (batch={gn_batch})...", flush=True)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                        enabled=x.device.type != "cpu"):
        seq_logits = model(x_gn, mode="sequential", gradient_checkpointing=True)
        par_logits = model(x_gn, mode="parallel", gradient_checkpointing=True)
        lm_loss = (
            (1 - pw) * F.cross_entropy(
                seq_logits.view(-1, seq_logits.size(-1)), y_gn.reshape(-1))
            + pw * F.cross_entropy(
                par_logits.view(-1, par_logits.size(-1)), y_gn.reshape(-1)))
    norm_lm = _grad_norm(lm_loss)
    del seq_logits, par_logits, lm_loss
    model.zero_grad(set_to_none=True)

    print(f"  [gradnorm] con fwd+grad (batch={gn_batch})...", flush=True)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                        enabled=x.device.type != "cpu"):
        seq_logits = model(x_gn, mode="sequential", gradient_checkpointing=True)
        par_logits = model(x_gn, mode="parallel", gradient_checkpointing=True)
        con_loss = _compute_consistency(
            seq_logits, par_logits, con_type,
            teacher_detach=td, temperature=con_temp)
    norm_con = _grad_norm(con_loss)
    del seq_logits, par_logits, con_loss
    model.zero_grad(set_to_none=True)

    cap = execution_cfg.get("gradnorm_cw_max", 10.0)
    cw = float(rho * norm_lm / max(norm_con, 1e-8))
    cw = max(1e-4, min(cw, cap))
    _record_gradnorm_diag(cw, norm_lm, norm_con, rho, cap)
    if warmup > 0 and step < 2 * warmup:
        fallback = execution_cfg.get("consistency_weight", 0.1)
        alpha = min(1.0, (step - warmup) / warmup)
        raw_cw = cw
        cw = fallback + alpha * (cw - fallback)
        print(f"  [gradnorm] ||∇LM||={norm_lm:.4f} ||∇con||={norm_con:.4f} "
              f"raw={raw_cw:.4f} ramp={alpha:.2f} cw={cw:.4f}", flush=True)
    else:
        print(f"  [gradnorm] ||∇LM||={norm_lm:.4f} ||∇con||={norm_con:.4f} cw={cw:.4f}",
              flush=True)
    _gradnorm_cache["cw"] = cw
    _gradnorm_cache["step"] = step
    return cw


def consistency_weight(step, total, config):
    start_weight = config.get("consistency_weight", 0.1)
    end_weight = config.get("consistency_weight_end", start_weight)
    decay_start = int(total * config.get("consistency_decay_start", 1.0))
    decay_end = int(total * config.get("consistency_decay_end", 1.0))
    if step <= decay_start:
        return start_weight
    if step >= decay_end or decay_end <= decay_start:
        return end_weight
    progress = (step - decay_start) / (decay_end - decay_start)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return end_weight + (start_weight - end_weight) * cosine


def layerwise_consistency(seq_hidden, alt_hidden, teacher_detach=False,
                          normalize=True):
    """Per-layer hidden-state divergence, averaged over layers.

    Motivation: the composition law is additive over per-layer defects
    (probe-only Spearman 0.92 across 6^20 masks, F28/F37), so the per-layer
    hidden divergence is the causal quantity. Aligning only the final logits
    corrects the *accumulated* divergence at the output; aligning per layer
    corrects it at source, which should need less distortion of the sequential
    pathway -- i.e. it may move the quality/alignment frontier rather than just
    sliding along it.

    Normalized by the sequential activation RMS so deep layers (larger scale)
    do not dominate.
    """
    total = 0.0
    for hs, ha in zip(seq_hidden, alt_hidden):
        tgt = hs.detach() if teacher_detach else hs
        diff = (ha - tgt).float()
        if normalize:
            scale = tgt.detach().float().pow(2).mean().clamp_min(1e-8).sqrt()
            diff = diff / scale
        total = total + diff.pow(2).mean()
    return total / max(len(seq_hidden), 1)


def _compute_consistency(sequential_logits, parallel_logits, consistency_type,
                         teacher_detach=False, temperature=1.0):
    if consistency_type == "margin_mse":
        # Agreement is lost at near-ties: over-weighting flattens the logit
        # landscape and flips tokens whose top-2 margin was already small (F22).
        # Uniform centered-MSE spends most of its budget on positions that were
        # never going to flip. Weight by exp(-margin/tau) so effort concentrates
        # where argmax agreement is actually decided. Weights are normalized to
        # mean 1 so the effective coefficient scale matches centered_mse.
        target = sequential_logits.detach() if teacher_detach else sequential_logits
        with torch.no_grad():
            top2 = sequential_logits.detach().float().topk(2, dim=-1).values
            margin = (top2[..., 0] - top2[..., 1]).clamp_min(0.0)
            w = torch.exp(-margin / max(temperature, 1e-6))
            w = w / w.mean().clamp_min(1e-8)
        d = (parallel_logits - parallel_logits.mean(dim=-1, keepdim=True)) - \
            (target - target.mean(dim=-1, keepdim=True))
        return (w.unsqueeze(-1) * d.pow(2)).mean()
    if consistency_type == "raw_mse":
        target = sequential_logits.detach() if teacher_detach else sequential_logits
        return F.mse_loss(parallel_logits, target)
    if consistency_type == "kl_forward":
        seq_logprob = F.log_softmax(
            (sequential_logits.detach() if teacher_detach else sequential_logits) / temperature,
            dim=-1)
        par_logprob = F.log_softmax(parallel_logits / temperature, dim=-1)
        return F.kl_div(par_logprob, seq_logprob.exp(), reduction="batchmean") * (temperature ** 2)
    if consistency_type == "symmetric_kl":
        seq_lp = F.log_softmax(
            (sequential_logits.detach() if teacher_detach else sequential_logits) / temperature,
            dim=-1)
        par_lp = F.log_softmax(parallel_logits / temperature, dim=-1)
        return (F.kl_div(par_lp, seq_lp.exp(), reduction="batchmean")
                + F.kl_div(seq_lp, par_lp.exp(), reduction="batchmean")) / 2 * (temperature ** 2)
    if consistency_type == "jensen_shannon":
        seq_lp = F.log_softmax(
            (sequential_logits.detach() if teacher_detach else sequential_logits) / temperature,
            dim=-1)
        par_lp = F.log_softmax(parallel_logits / temperature, dim=-1)
        m = (seq_lp.exp() + par_lp.exp()) / 2
        return (F.kl_div(seq_lp, m, reduction="batchmean")
                + F.kl_div(par_lp, m, reduction="batchmean")) / 2 * (temperature ** 2)
    # Default: centered_mse
    seq_centered = sequential_logits - sequential_logits.mean(dim=-1, keepdim=True)
    par_centered = parallel_logits - parallel_logits.mean(dim=-1, keepdim=True)
    target = seq_centered.detach() if teacher_detach else seq_centered
    return F.mse_loss(par_centered, target)


def polymorphic_loss(model, inputs, targets, parallel_weight, consistency_weight,
                     teacher_detach=False, memory_efficient=False,
                     consistency_type="centered_mse", temperature=1.0,
                     gradient_checkpointing=False):
    if memory_efficient:
        return _polymorphic_loss_memory_efficient(
            model, inputs, targets, parallel_weight, consistency_weight,
            consistency_type=consistency_type, temperature=temperature)
    sequential_logits = model(
        inputs, mode="sequential",
        gradient_checkpointing=gradient_checkpointing)
    parallel_logits = model(
        inputs, mode="parallel",
        gradient_checkpointing=gradient_checkpointing)
    sequential_loss = F.cross_entropy(
        sequential_logits.view(-1, sequential_logits.size(-1)), targets.reshape(-1))
    parallel_loss = F.cross_entropy(
        parallel_logits.view(-1, parallel_logits.size(-1)), targets.reshape(-1))
    consistency = _compute_consistency(
        sequential_logits, parallel_logits, consistency_type, teacher_detach, temperature)
    total = (
        (1 - parallel_weight) * sequential_loss
        + parallel_weight * parallel_loss
        + consistency_weight * consistency
    )
    return total, {
        "sequential_loss": sequential_loss,
        "parallel_loss": parallel_loss,
        "consistency": consistency,
    }


def _polymorphic_loss_memory_efficient(model, inputs, targets, parallel_weight,
                                       consistency_weight,
                                       consistency_type="centered_mse",
                                       temperature=1.0):
    """Backward each graph separately with gradient checkpointing.

    Uses teacher_detach semantics for the consistency term and recomputes
    layer activations during backward. This reduces peak memory from
    ~2x model activations to ~1x per-layer activations, enabling 7B
    training on a single 96GB GPU.
    """
    # Forward + backward sequential path (with gradient checkpointing)
    sequential_logits = model(inputs, mode="sequential", gradient_checkpointing=True)
    sequential_loss = F.cross_entropy(
        sequential_logits.view(-1, sequential_logits.size(-1)), targets.reshape(-1))
    ((1 - parallel_weight) * sequential_loss).backward()
    sequential_logits_detached = sequential_logits.detach()
    sequential_loss_val = sequential_loss.detach()
    del sequential_logits

    # Forward + backward parallel path (with gradient checkpointing)
    parallel_logits = model(inputs, mode="parallel", gradient_checkpointing=True)
    parallel_loss = F.cross_entropy(
        parallel_logits.view(-1, parallel_logits.size(-1)), targets.reshape(-1))
    consistency = _compute_consistency(
        sequential_logits_detached, parallel_logits, consistency_type,
        teacher_detach=True, temperature=temperature)
    (parallel_weight * parallel_loss + consistency_weight * consistency).backward()
    parallel_loss_val = parallel_loss.detach()
    consistency_val = consistency.detach()
    del parallel_logits

    total = (
        (1 - parallel_weight) * sequential_loss_val
        + parallel_weight * parallel_loss_val
        + consistency_weight * consistency_val
    )
    return total, {
        "sequential_loss": sequential_loss_val,
        "parallel_loss": parallel_loss_val,
        "consistency": consistency_val,
    }


def save_checkpoint(model, out_dir: Path, step: int,
                    muon=None, adamw=None):
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.bfloat16() for k, v in model.state_dict().items()
             if not k.startswith("rope_")}
    save_file(state, str(out_dir / f"step{step:06d}.safetensors"))
    if muon is not None and adamw is not None:
        torch.save({"muon": muon.state_dict(), "adamw": adamw.state_dict(),
                     "step": step}, str(out_dir / f"opt{step:06d}.pt"))


def checkpoint_steps(cfg: dict, total: int) -> set[int]:
    steps = {0, total}
    every = cfg.get("ckpt_every", 100)
    steps.update(range(0, total + 1, every))
    for lo, hi, dense in cfg.get("dense_windows", []):
        steps.update(range(lo, min(hi, total) + 1, dense))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--init-ckpt")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from latest checkpoint in output dir")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--mlflow", action="store_true",
                    help="Use MLflow for experiment tracking")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out or f"runs/{cfg['run_name']}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.dump({**cfg, "seed": args.seed}))

    mcfg = ModelConfig(**cfg["model"])
    param_dtype = torch.bfloat16 if cfg.get("train", {}).get("bf16_params", False) else torch.float32
    model = GPT(mcfg).to(device=device, dtype=param_dtype)

    resume_step = 0
    resume_opt_path = None
    init_ckpt = args.init_ckpt
    if args.resume:
        ckpt_dir = out / "ckpts"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("step*.safetensors"))
            if ckpts:
                init_ckpt = str(ckpts[-1])
                resume_step = int(ckpts[-1].stem.replace("step", ""))
                opt_path = ckpt_dir / f"opt{resume_step:06d}.pt"
                if opt_path.exists():
                    resume_opt_path = str(opt_path)
                print(f"Resuming from {init_ckpt} (step {resume_step})"
                      f"{' + optimizer' if resume_opt_path else ' (no optimizer state)'}")
    if init_ckpt:
        from safetensors.torch import load_file
        state = {key: value.to(param_dtype) for key, value in load_file(init_ckpt).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected
        assert all(key.startswith("rope_") for key in missing)
    print(f"params: {model.num_params()/1e6:.1f}M  device: {device}")

    tokenizer = load_tokenizer(cfg["data"]["tokenizer_dir"])
    loader = ShardedLoader(cfg["data"]["shard_dir"], cfg["batch_seqs"],
                           mcfg.ctx_len, seed=args.seed, device=device,
                           one_doc_per_seq=cfg["data"].get("one_doc_per_seq", False),
                           mask_padding=cfg["data"].get("mask_padding", False),
                           max_tokens=cfg["data"].get("max_tokens"))
    pb = cfg["data"].get("phase_b")
    loader_b, switch_step = None, None
    if pb:
        # seed+1: independent offset stream for phase B, avoids replaying
        # phase A's offsets on a different shard set
        loader_b = ShardedLoader(pb["shard_dir"], cfg["batch_seqs"],
                                 mcfg.ctx_len, seed=args.seed + 1,
                                 device=device,
                                 one_doc_per_seq=cfg["data"].get("one_doc_per_seq", False),
                                 mask_padding=cfg["data"].get("mask_padding", False),
                                 max_tokens=pb.get("max_tokens"))
        switch_step = int(pb["switch_step"])
    battery = load_battery(cfg["probes"]["battery_path"])
    scorer = fogen_scorer(model, tokenizer, device=device,
                          batch_size=cfg["probes"].get("batch_size", 256))
    guard_cfg = cfg.get("margin_guard", {})
    guard_items = []
    if guard_cfg.get("enabled", False):
        guard_items = [item for item in battery
                       if item["probe"] == guard_cfg["probe"]
                       and item["split"] == guard_cfg.get("split", "train")]
        guard_items = guard_items[:guard_cfg.get("max_items", len(guard_items))]
        if not guard_items:
            raise ValueError("margin_guard selected no probe items")

    t = cfg["train"]
    execution_cfg = cfg.get("execution_training", {})
    execution_generator = torch.Generator().manual_seed(args.seed + 10_000)
    muon = Muon(model.matrix_params(), lr=t["matrix_lr"],
                weight_decay=t["weight_decay"])
    # head_lr (optional): separate unembedding LR, as in v1's released
    # train.py (unembedding_lr 0.004 vs embedding_lr 0.2). Absent -> head
    # stays in the embed group, preserving all pre-2026-06-10 runs.
    if t.get("head_lr") is not None:
        adamw_groups = [dict(params=model.embed_params(exclude_head=True),
                             lr=t["embed_lr"]),
                        dict(params=model.head_params(), lr=t["head_lr"])]
    else:
        adamw_groups = [dict(params=model.embed_params(), lr=t["embed_lr"])]
    if model.scalar_params():
        # ve mixing scalars; v1 paper gives no scalar LR, use matrix LR
        adamw_groups.append(dict(params=model.scalar_params(), lr=t["matrix_lr"]))
    adamw = torch.optim.AdamW(adamw_groups, betas=(0.9, 0.95), weight_decay=0.0)
    for g in adamw.param_groups:
        g["base_lr"] = g["lr"]
    if resume_opt_path:
        opt_state = torch.load(resume_opt_path, map_location=device, weights_only=False)
        muon.load_state_dict(opt_state["muon"])
        adamw.load_state_dict(opt_state["adamw"])
        print(f"Loaded optimizer state from step {opt_state['step']}")
    total = t["steps"]
    ckpt_at = checkpoint_steps(cfg.get("checkpointing", {}), total)

    tracker = create_tracker(
        {**cfg, "seed": args.seed}, out,
        use_mlflow=args.mlflow, use_wandb=not args.no_wandb)

    probe_log = (out / "probe_log.jsonl").open("a")
    train_log = (out / "train_log.jsonl").open("a")

    def run_probes(step: int):
        model.eval()
        rows = scorer.score_items(battery)
        aggs = aggregate(rows)
        for a in aggs:
            probe_log.write(json.dumps({"step": step, **a}) + "\n")
        probe_log.flush()
        tracker.log_probes(step, aggs)
        model.train()

    model.train()
    if resume_step > 0:
        print(f"Fast-forwarding data loader to step {resume_step}...")
        for skip in range(resume_step):
            active_loader(skip, loader, loader_b, switch_step).next_batch()
        print(f"Resumed. Starting from step {resume_step}.")
    t0 = time.time()
    for step in range(resume_step, total + 1):
        s = lr_scale(step, total, t.get("warmdown_frac", 0.3))
        wd = t["weight_decay"] * (1 - step / total)  # decay wd to 0
        for g in muon.param_groups:
            g["lr"], g["weight_decay"] = t["matrix_lr"] * s, wd
        for g in adamw.param_groups:
            g["lr"] = g["base_lr"] * s

        if step in ckpt_at:
            save_checkpoint(model, out / "ckpts", step, muon, adamw)
            tracker.log_checkpoint(step)
        pe = cfg["probes"]["every"]
        if step <= cfg["probes"].get("dense_until", 50) or step % pe == 0:
            run_probes(step)
        if step == total:
            break

        x, y = active_loader(step, loader, loader_b, switch_step).next_batch()
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                            enabled=(device != "cpu")):
            if (execution_cfg.get("enabled", False)
                    and execution_cfg.get("strategy") == "random_mask_consistent"):
                execution_mask = random_execution_mask(
                    mcfg.n_layer,
                    execution_cfg.get("parallel_probability", 0.5),
                    execution_generator,
                    skip_probability=curriculum_skip_probability(
                        step, total, execution_cfg))
                # Measure gradnorm BEFORE training forward pass to avoid OOM
                gradnorm_rho = execution_cfg.get("gradnorm_rho")
                if gradnorm_rho is not None:
                    cw = _gradnorm_cw_inline(
                        model, x, y, execution_mask, execution_cfg,
                        gradnorm_rho, step=step)
                else:
                    cw = (losstarget_cw(execution_cfg)
                          if execution_cfg.get("consistency_loss_target") is not None
                          else consistency_weight(step, total, execution_cfg))
                con_type = execution_cfg.get("consistency_type", "centered_mse")
                con_temp = execution_cfg.get("consistency_temperature", 1.0)
                if execution_cfg.get("memory_efficient", False):
                    # Separate fwd+bwd: one graph alive at a time (fits 7B on 96GB)
                    seq_logits = model(x, mode="sequential", gradient_checkpointing=True)
                    seq_loss = F.cross_entropy(
                        seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1))
                    (0.5 * seq_loss).backward()
                    seq_logits_detached = seq_logits.detach()
                    seq_loss_val = seq_loss.detach()
                    del seq_logits, seq_loss

                    mask_logits = model(x, mode=execution_mask, gradient_checkpointing=True)
                    mask_loss = F.cross_entropy(
                        mask_logits.view(-1, mask_logits.size(-1)), y.reshape(-1))
                    consistency = _compute_consistency(
                        seq_logits_detached, mask_logits, con_type,
                        teacher_detach=True, temperature=con_temp)
                    (0.5 * mask_loss + cw * consistency).backward()
                    mask_loss_val = mask_loss.detach()
                    consistency_val = consistency.detach()
                    del mask_logits, mask_loss, consistency

                    loss = 0.5 * seq_loss_val + 0.5 * mask_loss_val + cw * consistency_val
                    execution_metrics = {
                        "sequential_loss": seq_loss_val,
                        "mask_loss": mask_loss_val,
                        "consistency": consistency_val,
                        "consistency_weight": torch.tensor(cw, device=loss.device),
                        **({"layerwise": layerwise} if execution_cfg.get("layerwise_weight", 0.0) > 0 else {}),
                        **_gradnorm_diag(loss.device),
                        **losstarget_diag(loss.device),
                    }
                    losstarget_update(
                        float(execution_metrics["consistency"])
                        if isinstance(execution_metrics, dict) else 0.0,
                        execution_cfg, step, total)
                else:
                    # masks_per_step > 1 aligns sequential against several
                    # programs per batch, densifying coverage of the program
                    # space at one extra forward pass per extra mask.
                    n_masks = max(1, execution_cfg.get("masks_per_step", 1))
                    mask_set = [execution_mask]
                    for _ in range(n_masks - 1):
                        mask_set.append(random_execution_mask(
                            mcfg.n_layer,
                            execution_cfg.get("parallel_probability", 0.5),
                            execution_generator,
                            skip_probability=curriculum_skip_probability(
                                step, total, execution_cfg)))
                    lw_weight = execution_cfg.get("layerwise_weight", 0.0)
                    td_flag = execution_cfg.get("teacher_detach", False)
                    grad_ckpt = execution_cfg.get(
                        "gradient_checkpointing", False)
                    want_h = lw_weight > 0.0
                    out_seq = model(x, mode="sequential", return_hidden=want_h,
                                    gradient_checkpointing=grad_ckpt)
                    seq_logits, seq_h = out_seq if want_h else (out_seq, None)
                    seq_loss = F.cross_entropy(
                        seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1))
                    mask_loss = 0.0
                    consistency = 0.0
                    layerwise = 0.0
                    for m in mask_set:
                        out_m = model(x, mode=m, return_hidden=want_h,
                                      gradient_checkpointing=grad_ckpt)
                        m_logits, m_h = out_m if want_h else (out_m, None)
                        mask_loss = mask_loss + F.cross_entropy(
                            m_logits.view(-1, m_logits.size(-1)), y.reshape(-1))
                        consistency = consistency + _compute_consistency(
                            seq_logits, m_logits, con_type, temperature=con_temp,
                            teacher_detach=td_flag)
                        if want_h:
                            layerwise = layerwise + layerwise_consistency(
                                seq_h, m_h, teacher_detach=td_flag)
                    mask_loss = mask_loss / len(mask_set)
                    consistency = consistency / len(mask_set)
                    loss = 0.5 * seq_loss + 0.5 * mask_loss + cw * consistency
                    if want_h:
                        layerwise = layerwise / len(mask_set)
                        loss = loss + lw_weight * layerwise
                    execution_metrics = {
                        "sequential_loss": seq_loss,
                        "mask_loss": mask_loss,
                        "consistency": consistency,
                        "consistency_weight": torch.tensor(cw, device=loss.device),
                        **({"layerwise": layerwise} if execution_cfg.get("layerwise_weight", 0.0) > 0 else {}),
                        **_gradnorm_diag(loss.device),
                        **losstarget_diag(loss.device),
                    }
                    losstarget_update(
                        float(execution_metrics["consistency"])
                        if isinstance(execution_metrics, dict) else 0.0,
                        execution_cfg, step, total)
            elif (execution_cfg.get("enabled", False)
                    and execution_cfg.get("strategy") == "random_mask_6mode"):
                execution_mask = random_execution_mask_6mode(
                    mcfg.n_layer,
                    execution_cfg.get("mode_probabilities"),
                    execution_generator)
                gradnorm_rho = execution_cfg.get("gradnorm_rho")
                if gradnorm_rho is not None:
                    cw = _gradnorm_cw_inline(
                        model, x, y, execution_mask, execution_cfg,
                        gradnorm_rho, step=step)
                else:
                    cw = (losstarget_cw(execution_cfg)
                          if execution_cfg.get("consistency_loss_target") is not None
                          else consistency_weight(step, total, execution_cfg))
                con_type = execution_cfg.get("consistency_type", "centered_mse")
                con_temp = execution_cfg.get("consistency_temperature", 1.0)
                # Default 0.5 preserves every existing 6-mode config's behavior
                # exactly. Real gap found this session: every 6-mode checkpoint
                # tested (gradnorm, losstarget, fixedcw) used this hardcoded
                # equal split, meaning the sequential path only ever got half
                # the LM-loss gradient signal a dedicated specialist gets --
                # never tested as a lever for narrowing the specialist gap.
                seq_w = execution_cfg.get("sequential_weight", 0.5)
                if execution_cfg.get("memory_efficient", False):
                    seq_logits = model(x, mode="sequential", gradient_checkpointing=True)
                    seq_loss = F.cross_entropy(
                        seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1))
                    (seq_w * seq_loss).backward()
                    seq_logits_detached = seq_logits.detach()
                    seq_loss_val = seq_loss.detach()
                    del seq_logits, seq_loss

                    mask_logits = model(x, mode=execution_mask, gradient_checkpointing=True)
                    mask_loss = F.cross_entropy(
                        mask_logits.view(-1, mask_logits.size(-1)), y.reshape(-1))
                    consistency = _compute_consistency(
                        seq_logits_detached, mask_logits, con_type,
                        teacher_detach=True, temperature=con_temp)
                    ((1 - seq_w) * mask_loss + cw * consistency).backward()
                    mask_loss_val = mask_loss.detach()
                    consistency_val = consistency.detach()
                    del mask_logits, mask_loss, consistency

                    loss = seq_w * seq_loss_val + (1 - seq_w) * mask_loss_val + cw * consistency_val
                    execution_metrics = {
                        "sequential_loss": seq_loss_val,
                        "mask_loss": mask_loss_val,
                        "consistency": consistency_val,
                        "consistency_weight": torch.tensor(cw, device=loss.device),
                        **({"layerwise": layerwise} if execution_cfg.get("layerwise_weight", 0.0) > 0 else {}),
                        **_gradnorm_diag(loss.device),
                        **losstarget_diag(loss.device),
                    }
                    losstarget_update(
                        float(execution_metrics["consistency"])
                        if isinstance(execution_metrics, dict) else 0.0,
                        execution_cfg, step, total)
                else:
                    # masks_per_step > 1 aligns sequential against several
                    # programs per batch, densifying coverage of the program
                    # space at one extra forward pass per extra mask.
                    n_masks = max(1, execution_cfg.get("masks_per_step", 1))
                    mask_set = [execution_mask]
                    for _ in range(n_masks - 1):
                        mask_set.append(random_execution_mask_6mode(
                            mcfg.n_layer,
                            execution_cfg.get("mode_probabilities"),
                            execution_generator))
                    lw_weight = execution_cfg.get("layerwise_weight", 0.0)
                    td_flag = execution_cfg.get("teacher_detach", False)
                    grad_ckpt = execution_cfg.get(
                        "gradient_checkpointing", False)
                    want_h = lw_weight > 0.0
                    out_seq = model(x, mode="sequential", return_hidden=want_h,
                                    gradient_checkpointing=grad_ckpt)
                    seq_logits, seq_h = out_seq if want_h else (out_seq, None)
                    seq_loss = F.cross_entropy(
                        seq_logits.view(-1, seq_logits.size(-1)), y.reshape(-1))
                    mask_loss = 0.0
                    consistency = 0.0
                    layerwise = 0.0
                    for m in mask_set:
                        out_m = model(x, mode=m, return_hidden=want_h,
                                      gradient_checkpointing=grad_ckpt)
                        m_logits, m_h = out_m if want_h else (out_m, None)
                        mask_loss = mask_loss + F.cross_entropy(
                            m_logits.view(-1, m_logits.size(-1)), y.reshape(-1))
                        consistency = consistency + _compute_consistency(
                            seq_logits, m_logits, con_type, temperature=con_temp,
                            teacher_detach=td_flag)
                        if want_h:
                            layerwise = layerwise + layerwise_consistency(
                                seq_h, m_h, teacher_detach=td_flag)
                    mask_loss = mask_loss / len(mask_set)
                    consistency = consistency / len(mask_set)
                    loss = seq_w * seq_loss + (1 - seq_w) * mask_loss + cw * consistency
                    if want_h:
                        layerwise = layerwise / len(mask_set)
                        loss = loss + lw_weight * layerwise
                    execution_metrics = {
                        "sequential_loss": seq_loss,
                        "mask_loss": mask_loss,
                        "consistency": consistency,
                        "consistency_weight": torch.tensor(cw, device=loss.device),
                        **({"layerwise": layerwise} if execution_cfg.get("layerwise_weight", 0.0) > 0 else {}),
                        **_gradnorm_diag(loss.device),
                        **losstarget_diag(loss.device),
                    }
                    losstarget_update(
                        float(execution_metrics["consistency"])
                        if isinstance(execution_metrics, dict) else 0.0,
                        execution_cfg, step, total)
            elif (execution_cfg.get("enabled", False)
                    and execution_cfg.get("strategy") == "random_mask"):
                execution_mask = random_execution_mask(
                    mcfg.n_layer,
                    execution_cfg.get("parallel_probability", 0.5),
                    execution_generator,
                    skip_probability=curriculum_skip_probability(
                        step, total, execution_cfg))
                loss = model.loss(x, y, mode=execution_mask)
                execution_metrics = {
                    "parallel_fraction": torch.tensor(
                        execution_mask.count("parallel") / mcfg.n_layer,
                        device=loss.device)
                }
            elif execution_cfg.get("enabled", False):
                gradnorm_rho = execution_cfg.get("gradnorm_rho")
                if execution_cfg.get("consistency_loss_target") is not None:
                    current_consistency_weight = losstarget_cw(execution_cfg)
                elif gradnorm_rho is not None:
                    current_consistency_weight = _gradnorm_cw(
                        model, x, y, execution_cfg, gradnorm_rho, step=step)
                else:
                    current_consistency_weight = consistency_weight(
                        step, total, execution_cfg)
                mem_efficient = execution_cfg.get("memory_efficient", False)
                loss, execution_metrics = polymorphic_loss(
                    model, x, y,
                    execution_cfg.get("parallel_weight", 0.5),
                    current_consistency_weight,
                    execution_cfg.get("teacher_detach", False),
                    memory_efficient=mem_efficient,
                    consistency_type=execution_cfg.get(
                        "consistency_type", "centered_mse"),
                    temperature=execution_cfg.get(
                        "consistency_temperature", 1.0),
                    gradient_checkpointing=execution_cfg.get(
                        "gradient_checkpointing", False))
                execution_metrics["consistency_weight"] = torch.tensor(
                    current_consistency_weight, device=loss.device)
                losstarget_update(
                    execution_metrics["consistency"].detach().item(),
                    execution_cfg, step, total)
                execution_metrics.update(losstarget_diag(loss.device))
            else:
                loss = model.loss(x, y)
                execution_metrics = None
        if not (execution_cfg.get("enabled", False)
                and execution_cfg.get("memory_efficient", False)):
            loss.backward()
        guard_record = None
        if guard_items and step % guard_cfg.get("every", 1) == 0:
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=(device != "cpu")):
                margin = forced_choice_margin(
                    model, lambda text: tokenizer.encode(text).ids,
                    guard_items, device)
            parameters = [parameter for parameter in model.parameters()
                          if parameter.requires_grad]
            margin_gradients = torch.autograd.grad(
                margin, parameters, allow_unused=True)
            if margin.item() < guard_cfg.get("trigger_margin", 0.0):
                guard_record = project_gradient_(
                    parameters, margin_gradients,
                    max_dot=guard_cfg.get("max_dot", 0.0))
            else:
                guard_record = {"projected": False}
            guard_record["margin"] = margin.item()
        muon.step(); adamw.step()
        model.zero_grad(set_to_none=True)

        if step % 20 == 0:
            rec = {"step": step, "loss": loss.item(),
                   "tok_s": cfg["batch_seqs"] * mcfg.ctx_len * max(step - resume_step, 1)
                            / (time.time() - t0)}
            if guard_record is not None:
                rec.update({f"guard_{key}": value
                            for key, value in guard_record.items()})
            if execution_metrics is not None:
                rec.update({f"execution_{key}": value.item()
                            for key, value in execution_metrics.items()})
            train_log.write(json.dumps(rec) + "\n"); train_log.flush()
            tracker.log_step(step, rec, execution_metrics, guard_record,
                             muon_lr=muon.param_groups[0]["lr"],
                             adamw_lr=adamw.param_groups[0]["lr"])
            print(rec)

    tracker.finish()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
