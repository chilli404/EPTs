"""6-mode compiler: combine a hardware-INDEPENDENT accuracy cost table with a
hardware-SPECIFIC latency cost table to pick the actual best mask per GPU,
instead of hoping a mask found by random sampling on one GPU transfers to
another (it does not -- see F61: Blackwell shows ~1.0x for binary/parallel
where MPS/L40S may not).

Two things are measured, and only one needs to be re-measured per hardware:

  1. Accuracy cost per (layer, mode): reused directly from an existing
     eval_6mode_masks.py output's `single_layer_probes` -- this is a
     property of the model weights (delta_bpb from perturbing one layer),
     not the GPU, so measuring it once is enough.
  2. Latency cost per (layer, mode): measured fresh here, on whatever
     hardware this script runs on. This is exactly the axis that differs
     across Blackwell/MPS/L40S.

Then `compile_multichoice_dp` (fogen.execution_graph) selects, for each of
several quality budgets, the per-layer mode assignment maximizing predicted
latency saving -- the N-ary generalization of the binary defect-budget
compiler already used for ternary/binary. The picked masks are then
VALIDATED by actually running the model under them and measuring real
bpb/latency, the same discipline eval_graph_rewrites.py uses for binary.

Usage:
  python scripts/eval_6mode_compiler.py \
      --ckpt .../1b_6mode_lt_sym/ckpts/step012000.safetensors \
      --config configs/scale1b_6mode_losstarget.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --probes_json blackwell/results/6mode_masks_1b_lt_sym.json \
      --budgets 0.0005,0.002,0.01,0.03 \
      --output blackwell/results/6mode_compiler_1b_lt_sym_<hw>.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import load_tokenizer
from fogen.evals.bpb import evaluate_bpb, token_byte_table, val_stream
from fogen.execution_graph import (
    compile_iterative_dp,
    compile_multichoice_dp,
    fit_additive_composition,
    interpolate_bracketed_baseline,
)
from fogen.model import GPT, ModelConfig
from scipy.optimize import minimize as _minimize

SIX_MODE = ["parallel", "skip", "reverse", "attn_only", "ffn_only"]
TERNARY_MODES = ["parallel", "skip"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_forward(model, x, mode, warmup=None, repeats=15):
    # MPS compiles a distinct Metal graph per novel mode/mask shape on first
    # use; the compile stall showed up as one-off latency spikes of
    # hundreds of ms on layers whose mode hadn't been exercised yet (observed:
    # 190ms baseline, +438ms on a single-layer switch) that 3 warmup calls did
    # not absorb. CUDA does not need this; give MPS much more warmup.
    if warmup is None:
        warmup = 20 if x.device.type == "mps" else 3
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mode=mode)
        _sync(x.device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x, mode=mode)
            _sync(x.device)
    return (time.perf_counter() - t0) / repeats


def measure_single_layer_latency(model, sample, n_layers, baseline_ms, alt_modes,
                                 bracket_every=4):
    """Per-(layer, mode) latency cost table, this hardware only.

    Re-measures the sequential baseline every `bracket_every` layers and
    interpolates (`interpolate_bracketed_baseline`) rather than using one
    fixed upfront value. A locked GPU clock (`nvidia-smi -lgc`) does NOT
    eliminate this -- real finding, this session: after locking the clock,
    a sharp cost discontinuity still appeared mid-sweep, but at a DIFFERENT
    layer position across two separate runs of the same checkpoint (layer
    1->2 in one run, layer 10->11 in another) -- a fixed architectural
    effect would appear at the same layer every time, so this is a
    time-based settling/drift artifact independent of clock lock, not
    thermal throttling specifically. Bracketing was removed once
    (mistakenly, on the assumption clock-lock alone would fix it) and is
    restored here.
    """
    brackets = [(0, baseline_ms)]
    costs = []
    for layer in range(n_layers):
        row = {}
        for mode in alt_modes:
            mask = ["sequential"] * n_layers
            mask[layer] = mode
            t = time_forward(model, sample, mask) * 1000
            local_baseline = interpolate_bracketed_baseline(brackets, layer)
            row[mode] = t - local_baseline
        costs.append(row)
        print(f"  layer {layer:>2}/{n_layers}: "
              + " ".join(f"{m}={v:+.3f}ms" for m, v in row.items()), flush=True)
        if (layer + 1) % bracket_every == 0 and layer + 1 < n_layers:
            fresh_baseline = time_forward(model, sample, "sequential") * 1000
            brackets.append((layer + 1, fresh_baseline))
            print(f"    [bracket] re-measured baseline at layer {layer+1}: "
                  f"{fresh_baseline:.3f}ms (was {baseline_ms:.3f}ms)", flush=True)
    return costs


def load_accuracy_costs(probes_json, n_layers, alt_modes, target_field,
                        source="ridge"):
    """Hardware-independent per-(layer, mode) accuracy cost table.

    Works for any alt_modes/target_field, not just 6-mode's -- e.g.
    alt_modes=["parallel","skip"], target_field="ternary_degradation" for a
    ternary eval_ternary_masks.py output.

    `source="ridge"` (default): refit the additive composition law
    (`fit_additive_composition`, fogen.execution_graph) directly from the
    `rows` (mask, target) already saved in probes_json -- pure CPU, no GPU
    time, reuses data already on disk. Verified against 6-mode/430M: mean
    |error| is roughly HALF that of the single-layer-probe sum, and
    critically the probe sum gets the *sign* wrong across most of the
    measured perturbation-count range (predicts positive/costly where actual
    is negative/improving) while the ridge fit does not. Use `source="probe"`
    only for comparison against the known-worse baseline.

    Caveat, not yet resolved: the calibration data (`rows`) may cover a
    perturbation-count range that does not include the compiler's actual
    operating points -- this is linear-model extrapolation, not verified
    accurate outside the fitted range. Always validate the compiled mask
    against real measurements (this script already does).
    """
    d = json.load(open(probes_json))
    if source == "probe":
        probes = d.get("single_layer_probes")
        if probes is None:
            raise ValueError(f"{probes_json} has no single_layer_probes")
        costs = [{} for _ in range(n_layers)]
        for p in probes:
            costs[p["layer"]][p["mode"]] = p[target_field]
        return costs

    rows = d.get("rows")
    if not rows:
        raise ValueError(f"{probes_json} has no rows -- cannot refit ridge costs; "
                         "run the appropriate mask-eval script on this checkpoint first")

    masks = [r["mask"] for r in rows]
    targets = [r[target_field] for r in rows]
    coef = fit_additive_composition(masks, targets, n_layers, alt_modes)
    costs = [{} for _ in range(n_layers)]
    for layer in range(n_layers):
        for j, mode in enumerate(alt_modes):
            costs[layer][mode] = float(coef[layer * len(alt_modes) + j])
    return costs


def _dist_bin(d):
    return 0 if d == 1 else (1 if d <= 4 else 2)


def fit_m2m3(rows, n_layers, alt_modes, target_field):
    """Fit the M0 base additive coefficients (same as load_accuracy_costs'
    ridge path), then mode-specific attenuation (M2: alpha_k, beta_k per
    mode) and a structural pairwise-distance interaction correction (M3:
    gamma) on top -- validated (F73/F74/F75) to roughly halve the compiler's
    real prediction error on its own selected masks for primary architecture
    and ternary; genuinely mixed for Llama (helps a lot at 430M, slightly
    hurts at 120M) -- do not assume this transfers to Llama without checking.
    """
    masks = [r["mask"] for r in rows]
    targets = np.array([r[target_field] for r in rows])
    coef = fit_additive_composition(masks, targets, n_layers, alt_modes)
    base_costs = {(l, alt_modes[j]): float(coef[l * len(alt_modes) + j])
                  for l in range(n_layers) for j in range(len(alt_modes))}

    def mode_counts(ms):
        counts = {k: np.zeros(len(ms)) for k in alt_modes}
        for i, m in enumerate(ms):
            for mode in m:
                if mode in counts:
                    counts[mode][i] += 1
        return counts

    def mode_subsums(ms):
        Ak = {}
        for k_idx, k in enumerate(alt_modes):
            vals = [sum(coef[l * len(alt_modes) + k_idx] for l, mode in enumerate(m) if mode == k)
                    for m in ms]
            Ak[k] = np.array(vals)
        return Ak

    counts = mode_counts(masks)
    Ak = mode_subsums(masks)

    def loss(params):
        pred = np.zeros(len(masks))
        for i, k in enumerate(alt_modes):
            a, b = params[2 * i], params[2 * i + 1]
            pred += a * Ak[k] * np.power(np.maximum(counts[k], 1.0), -b)
        return np.mean((targets - pred) ** 2)

    res = _minimize(loss, x0=np.array([1.0, 0.0] * len(alt_modes)),
                    method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-6})
    alpha = {k: float(res.x[2 * i]) for i, k in enumerate(alt_modes)}
    beta = {k: float(res.x[2 * i + 1]) for i, k in enumerate(alt_modes)}

    m2_pred = np.zeros(len(masks))
    for k in alt_modes:
        m2_pred += alpha[k] * Ak[k] * np.power(np.maximum(counts[k], 1.0), -beta[k])
    resid = targets - m2_pred

    # canonical key order must match execution_graph.py's evaluate_true_cost /
    # _marginal_m3, which sort the mode-pair alphabetically (not by alt_modes
    # index) -- this was a real bug caught by a local dry run before touching
    # the rig (KeyError on a pair whose index-order and alpha-order disagreed).
    pair_keys = sorted({tuple(sorted((k, j))) for k in alt_modes for j in alt_modes})
    key_idx = {(k, j, r): i * 3 + r for i, (k, j) in enumerate(pair_keys) for r in range(3)}
    X = np.zeros((len(masks), len(pair_keys) * 3))
    for row, m in enumerate(masks):
        positions = [(l, mode) for l, mode in enumerate(m) if mode != "sequential"]
        for a in range(len(positions)):
            for b in range(a + 1, len(positions)):
                li, mi = positions[a]
                lj, mj = positions[b]
                k, j = tuple(sorted((mi, mj)))
                X[row, key_idx[(k, j, _dist_bin(abs(li - lj)))]] += 1.0
    A = X.T @ X + 1.0 * np.eye(X.shape[1])
    gamma_flat = np.linalg.solve(A, X.T @ resid)
    gamma = {(k, j, r): float(gamma_flat[key_idx[(k, j, r)]]) for (k, j) in pair_keys for r in range(3)}

    # Starvation term: real finding (F80's follow-up, refined further this
    # session) -- masks that leave ZERO "safe" layers (sequential + mild
    # alt-modes) collapse catastrophically regardless of which aggressive
    # alt-modes fill the rest, a threshold M2's per-mode-independent terms
    # cannot represent. Fit gamma_seq*(n_safe+1)^-delta_seq on the M2+M3
    # residual (only informative if the calibration data actually contains
    # masks with few safe layers -- if not, this fits near-zero, a harmless
    # no-op, not a fabricated effect).
    m3_pred = X @ gamma_flat
    resid2 = resid - m3_pred
    # n_safe = sequential + mild alt-modes (parallel/reverse) -- generalizes
    # n_sequential to also count non-FLOP-reducing alt-modes as "safe" for
    # starvation purposes (see evaluate_true_cost docstring: an all-parallel
    # rewrite of the 1B specialist degraded LESS than partial skip/attn_only
    # masks, so raw n_sequential alone can't distinguish mild-heavy from
    # aggressive-heavy masks at the same n_sequential).
    from fogen.execution_graph import AGGRESSIVE_MODES
    n_safe = np.array([sum(1 for mo in m if mo not in AGGRESSIVE_MODES) for m in masks],
                      dtype=float)

    def seq_loss(params):
        g, d = params
        pred = g * np.power(n_safe + 1.0, -d)
        return float(np.mean((resid2 - pred) ** 2))

    # Bounded L-BFGS-B, not unconstrained Nelder-Mead -- real bug found and
    # fixed this session: an unbounded fit degenerated to delta_seq=49.09, a
    # step function that is 0.0215 at n_sequential=0 and EXACTLY ZERO for
    # every n_sequential>=1 (verified: (1+1)^-49 rounds to 0 in float64).
    # That "fit" perfectly matched the handful of n_seq=0 training points
    # while providing zero protection at n_seq=1-2, which F80/F83 already
    # established is still a genuinely risky region -- caused a real search
    # failure (0/8 real-feasible candidates) that only real GPU validation
    # caught. delta_seq=15 already gives (2)^-15 ~ 3e-5, i.e. correctly
    # near-zero one step away from the origin without the discontinuity.
    seq_res = _minimize(seq_loss, x0=np.array([0.0, 1.0]), method="L-BFGS-B",
                        bounds=[(0.0, None), (0.0, 15.0)])
    gamma_seq, delta_seq = float(seq_res.x[0]), float(seq_res.x[1])

    print(f"  M2 fit: " + ", ".join(f"{k}=(a={alpha[k]:.2f},b={beta[k]:.2f})" for k in alt_modes))
    print(f"  Sequential-starvation fit: gamma_seq={gamma_seq:.4f} delta_seq={delta_seq:.4f}")
    return base_costs, alpha, beta, gamma, (gamma_seq, delta_seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--probes_json", required=True,
                        help="existing mask-eval output for this checkpoint "
                             "(eval_6mode_masks.py or eval_ternary_masks.py), "
                             "supplies hardware-independent accuracy costs")
    parser.add_argument("--alt_modes", default=",".join(SIX_MODE),
                        help="comma-separated alternative modes per layer, "
                             "excluding sequential. Default is the 6-mode "
                             "set; pass 'parallel,skip' for ternary.")
    parser.add_argument("--target_field", default="delta_bpb",
                        help="field in probes_json's rows to fit against -- "
                             "'delta_bpb' for eval_6mode_masks.py output, "
                             "'ternary_degradation' for eval_ternary_masks.py")
    parser.add_argument("--cost_source", choices=["ridge", "probe"], default="ridge",
                        help="ridge: refit the composition law from rows "
                             "(better calibrated, default); probe: raw "
                             "single-layer sums (known worse, sign errors)")
    parser.add_argument("--cost_model", choices=["additive", "m2m3"], default="additive",
                        help="additive: current production law (M0), "
                             "compile_multichoice_dp (default). m2m3: "
                             "mode-specific attenuation + structural "
                             "interaction (F73/F74/F75), compile_iterative_dp. "
                             "m2m3 requires --cost_source ridge (ignores "
                             "--cost_source probe).")
    parser.add_argument("--budgets", default="0.0005,0.002,0.01,0.03")
    parser.add_argument("--reference_bpb", type=float, default=None,
                        help="gate the budget against a DIFFERENT "
                             "checkpoint's (the specialist's) real BPB "
                             "instead of this checkpoint's own sequential "
                             "BPB -- same fix as C28 in eval_6mode_blackbox_"
                             "search.py/eval_6mode_active_loop.py, applied "
                             "here since this script had the same "
                             "own-sequential-relative gating bug.")
    parser.add_argument("--max_free", type=int, default=None,
                        help="cap on simultaneous unconditional free picks "
                             "(see compile_multichoice_dp docstring -- "
                             "single-layer costs do not sum linearly once "
                             "many are stacked; leave unset only to "
                             "reproduce the known-broken unbounded behavior)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_windows", type=int, default=64)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device, hw = torch.device("cuda"), "cuda"
    elif torch.backends.mps.is_available():
        device, hw = torch.device("mps"), "mps"
    else:
        device, hw = torch.device("cpu"), "cpu"

    alt_modes = [m.strip() for m in args.alt_modes.split(",") if m.strip()]

    cfg = yaml.safe_load(open(args.config))
    mcfg = ModelConfig(**cfg["model"])
    n_layers = mcfg.n_layer

    model = GPT(mcfg).to(device)
    state = {k: v.float() for k, v in load_file(args.ckpt).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(k.startswith("rope_") for k in missing)
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer_dir)
    stream = val_stream(args.val_shards)
    byte_table = token_byte_table(tokenizer)
    context = mcfg.ctx_len
    sample = torch.tensor(
        np.asarray(stream[:2 * context], dtype=np.int64).reshape(2, context),
        device=device)

    print(f"Hardware: {hw}")
    if hw in ("mps", "cuda"):
        # Steady-state throughput takes time to settle -- MPS needs ~4 calls
        # (per-mode JIT compile), CUDA needs sustained load for boost clocks
        # to ramp up (found live on a real 1B run: EVERY mode's single-layer
        # latency delta drifted smoothly upward for the first ~10 of 28
        # layers then plateaued -- classic clock-ramp shape, not a real
        # per-layer property, since it hit every mode identically regardless
        # of what that mode does). Warm the whole process up ONCE, before the
        # baseline itself is measured, so baseline and every subsequent probe
        # are on the same footing.
        print(f"Warming up {hw} to steady state before measuring anything...")
        # Warm every mode, not just sequential -- each is a genuinely
        # different branch in Block.forward (parallel fuses attn+mlp, skip
        # and attn_only/ffn_only take early-return paths), and the first
        # touch of an untouched branch showed up as a residual bias on
        # whichever layer happened to exercise it first in the sweep.
        n_warmup = 15 if hw == "mps" else 40
        for mode in ["sequential"] + alt_modes:
            for _ in range(n_warmup):
                with torch.no_grad():
                    model(sample, mode=[mode] * n_layers)
        _sync(device)
    baseline_ms = time_forward(model, sample, "sequential") * 1000
    print(f"Baseline (all-sequential): {baseline_ms:.3f}ms")

    print("Measuring per-(layer, mode) latency on this hardware...")
    latency_costs = measure_single_layer_latency(model, sample, n_layers,
                                                  baseline_ms, alt_modes)

    m2m3_params = None
    if args.cost_model == "m2m3":
        print("Fitting M2 (mode-specific attenuation) + M3 (structural "
              "interaction) cost model...")
        rows = json.load(open(args.probes_json)).get("rows")
        if not rows:
            raise ValueError(f"{args.probes_json} has no rows -- m2m3 needs raw masks")
        base_costs, alpha, beta, gamma, seq_penalty = fit_m2m3(
            rows, n_layers, alt_modes, args.target_field)
        benefits = {(l, m): -latency_costs[l][m] for l in range(n_layers) for m in alt_modes
                    if m in latency_costs[l]}
        m2m3_params = (base_costs, benefits, alpha, beta, gamma, seq_penalty)
    else:
        print("Loading per-(layer, mode) accuracy costs (hardware-independent)...")
        accuracy_costs = load_accuracy_costs(args.probes_json, n_layers, alt_modes,
                                             args.target_field, source=args.cost_source)

        layer_options = []
        for layer in range(n_layers):
            opts = {"sequential": (0.0, 0.0)}
            for mode in alt_modes:
                acc_cost = accuracy_costs[layer].get(mode)
                lat_delta = latency_costs[layer].get(mode)
                if acc_cost is None or lat_delta is None:
                    continue
                benefit = -lat_delta  # negative latency delta = time saved
                opts[mode] = (acc_cost, benefit)
            layer_options.append(opts)

    # Ascending order is required for the extra_candidates monotonicity fix
    # below (a smaller budget's winning mask is always feasible at a larger
    # budget too, so it must be checked, not just found earlier).
    budgets = sorted(float(b) for b in args.budgets.split(","))
    prior_masks = []
    compiled = []
    with torch.no_grad():
        sequential_logits = model(sample, mode="sequential").float()
    sequential_logprob = F.log_softmax(sequential_logits, dim=-1)
    sequential_probability = sequential_logprob.exp()
    model.cfg.execution_mode = "sequential"
    seq_bpb = evaluate_bpb(model, stream, byte_table, context, batch_size=32,
                           max_windows=args.max_windows, device=str(device))["val_bpb"]

    # Refresh the baseline right before validating any budget -- real bug
    # found by direct inspection (this session): comparing each budget's
    # freshly-measured actual_latency_ms against the ORIGINAL baseline_ms
    # (measured before the single-layer sweep and seq_bpb eval, both of
    # which take real wall-clock time under sustained load) understated the
    # all-sequential case's own speedup by ~3.6-3.8% -- the same GPU
    # thermal drift as C21, just showing up here instead of in the sweep.
    baseline_ms = time_forward(model, sample, "sequential") * 1000
    print(f"Refreshed baseline right before budget validation: {baseline_ms:.3f}ms")

    for budget in budgets:
        # Gate against the specialist's real BPB, not this checkpoint's own
        # sequential BPB, when --reference_bpb is given (same fix as C28):
        # want actual_bpb - reference_bpb <= budget, i.e.
        # seq_bpb + degradation - reference_bpb <= budget, i.e.
        # degradation <= budget - seq_bpb + reference_bpb.
        effective_budget = (budget - seq_bpb + args.reference_bpb
                           if args.reference_bpb is not None else budget)
        if args.cost_model == "m2m3":
            base_costs, benefits, alpha, beta, gamma, seq_penalty = m2m3_params
            result = compile_iterative_dp(base_costs, benefits, alt_modes, alpha, beta,
                                          gamma, effective_budget, n_layers, resolution=2000,
                                          max_free=args.max_free, extra_candidates=prior_masks,
                                          seq_penalty=seq_penalty)
            mask = result["mask"]
            prior_masks.append(mask)
        else:
            result = compile_multichoice_dp(layer_options, effective_budget, resolution=2000,
                                            max_free=args.max_free)
            mask = result["choices"]
        model.cfg.execution_mode = mask
        actual_bpb = evaluate_bpb(model, stream, byte_table, context, batch_size=32,
                                  max_windows=args.max_windows,
                                  device=str(device))["val_bpb"]
        actual_latency_ms = time_forward(model, sample, mask) * 1000
        with torch.no_grad():
            logits = model(sample, mode=mask).float()
        logprob = F.log_softmax(logits, dim=-1)
        symmetric_kl = float((
            F.kl_div(logprob, sequential_probability, reduction="batchmean")
            + F.kl_div(sequential_logprob, logprob.exp(), reduction="batchmean")
        ) / 2)
        agree = float((logits.argmax(dim=-1) == sequential_logits.argmax(dim=-1))
                     .float().mean())
        row = {
            "budget": budget,
            "effective_budget": effective_budget,
            "reference_bpb": args.reference_bpb,
            "mask": mask,
            "mode_counts": {m: mask.count(m) for m in ["sequential"] + alt_modes},
            "predicted_cost": result["predicted_cost"],
            "predicted_benefit_ms": result["predicted_benefit"],
            "actual_bpb": actual_bpb,
            "actual_bpb_degradation": actual_bpb - seq_bpb,
            "actual_latency_ms": actual_latency_ms,
            "actual_speedup": baseline_ms / actual_latency_ms,
            "argmax_agreement": agree,
            "symmetric_kl": symmetric_kl,
        }
        compiled.append(row)
        print(f"  budget={budget:<8} -> speedup={row['actual_speedup']:.3f}x "
              f"bpb={actual_bpb:.4f} agree={agree:.4f}", flush=True)

    output = {
        "checkpoint": args.ckpt,
        "hardware": hw,
        "max_free": args.max_free,
        "cost_source": args.cost_source,
        "cost_model": args.cost_model,
        "alt_modes": alt_modes,
        "target_field": args.target_field,
        "n_layers": n_layers,
        "baseline_ms": baseline_ms,
        "seq_bpb": seq_bpb,
        "latency_costs": latency_costs,
        "compiled": compiled,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
