"""Black-box local search directly on real measurements -- no composition
law, no predicted cost, no fitted parameters at all. Tests whether the
composition-law-guided compiler (eval_6mode_compiler.py) is actually
limiting what configurations get found, or whether it's already close to
what's achievable.

Method: hill-climbing. Start from a random mask, repeatedly try K random
single-layer mode-swap neighbors, measure each for REAL (val_bpb + latency,
same evaluate_bpb/time_forward as everywhere else in this project), move to
whichever neighbor most improves speedup subject to agreement staying above
a target floor (measured directly, never predicted). No model of any kind
mediates the decision -- every step is a real GPU measurement.

Usage:
  python scripts/eval_6mode_blackbox_search.py \
      --ckpt .../430m_6mode_lt/ckpts/step012000.safetensors \
      --config configs/scale430m_6mode_losstarget.yaml \
      --val_shards data/climbmix/bpe8192/shards --tokenizer_dir data/climbmix/bpe8192 \
      --agreement_floor 0.75 --n_restarts 3 --steps_per_restart 15 \
      --output blackwell/results/6mode_blackbox_430m.json

Optional --surrogate_filter mode: keeps every accept/reject decision a REAL
GPU measurement (unchanged), but proposes a larger pool of random neighbor
mutations per step, ranks the whole pool with a cheap fitted M2+M3 cost
model (no GPU needed for the ranking itself), and only real-measures the
top --neighbors_per_step of them -- same real-eval budget as plain
hill-climbing, spent on more promising candidates. Off by default; exact
current behavior is preserved when the flag is not passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import load_tokenizer
from fogen.evals.bpb import evaluate_bpb, token_byte_table, val_stream
from fogen.execution_graph import evaluate_true_cost
from fogen.model import GPT, ModelConfig

ALL_MODES = ["sequential", "parallel", "skip", "reverse", "attn_only", "ffn_only"]
MODES = ALL_MODES

# Real per-layer marginal latency BENEFIT (ms saved per layer if that layer
# uses this mode instead of sequential), from this session's real
# measurements (same table sa_6mode_compiler.py uses) -- used ONLY to rank
# --surrogate_filter candidate pools relative to each other, never trusted
# as an absolute prediction fed back into any accept/reject decision (every
# accept/reject in this script remains a REAL GPU measurement).
PER_LAYER_BENEFIT_MS = {
    "skip": 2.85,
    "attn_only": 1.48,
    "parallel": 0.03,
    "ffn_only": 0.03,
    "reverse": -1.33,
}


def _surrogate_benefit(mask):
    return sum(PER_LAYER_BENEFIT_MS.get(m, 0.0) for m in mask)


def load_surrogate_training_rows(uniform_json, structured_jsons):
    """Load the uniform-random corpus plus structured (hill-climb/greedy)
    real-eval rows used to fit the cheap M2+M3 surrogate cost model --
    exact same sources and field-name normalization as sa_6mode_compiler.py
    established this session ('mask' + 'delta_bpb' for uniform corpus rows,
    'mask' + 'bpb_degradation' -> renamed to 'delta_bpb' for structured
    all_evals rows)."""
    uniform_rows = json.load(open(uniform_json))["rows"]
    uniform = [{"mask": r["mask"], "delta_bpb": r["delta_bpb"]} for r in uniform_rows]
    structured = []
    for path in structured_jsons:
        d = json.load(open(path))
        for r in d["all_evals"]:
            structured.append({"mask": r["mask"], "delta_bpb": r["bpb_degradation"]})
    return uniform, structured


def fit_surrogate_model(uniform_json, structured_jsons, n_layers, alt_modes):
    """Fit the cheap M2+M3 cost model (fit_m2m3, uniform + structured*40
    upweighted -- the best-known enriched fit established this session) once,
    up front. Returns a callable predicted_cost(mask) -> predicted delta_bpb.
    CPU-only, no GPU measurement -- this is exactly the model whose ABSOLUTE
    point estimates are known (this session) to be poorly calibrated on
    extreme/search-optimized masks; it is used below ONLY to rank candidates
    relative to each other, never as an accept/reject threshold."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_6mode_compiler import fit_m2m3  # noqa: E402

    uniform, structured = load_surrogate_training_rows(uniform_json, structured_jsons)
    combined = uniform + structured * 40
    print(f"[surrogate] fitting on {len(uniform)} uniform + {len(structured)}*40="
          f"{len(structured) * 40} structured (upweighted) = {len(combined)} rows", flush=True)
    base_costs, alpha, beta, gamma, seq_penalty = fit_m2m3(combined, n_layers, alt_modes, "delta_bpb")

    def predicted_cost(mask):
        return evaluate_true_cost(mask, base_costs, alt_modes, alpha, beta, gamma,
                                  seq_penalty=seq_penalty)

    return predicted_cost


def fit_quantile_surrogate_model(uniform_json, structured_jsons, n_layers, alt_modes):
    """Alternative surrogate: median (q=0.5) linear quantile regression on
    the same plain one-hot (layer, alt_mode) feature basis as
    fit_additive_composition -- found this session (eval_quantile_regression_6mode.py)
    to have MUCH better-calibrated ABSOLUTE accuracy predictions for a known
    held-out search-optimized mask than M2M3 (predicted 0.0500 vs real
    0.0498, essentially exact, vs M2M3's 30-40%+ overestimate) -- though a
    later check found this specific calibration didn't generalize to a
    NEW mask picked by a DP built on it (real 0.0631 vs predicted 0.0497).
    Still worth testing as an alternative RANKING signal here, since good
    calibration on at least some points is a different question from
    whether it ranks candidates any better than M2M3 does."""
    from sklearn.linear_model import QuantileRegressor

    uniform, structured = load_surrogate_training_rows(uniform_json, structured_jsons)
    combined = uniform + structured
    print(f"[surrogate-quantile] fitting median quantile regression on "
          f"{len(uniform)} uniform + {len(structured)} structured = "
          f"{len(combined)} rows", flush=True)

    def onehot(mask):
        v = [0.0] * (n_layers * len(alt_modes))
        for l, m in enumerate(mask):
            if m in alt_modes:
                v[l * len(alt_modes) + alt_modes.index(m)] = 1.0
        return v

    X = np.array([onehot(r["mask"]) for r in combined])
    y = np.array([r["delta_bpb"] for r in combined])
    model = QuantileRegressor(quantile=0.5, alpha=1e-4, solver="highs")
    model.fit(X, y)

    def predicted_cost(mask):
        return float(model.intercept_ + model.coef_ @ onehot(mask))

    return predicted_cost


def compute_mode_bias_weights(history_jsons, modes, top_frac=0.1, floor=0.05):
    """Empirical mode-sampling weights from real prior search results (not a
    fitted model): pool every measured mask across `history_jsons`, take the
    top `top_frac` by real bpb_degradation, and weight each mode by its
    frequency in that top slice relative to its frequency in the full pool
    (a simple over/under-representation ratio, floored at `floor` so no
    mode's proposal probability drops to exactly zero -- real finding this
    session: the best masks found were ~80% sequential/14% parallel/6%
    reverse with ~0% skip/attn_only/ffn_only, so this concretely
    deprioritizes the FLOP-reducing modes for a quality-improvement search
    while still allowing occasional exploration of them.
    """
    pool = []
    for path in history_jsons:
        d = json.load(open(path))
        pool.extend(d["all_evals"])
    if not pool:
        return {m: 1.0 for m in modes}
    pool_sorted = sorted(pool, key=lambda r: r["bpb_degradation"])
    top_n = max(1, int(top_frac * len(pool_sorted)))
    top = pool_sorted[:top_n]

    def freq(rows):
        counts = {m: 0 for m in modes}
        total = 0
        for r in rows:
            for m in r["mask"]:
                if m in counts:
                    counts[m] += 1
                total += 1
        return {m: counts[m] / max(total, 1) for m in modes}

    top_freq = freq(top)
    pool_freq = freq(pool_sorted)
    weights = {}
    for m in modes:
        ratio = top_freq[m] / max(pool_freq[m], 1e-6)
        weights[m] = max(ratio, floor)
    total_w = sum(weights.values())
    weights = {m: w / total_w for m, w in weights.items()}
    print(f"[bias] mode weights from {len(pool)} pooled real evals "
          f"(top {top_frac:.0%}={top_n}): "
          + ", ".join(f"{m}={w:.3f}" for m, w in weights.items()), flush=True)
    return weights


def surrogate_rank_score(mask, predicted_cost, budget, objective, penalty_weight=50.0):
    """Relative ranking score for one candidate mask under the cheap
    surrogate, used only to pick which of a larger candidate pool are worth
    spending a REAL GPU measurement on -- never used to accept/reject.
    'quality' objective: rank by most-negative predicted cost (biggest
    predicted quality IMPROVEMENT). 'speedup' objective (or no budget given):
    rank by predicted-benefit-per-unit-predicted-cost via the same
    Lagrangian/penalty form sa_6mode_compiler.py validated this session --
    reward predicted latency benefit, quadratically penalize predicted
    overshoot past the budget, no reward/penalty for being under budget."""
    pred_cost = predicted_cost(mask)
    if objective == "quality":
        return -pred_cost
    pred_benefit = _surrogate_benefit(mask)
    if budget is None:
        return pred_benefit
    over = max(0.0, pred_cost - budget)
    return pred_benefit - penalty_weight * over * over


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_forward(model, x, mode, warmup=3, repeats=10):
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mode=mode)
        _sync(x.device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x, mode=mode)
            _sync(x.device)
    return (time.perf_counter() - t0) / repeats


def measure_real(model, sample, mask, stream, byte_table, ctx_len, max_windows,
                 device, sequential_logits, sequential_probability_argmax, seq_bpb,
                 reference_bpb=None):
    model.cfg.execution_mode = mask
    bpb = evaluate_bpb(model, stream, byte_table, ctx_len, batch_size=32,
                       max_windows=max_windows, device=str(device))["val_bpb"]
    latency = time_forward(model, sample, mask) * 1000
    with torch.no_grad():
        logits = model(sample, mode=mask).float()
    agree = float((logits.argmax(dim=-1) == sequential_probability_argmax).float().mean())
    result = {"mask": list(mask), "val_bpb": bpb, "bpb_degradation": bpb - seq_bpb,
            "latency_ms": latency, "argmax_agreement": agree}
    if reference_bpb is not None:
        result["bpb_degradation_vs_reference"] = bpb - reference_bpb
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--agreement_floor", type=float, default=None,
                        help="feasibility gate on argmax_agreement (matches "
                             "sequential's exact token choices -- a proxy, "
                             "not an objective quality measure: a mask can "
                             "disagree with sequential while predicting the "
                             "real text equally well or better)")
    parser.add_argument("--bpb_floor", type=float, default=None,
                        help="feasibility gate on bpb_degradation (bits-per-"
                             "byte on held-out real text -- the objective "
                             "language-modeling performance metric, doesn't "
                             "depend on matching sequential's specific "
                             "predictions). Prefer this over --agreement_floor.")
    parser.add_argument("--n_restarts", type=int, default=3)
    parser.add_argument("--steps_per_restart", type=int, default=15)
    parser.add_argument("--neighbors_per_step", type=int, default=4)
    parser.add_argument("--stagnation_patience", type=int, default=5,
                        help="break a restart early if no improvement for "
                             "this many consecutive steps, and use the "
                             "freed-up budget to start a fresh restart "
                             "instead of repeating an unbeatable point "
                             "(real behavior observed: 21 straight identical "
                             "steps in one restart, wasting most of its "
                             "budget on a local optimum)")
    parser.add_argument("--max_windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--seed_mask_from", default=None,
                        help="path to an eval_6mode_compiler.py output JSON -- "
                             "seed restart 0 from its best (highest-speedup) "
                             "compiled mask instead of a random start, to "
                             "test how much a short refinement closes the "
                             "gap versus a full random-start search")
    parser.add_argument("--seed_mask_budget", type=float, default=None,
                        help="which budget's mask to seed from, if "
                             "--seed_mask_from is set (default: highest "
                             "actual_speedup among all its compiled rows)")
    parser.add_argument("--objective", choices=["speedup", "quality"], default="speedup",
                        help="'speedup' (default): maximize speed subject to "
                             "the feasibility floor -- correct for finding "
                             "fast-but-acceptable configs. 'quality': "
                             "maximize quality IMPROVEMENT (most negative "
                             "bpb_degradation) instead -- required for a "
                             "genuine BPB-improvement search, since "
                             "'speedup' with bpb_floor=0.0 just finds the "
                             "FASTEST mask that barely clears feasibility "
                             "(e.g. bpb_degradation=-0.0001), not the "
                             "mask that improves quality the most. Real bug "
                             "found and fixed this session.")
    parser.add_argument("--seed_all_sequential", action="store_true",
                        help="seed restart 0 from the plain all-sequential "
                             "mask instead of a random start -- correct "
                             "starting point for a BPB-IMPROVEMENT search "
                             "specifically (looking for masks that beat "
                             "sequential should start AT sequential and "
                             "perturb outward, not start from an arbitrary "
                             "random mask that has no particular relationship "
                             "to the thing you're trying to improve on)")
    parser.add_argument("--reference_bpb", type=float, default=None,
                        help="real val_bpb of a DIFFERENT reference checkpoint "
                             "(e.g. the dedicated sequential specialist) -- if "
                             "given, --bpb_floor gates on degradation vs THIS "
                             "value instead of the search checkpoint's own "
                             "sequential mode.")
    parser.add_argument("--seed_from_corpus", default=None,
                        help="path to an eval_6mode_masks.py random-corpus "
                             "JSON -- seed restart 0 from its best (lowest "
                             "delta_bpb) mask instead of an arbitrary random "
                             "start. Takes priority over --seed_all_sequential "
                             "and --seed_mask_from if given.")
    parser.add_argument("--seed_mask_json", default=None,
                        help="path to a plain JSON list-of-masks file (the "
                             "format eval_specific_masks.py --masks_file "
                             "takes) -- seed restart 0 from its first mask. "
                             "For refining a heuristic-found mask (e.g. from "
                             "simulated annealing, beam search) via real "
                             "local search rather than trusting its raw, "
                             "cost-model-only output. Takes priority over "
                             "--seed_from_corpus, --seed_all_sequential, and "
                             "--seed_mask_from if given.")
    parser.add_argument("--alt_modes", default=None,
                        help="comma-separated subset of non-sequential modes "
                             "to restrict mutation/neighbor generation to "
                             "(e.g. 'skip' or 'parallel,skip'). 'sequential' "
                             "is always included implicitly. Default: all "
                             "five alternative modes (full 6-mode space).")
    parser.add_argument("--surrogate_filter", action="store_true",
                        help="instead of proposing --neighbors_per_step "
                             "purely-random single-layer mutations per step, "
                             "propose a larger pool of --surrogate_pool_size "
                             "random mutations, score ALL of them with the "
                             "cheap fitted M2+M3 cost model (no GPU needed), "
                             "and real-measure only the top "
                             "--neighbors_per_step (by predicted-benefit-per-"
                             "predicted-cost, or most-negative predicted cost "
                             "for --objective quality) -- same real-eval "
                             "budget as plain hill-climbing, spent on more "
                             "promising candidates. The model's ABSOLUTE "
                             "point estimates are not trusted (this session "
                             "established they're poorly calibrated on "
                             "extreme masks); only its RELATIVE ranking of "
                             "candidates is used, and every accept/reject "
                             "decision still comes from a REAL GPU "
                             "measurement exactly as without this flag. "
                             "Requires --bpb_floor (the surrogate predicts "
                             "delta_bpb, not argmax_agreement).")
    parser.add_argument("--bias_history_json", default=None,
                        help="comma-separated list of prior search-result "
                             "JSON files (each with an 'all_evals' list) -- "
                             "compute empirical mode frequencies among the "
                             "top decile (best real degradation) vs the "
                             "full pool, and bias future neighbor-mutation "
                             "mode sampling toward whatever modes are "
                             "over-represented in the best masks. Not a "
                             "fitted model prediction (which this session "
                             "showed doesn't reliably rank candidates) -- a "
                             "purely empirical frequency reweighting of "
                             "what has actually worked in real measurements "
                             "so far. Layer selection remains uniform "
                             "random; only the MODE choice is reweighted.")
    parser.add_argument("--surrogate_type", choices=["m2m3", "quantile"], default="m2m3",
                        help="which cheap model to use for --surrogate_filter "
                             "ranking. 'm2m3' (default): mode-attenuation + "
                             "pairwise-interaction model. 'quantile': median "
                             "linear quantile regression on the same feature "
                             "basis -- found this session to have much "
                             "better absolute calibration on at least one "
                             "known point, worth testing as a ranking "
                             "signal in its own right.")
    parser.add_argument("--surrogate_pool_size", type=int, default=40,
                        help="size of the random-mutation candidate pool "
                             "scored by the cheap surrogate per step when "
                             "--surrogate_filter is set, before filtering "
                             "down to --neighbors_per_step for real GPU "
                             "measurement. Ignored if --surrogate_filter is "
                             "not set.")
    parser.add_argument("--surrogate_uniform_json",
                        default="blackwell/results/6mode_masks_430m_gradnorm.json",
                        help="uniform-random mask corpus used to fit the "
                             "--surrogate_filter cost model (field 'rows', "
                             "each with 'mask'+'delta_bpb'). Ignored if "
                             "--surrogate_filter is not set.")
    parser.add_argument("--surrogate_structured_jsons",
                        default="blackwell/results/6mode_hillclimb_430m_gradnorm_12k_floor005.json,"
                                "blackwell/results/greedy_430m_gradnorm_12k_budget05_real.json,"
                                "blackwell/results/m2m3_greedy_430m_gradnorm_12k_budget05_real.json",
                        help="comma-separated structured (hill-climb/greedy) "
                             "real-eval JSONs used to enrich the "
                             "--surrogate_filter cost model fit (field "
                             "'all_evals', each with 'mask'+'bpb_degradation', "
                             "duplicated 40x same as sa_6mode_compiler.py's "
                             "established best fit). Ignored if "
                             "--surrogate_filter is not set.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    global MODES
    if args.alt_modes:
        alt = [m.strip() for m in args.alt_modes.split(",")]
        assert all(m in ALL_MODES for m in alt), f"invalid mode in {alt}"
        MODES = ["sequential"] + alt
    print(f"=== {Path(__file__).name} config ===")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print(f"  effective MODES: {MODES}")
    print("=" * 40, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))
    mcfg = ModelConfig(**cfg["model"])
    n_layers = mcfg.n_layer

    model = GPT(mcfg).to(device)
    state = {k: v.float() for k, v in load_file(args.ckpt).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer_dir)
    stream = val_stream(args.val_shards)
    byte_table = token_byte_table(tokenizer)
    context = mcfg.ctx_len
    sample = torch.tensor(
        np.asarray(stream[:2 * context], dtype=np.int64).reshape(2, context), device=device)

    model.cfg.execution_mode = "sequential"
    seq_bpb = evaluate_bpb(model, stream, byte_table, context, batch_size=32,
                           max_windows=args.max_windows, device=str(device))["val_bpb"]
    baseline_ms = time_forward(model, sample, "sequential") * 1000
    with torch.no_grad():
        sequential_logits = model(sample, mode="sequential").float()
    sequential_argmax = sequential_logits.argmax(dim=-1)
    print(f"seq_bpb={seq_bpb:.4f} baseline_ms={baseline_ms:.3f}")

    rng = np.random.default_rng(args.seed)
    all_evals = []
    best_overall = None

    if args.agreement_floor is None and args.bpb_floor is None:
        raise ValueError("must set exactly one of --agreement_floor or --bpb_floor")
    if args.surrogate_filter and args.bpb_floor is None:
        raise ValueError("--surrogate_filter requires --bpb_floor -- the "
                         "surrogate cost model predicts delta_bpb, not "
                         "argmax_agreement")

    predicted_cost = None
    if args.surrogate_filter:
        alt_modes_for_fit = [m for m in MODES if m != "sequential"]
        structured_jsons = [p.strip() for p in args.surrogate_structured_jsons.split(",") if p.strip()]
        fit_fn = (fit_quantile_surrogate_model if args.surrogate_type == "quantile"
                 else fit_surrogate_model)
        predicted_cost = fit_fn(
            args.surrogate_uniform_json, structured_jsons, n_layers, alt_modes_for_fit)

    mode_weights = None
    if args.bias_history_json:
        history_jsons = [p.strip() for p in args.bias_history_json.split(",") if p.strip()]
        weights = compute_mode_bias_weights(history_jsons, MODES)
        mode_weights = [weights[m] for m in MODES]

    def sample_mode():
        if mode_weights is not None:
            return rng.choice(MODES, p=mode_weights)
        return rng.choice(MODES)

    def utility(m):
        """speedup subject to a hard feasibility floor -- infeasible points
        get -inf so the search never wanders there. Prefer --bpb_floor:
        bits-per-byte on real held-out text is the objective
        language-modeling performance metric; argmax_agreement is a proxy
        that conflates "matches sequential's exact predictions" with
        "good" -- a mask can disagree with sequential while predicting the
        real text equally well or better."""
        if args.bpb_floor is not None:
            # Gate on degradation vs the reference checkpoint (e.g. the
            # dedicated specialist) when given -- per C24/C25, degradation
            # vs a checkpoint's OWN (possibly degraded) sequential mode is
            # not the same question as degradation vs a properly-trained
            # baseline, and the floor should be measured against whichever
            # one is the actual quality bar being enforced.
            gate_key = "bpb_degradation_vs_reference" if args.reference_bpb is not None else "bpb_degradation"
            if m[gate_key] > args.bpb_floor:
                return -1e9
        else:
            if m["argmax_agreement"] < args.agreement_floor:
                return -1e9
        if args.objective == "quality":
            return -m["bpb_degradation"]
        return baseline_ms / m["latency_ms"]

    seed_mask = None
    if args.seed_mask_json:
        seed_mask = json.load(open(args.seed_mask_json))[0]
        print(f"Seeding restart 0 from {args.seed_mask_json}'s first mask "
              f"(real hill-climb refinement of a heuristic-found candidate)")
    elif args.seed_from_corpus:
        # Random corpus first, hill-climb from its best point second -- per
        # user direction: a one-shot random draw is not itself a search,
        # feed the best mask found by broad random sampling (eval_6mode_masks.py's
        # corpus) as the seed, then locally refine from there. Works for
        # both objectives: lowest real delta_bpb is a safe, already-good
        # quality anchor to either speed up from or refine further.
        corpus = json.load(open(args.seed_from_corpus))
        best_row = min(corpus["rows"], key=lambda r: r["delta_bpb"])
        seed_mask = best_row["mask"]
        print(f"Seeding restart 0 from {args.seed_from_corpus}'s best random "
              f"draw (delta_bpb={best_row['delta_bpb']:+.4f})")
    elif args.seed_all_sequential:
        seed_mask = ["sequential"] * n_layers
        print("Seeding restart 0 from the plain all-sequential mask "
              "(BPB-improvement search should start from what it's trying to improve on)")
    elif args.seed_mask_from:
        seed_data = json.load(open(args.seed_mask_from))
        rows = seed_data["compiled"]
        if args.seed_mask_budget is not None:
            row = min(rows, key=lambda r: abs(r["budget"] - args.seed_mask_budget))
        else:
            row = max(rows, key=lambda r: r["actual_speedup"])
        seed_mask = row["mask"]
        print(f"Seeding restart 0 from {args.seed_mask_from} "
              f"(budget={row['budget']}, its speedup={row['actual_speedup']:.3f}x)")

    target_evals = args.n_restarts * (1 + args.steps_per_restart * args.neighbors_per_step)
    restart = 0
    while len(all_evals) < target_evals:
        if seed_mask is not None and (restart == 0 or args.seed_all_sequential):
            # For --seed_all_sequential specifically, EVERY restart re-seeds
            # from all-sequential, not just the first -- otherwise most of
            # the budget would revert to arbitrary random starts once restart
            # 0 stagnates, which has no special relationship to "improving
            # on sequential" (the exact behavior this flag exists to avoid).
            current_mask = list(seed_mask)
            current = measure_real(model, sample, current_mask, stream, byte_table, context,
                                   args.max_windows, device, sequential_logits, sequential_argmax, seq_bpb,
                                   reference_bpb=args.reference_bpb)
            all_evals.append(current)
        else:
            # Real bug found and fixed this session: a fully-random draw is
            # almost never feasible under a tight bpb_floor (e.g. only ~6%
            # of gradnorm's own 3000-mask corpus clears 0.02) -- starting an
            # infeasible restart meant EVERY step compared -1e9 to -1e9,
            # never accepting a neighbor, so the search silently reported
            # the infeasible random start's own (meaningless) speedup
            # forever. Resample until feasible, with a capped retry count;
            # fall back to all-sequential (always feasible for any
            # non-negative bpb_floor) if retries exhaust.
            current = None
            for _ in range(30):
                candidate_mask = [rng.choice(MODES) for _ in range(n_layers)]
                candidate = measure_real(model, sample, candidate_mask, stream, byte_table, context,
                                         args.max_windows, device, sequential_logits, sequential_argmax, seq_bpb,
                                   reference_bpb=args.reference_bpb)
                all_evals.append(candidate)
                if utility(candidate) > -1e8:
                    current = candidate
                    break
            if current is None:
                print("  30 random draws all infeasible -- falling back to all-sequential", flush=True)
                current_mask = ["sequential"] * n_layers
                current = measure_real(model, sample, current_mask, stream, byte_table, context,
                                       args.max_windows, device, sequential_logits, sequential_argmax, seq_bpb,
                                   reference_bpb=args.reference_bpb)
                all_evals.append(current)
        print(f"--- restart {restart} start: agree={current['argmax_agreement']:.4f} "
              f"speedup={baseline_ms/current['latency_ms']:.3f}x ---")

        steps_since_improvement = 0
        for step in range(args.steps_per_restart):
            if len(all_evals) >= target_evals:
                break
            if args.surrogate_filter:
                # Generate a LARGER pool of random single-layer mutations,
                # score ALL of them for free with the cheap surrogate, and
                # only spend the real GPU-measurement budget
                # (neighbors_per_step, unchanged) on the top-ranked ones --
                # same real-eval budget as plain hill-climbing, spent on
                # more promising candidates instead of blind random ones.
                pool = {}
                while len(pool) < args.surrogate_pool_size:
                    neighbor_mask = list(current["mask"])
                    layer = rng.integers(0, n_layers)
                    neighbor_mask[layer] = sample_mode()
                    pool[tuple(neighbor_mask)] = neighbor_mask
                ranked = sorted(
                    pool.values(),
                    key=lambda m: surrogate_rank_score(m, predicted_cost, args.bpb_floor, args.objective),
                    reverse=True,
                )
                neighbor_masks = ranked[:args.neighbors_per_step]
            else:
                neighbor_masks = []
                for _ in range(args.neighbors_per_step):
                    neighbor_mask = list(current["mask"])
                    layer = rng.integers(0, n_layers)
                    neighbor_mask[layer] = sample_mode()
                    neighbor_masks.append(neighbor_mask)

            candidates = []
            for neighbor_mask in neighbor_masks:
                m = measure_real(model, sample, neighbor_mask, stream, byte_table, context,
                                 args.max_windows, device, sequential_logits, sequential_argmax, seq_bpb,
                                   reference_bpb=args.reference_bpb)
                all_evals.append(m)
                candidates.append(m)
            best_candidate = max(candidates, key=utility)
            if utility(best_candidate) > utility(current):
                current = best_candidate
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            print(f"  step {step}: agree={current['argmax_agreement']:.4f} "
                  f"speedup={baseline_ms/current['latency_ms']:.3f}x "
                  f"n_seq={current['mask'].count('sequential')}", flush=True)
            if steps_since_improvement >= args.stagnation_patience:
                print(f"  stagnant for {steps_since_improvement} steps -- "
                      f"breaking to a fresh restart instead of repeating "
                      f"an unbeatable point", flush=True)
                break

        if best_overall is None or utility(current) > utility(best_overall):
            best_overall = current
        restart += 1

    floor_desc = f"bpb_floor={args.bpb_floor}" if args.bpb_floor is not None else f"agreement_floor={args.agreement_floor}"
    print(f"\n=== Best found ({floor_desc}) ===")
    print(f"speedup={baseline_ms/best_overall['latency_ms']:.3f}x "
          f"agree={best_overall['argmax_agreement']:.4f} "
          f"bpb_deg={best_overall['bpb_degradation']:+.4f}")
    best_mask = best_overall["mask"]
    mode_counts_str = ", ".join(f"{m}: {best_mask.count(m)}" for m in MODES)
    print(f"mode_counts: {{{mode_counts_str}}}")

    output = {
        "checkpoint": args.ckpt, "seq_bpb": seq_bpb, "baseline_ms": baseline_ms,
        "agreement_floor": args.agreement_floor, "bpb_floor": args.bpb_floor, "n_layers": n_layers,
        "all_evals": all_evals, "best": best_overall,
        "best_speedup": baseline_ms / best_overall["latency_ms"],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w"), indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
