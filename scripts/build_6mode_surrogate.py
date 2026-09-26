"""Build a learned surrogate for the 6-mode program space, from every real
measurement collected this session -- no hand-designed functional form
(unlike the composition law: M2/M3/seq_penalty). Trains gradient-boosted
trees to predict (bpb_degradation, latency_ms, argmax_agreement) directly
from a mask's one-hot encoding, on the union of every rows/all_evals/
compiled entry across every result file for a scale.

Why: the composition law is provably exact for its own (linearized)
objective via the DP, but the DP's exactness doesn't help if the objective
itself is wrong -- a black-box search (real GPU measurements only) has
repeatedly found meaningfully better configurations than the law-guided
compiler (F84/F85). A learned surrogate lets a MUCH cheaper search explore
far more candidates (no GPU per candidate) while still capturing whatever
interaction structure exists in the data, without us having to hand-invent
each term.

Usage:
  python scripts/build_6mode_surrogate.py --scale 430m --n_layers 20 \
      --results_dir blackwell/results --output blackwell/results/6mode_surrogate_430m.pkl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

MODES = ["sequential", "parallel", "skip", "reverse", "attn_only", "ffn_only"]


def extract_rows(filepath, n_layers, checkpoint=None):
    """Normalize the 3 known schemas into (mask, bpb_degradation, latency_ms
    or None, argmax_agreement or None) tuples.

    If `checkpoint` is given, SKIP any file whose recorded "checkpoint"
    field doesn't exactly match -- real bug found and fixed this session:
    filename-pattern matching (e.g. "6mode_*430m*.json") silently mixed
    measurements from entirely different checkpoints sharing the same scale
    (e.g. 430m_6mode_lt vs 430m_6mode_even_cw10 vs 430m_6mode_ramp), and
    de-duplication by mask alone kept whichever file happened to be scanned
    first, discarding the correct checkpoint's measurement without warning.
    Verified: the identical mask gave delta_bpb of +0.0109 (cw10), -0.0063
    (cw100), -0.0171 (lt), +0.0093 (ramp), +0.0297 (llama_gradnorm) --
    different checkpoints, not noise. Every result file records its exact
    source checkpoint path; that is the robust ground truth, not the
    filename."""
    try:
        d = json.load(open(filepath))
    except Exception:
        return []
    file_checkpoint = d.get("checkpoint")
    if checkpoint is not None and file_checkpoint is not None and file_checkpoint != checkpoint:
        return []  # explicit, different checkpoint -- exclude
    # file_checkpoint is None: two script types (eval_6mode_active_loop.py,
    # eval_6mode_surrogate_search.py --validate) don't record this field.
    # Manually verified every such file produced this session used the
    # correct per-scale checkpoint consistently -- trusted, not excluded.
    out = []
    if "rows" in d and d["rows"]:
        for r in d["rows"]:
            mask = r.get("mask")
            if not mask or len(mask) != n_layers:
                continue
            bpb_deg = r.get("delta_bpb", r.get("bpb_degradation"))
            if bpb_deg is None:
                continue
            out.append((mask, bpb_deg, None, r.get("argmax_agreement")))
    if "new_rows" in d and d["new_rows"]:
        for r in d["new_rows"]:
            mask = r.get("mask")
            if not mask or len(mask) != n_layers:
                continue
            bpb_deg = r.get("delta_bpb")
            if bpb_deg is None:
                continue
            out.append((mask, bpb_deg, None, None))
    if "all_evals" in d and d["all_evals"]:
        for r in d["all_evals"]:
            mask = r.get("mask")
            if not mask or len(mask) != n_layers:
                continue
            out.append((mask, r.get("bpb_degradation"), r.get("latency_ms"), r.get("argmax_agreement")))
    if "results" in d and d["results"]:  # eval_6mode_active_loop.py's schema
        for r in d["results"]:
            mask = r.get("mask")
            if not mask or len(mask) != n_layers:
                continue
            out.append((mask, r.get("real_bpb_degradation"), r.get("real_latency_ms"), r.get("real_agreement")))
    if "validated" in d and d["validated"]:  # eval_6mode_surrogate_search.py's --validate schema
        for r in d["validated"]:
            mask = r.get("mask")
            if not mask or len(mask) != n_layers:
                continue
            out.append((mask, r.get("real_bpb_degradation"), r.get("real_latency_ms"), r.get("real_agreement")))
    if "compiled" in d and d["compiled"]:
        for r in d["compiled"]:
            mask = r.get("mask")
            if not mask or len(mask) != n_layers:
                continue
            out.append((mask, r.get("actual_bpb_degradation"), r.get("actual_latency_ms"), r.get("argmax_agreement")))
    return out


def mask_to_features(mask):
    """One-hot per (layer, mode) PLUS explicit per-mode counts and total
    sequential count -- the composition law's own history (M2's per-mode
    attenuation, the sequential-starvation term) shows these aggregate
    counts are dominant signal; handing them to the model directly is much
    easier than making a GBM rediscover a count effect from raw per-layer
    one-hot alone (it can, in principle, via many splits, but not as
    efficiently with a fixed tree budget)."""
    n_layers = len(mask)
    onehot = np.zeros(n_layers * len(MODES))
    counts = np.zeros(len(MODES))
    for l, m in enumerate(mask):
        idx = MODES.index(m)
        onehot[l * len(MODES) + idx] = 1.0
        counts[idx] += 1
    return np.concatenate([onehot, counts])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", required=True, help="e.g. 430m, 1b, 120m -- "
                        "used to glob matching result files")
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="exact source checkpoint path -- files whose "
                             "recorded checkpoint differs are excluded")
    parser.add_argument("--results_dir", default="blackwell/results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    import sys
    sys.path.insert(0, "src")
    sys.path.insert(0, "scripts")
    from fogen.execution_graph import evaluate_true_cost
    from eval_6mode_compiler import fit_m2m3
    ALT_MODES = ["parallel", "skip", "reverse", "attn_only", "ffn_only"]

    pattern = f"{args.results_dir}/6mode_*{args.scale}*.json"
    files = sorted(glob.glob(pattern))
    print(f"Scanning {len(files)} files matching '{pattern}'...")

    all_rows = []
    seen_masks = {}  # mask tuple -> index into all_rows
    for f in files:
        rows = extract_rows(f, args.n_layers, checkpoint=args.checkpoint)
        n_new = 0
        for mask, bpb_deg, lat, agree in rows:
            key = tuple(mask)
            if key in seen_masks:
                # Enrich, don't skip -- same bug as build_6mode_archive.py:
                # a later occurrence with latency/agreement data would be
                # silently dropped if an earlier file already had this mask
                # without it.
                idx = seen_masks[key]
                old_mask, old_bpb, old_lat, old_agree = all_rows[idx]
                all_rows[idx] = (old_mask, old_bpb,
                                old_lat if old_lat is not None else lat,
                                old_agree if old_agree is not None else agree)
                continue
            seen_masks[key] = len(all_rows)
            all_rows.append((mask, bpb_deg, lat, agree))
            n_new += 1
        if n_new:
            print(f"  {f.split('/')[-1]}: +{n_new} unique masks")

    print(f"\nTotal unique masks: {len(all_rows)}")
    n_with_latency = sum(1 for r in all_rows if r[2] is not None)
    n_with_agreement = sum(1 for r in all_rows if r[3] is not None)
    print(f"  with latency: {n_with_latency}  with agreement: {n_with_agreement}")

    X = np.array([mask_to_features(r[0]) for r in all_rows])
    y_bpb = np.array([r[1] for r in all_rows])
    masks_list = [r[0] for r in all_rows]

    def fit_and_report(X, y, label, loss="squared_error", alpha=None):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_frac, random_state=args.seed)
        kwargs = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=args.seed, loss=loss)
        if alpha is not None:
            kwargs["alpha"] = alpha
        model = GradientBoostingRegressor(**kwargs)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        r = np.corrcoef(pred, y_te)[0, 1]
        rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
        print(f"  {label}: n={len(y)} holdout_pearson_r={r:.4f} holdout_rmse={rmse:.5f}")
        model_full = GradientBoostingRegressor(**kwargs)
        model_full.fit(X, y)
        return model_full, float(r), rmse

    print("\nFitting q0 (the FULL composition law: M2 mode-specific attenuation + "
          "M3 pairwise + sequential-starvation term -- same fit_m2m3 used in "
          "production eval_6mode_compiler.py, not just plain M0 additive)...")
    fit_rows = [{"mask": m, "delta_bpb": float(v)} for m, v in zip(masks_list, y_bpb)]
    q0_base_costs, q0_alpha, q0_beta, q0_gamma, q0_seq_penalty = fit_m2m3(
        fit_rows, args.n_layers, ALT_MODES, "delta_bpb")
    def q0_predict(mask):
        return evaluate_true_cost(mask, q0_base_costs, ALT_MODES, q0_alpha, q0_beta,
                                  q0_gamma, seq_penalty=q0_seq_penalty)
    q0_pred_all = np.array([q0_predict(m) for m in masks_list])
    q0_r = np.corrcoef(q0_pred_all, y_bpb)[0, 1]
    print(f"  q0 alone: holdout-free in-sample pearson_r={q0_r:.4f} (fit on all data, no held-out split -- "
          f"this is the full production composition law, not a weaker stand-in)")
    residual = y_bpb - q0_pred_all

    print("\nFitting residual model (GBM on top of q0, per user-adopted design: "
          "'demote the composition law to a prior, learn only what it misses')...")
    resid_model, resid_r, resid_rmse = fit_and_report(X, residual, "residual (mean)")
    print("\nFitting conservative (upper-quantile) residual model -- a confidently-wrong "
          "UNDERestimate of true cost is much worse than an overestimate for a compiler's "
          "feasibility gate (real failure mode found and confirmed this session: job 54's "
          "surrogate underestimated real BPB by 2-6x in the heavy-skip region)...")
    resid_q95_model, resid_q95_r, resid_q95_rmse = fit_and_report(
        X, residual, "residual (q95, conservative)", loss="quantile", alpha=0.95)

    # Report the FULL (q0 + residual) model's holdout accuracy on actual bpb_degradation,
    # to compare directly against the old from-scratch GBM (F86: 0.8424).
    X_tr, X_te, y_tr, y_te, q0_tr, q0_te = train_test_split(
        X, y_bpb, q0_pred_all, test_size=args.test_frac, random_state=args.seed)
    resid_check = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=args.seed)
    resid_check.fit(X_tr, y_tr - q0_tr)
    full_pred = q0_te + resid_check.predict(X_te)
    full_r = np.corrcoef(full_pred, y_te)[0, 1]
    full_rmse = float(np.sqrt(np.mean((full_pred - y_te) ** 2)))
    print(f"\n  FULL (q0+residual) holdout: pearson_r={full_r:.4f} rmse={full_rmse:.5f} "
          f"(compare to from-scratch GBM's r=0.8424 on 430M in F86)")

    bpb_model, bpb_r, bpb_rmse = resid_model, full_r, full_rmse  # kept as (residual model, FULL accuracy) for downstream use

    lat_model, lat_r, lat_rmse = None, None, None
    if n_with_latency >= 30:
        mask_lat = np.array([r[2] is not None for r in all_rows])
        lat_model, lat_r, lat_rmse = fit_and_report(X[mask_lat], np.array([r[2] for r in all_rows if r[2] is not None]), "latency_ms")

    agree_model, agree_r, agree_rmse = None, None, None
    if n_with_agreement >= 30:
        mask_agree = np.array([r[3] is not None for r in all_rows])
        agree_model, agree_r, agree_rmse = fit_and_report(X[mask_agree], np.array([r[3] for r in all_rows if r[3] is not None]), "argmax_agreement")

    import pickle
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "n_layers": args.n_layers, "modes": MODES, "alt_modes": ALT_MODES,
            "q0_base_costs": q0_base_costs, "q0_alpha": q0_alpha, "q0_beta": q0_beta,
            "q0_gamma": q0_gamma, "q0_seq_penalty": q0_seq_penalty,  # full composition law prior
            "bpb_residual_model": resid_model, "bpb_residual_q95_model": resid_q95_model,
            "bpb_holdout_r": bpb_r, "bpb_holdout_rmse": bpb_rmse,  # FULL (q0+residual) accuracy
            "latency_model": lat_model, "latency_holdout_r": lat_r, "latency_holdout_rmse": lat_rmse,
            "agreement_model": agree_model, "agreement_holdout_r": agree_r, "agreement_holdout_rmse": agree_rmse,
            "n_training_masks": len(all_rows),
        }, f)
    print(f"\nSaved surrogate to {args.output}")


if __name__ == "__main__":
    main()
