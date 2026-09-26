"""Random-mask sweep over the 6-mode execution space, with a composition law.

The 10 named scenarios in eval_6mode_scenarios.py are hand-picked and give no
coverage of the space. This script instead:

  1. measures every single-layer probe: for each layer l and each non-sequential
     mode m, the cost of running layer l in mode m with all others sequential
     (L x 5 probes, e.g. 100 at 20 layers);
  2. samples N random masks from the full 6^L space and measures BPB, argmax
     agreement and symmetric KL against the all-sequential reference;
  3. fits an additive composition surrogate on a calibration split and reports
     held-out correlation, i.e. whether L x 5 probes predict 6^L programs.

Metric protocol matches eval_graph_rewrites.py so numbers are comparable to the
binary results.

Usage:
  python scripts/eval_6mode_masks.py \
      --ckpt .../430m_6mode_even_cw100/ckpts/step012000.safetensors \
      --config configs/scale430m_6mode_even_cw100.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --n_masks 3000 --output blackwell/results/6mode_masks_430m.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import load_tokenizer
from fogen.evals.bpb import evaluate_bpb, token_byte_table, val_stream
from fogen.model import GPT, ModelConfig

MODES = ["sequential", "parallel", "skip", "reverse", "attn_only", "ffn_only"]
ALT_MODES = MODES[1:]  # non-sequential


def load_model(ckpt, config_path, device):
    cfg = yaml.safe_load(open(config_path))
    mcfg = ModelConfig(**cfg["model"])
    model = GPT(mcfg).to(device)
    state = {k: v.float() for k, v in load_file(ckpt).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(k.startswith("rope_") for k in missing)
    return model.eval(), mcfg


def logits_for(model, sample, mask):
    with torch.no_grad(), torch.autocast(
        device_type=sample.device.type, dtype=torch.bfloat16,
        enabled=sample.device.type == "cuda",
    ):
        return model(sample, mode=mask).float()


def bpb_for(model, stream, byte_table, ctx, mask, device, max_windows):
    old = model.cfg.execution_mode
    model.cfg.execution_mode = mask
    out = evaluate_bpb(model, stream, byte_table, ctx, batch_size=16,
                       max_windows=max_windows, device=str(device))["val_bpb"]
    model.cfg.execution_mode = old
    return out


def agree_kl(model, sample, mask, ref_logits, ref_logprob, ref_prob):
    lg = logits_for(model, sample, mask)
    lp = F.log_softmax(lg, dim=-1)
    skl = (F.kl_div(lp, ref_prob, reduction="batchmean")
           + F.kl_div(ref_logprob, lp.exp(), reduction="batchmean")) / 2
    agree = (lg.argmax(-1) == ref_logits.argmax(-1)).float().mean()
    return float(agree), float(skl)


def fit_additive(masks, targets, n_layers, ridge=1e-6):
    """Least-squares fit of target ~ sum_l cost[l, mode_l].

    Design matrix is one-hot over (layer, alt-mode) pairs; sequential is the
    reference level and contributes zero, so the fitted coefficients are exactly
    the per-(layer, mode) costs.
    """
    idx = {(l, m): l * len(ALT_MODES) + j
           for l in range(n_layers) for j, m in enumerate(ALT_MODES)}
    X = np.zeros((len(masks), n_layers * len(ALT_MODES)))
    for i, mask in enumerate(masks):
        for l, m in enumerate(mask):
            if m != "sequential":
                X[i, idx[(l, m)]] = 1.0
    y = np.asarray(targets)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    coef = np.linalg.solve(A, X.T @ y)
    return coef, X


def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return pearson(ra, rb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--val_shards", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_masks", type=int, default=3000)
    p.add_argument("--max_windows", type=int, default=16,
                   help="BPB windows per mask (probes use 4x this)")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    print(f"=== {Path(__file__).name} config ===")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 40, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mcfg = load_model(args.ckpt, args.config, device)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    stream = val_stream(args.val_shards)
    byte_table = token_byte_table(tokenizer)
    ctx, L = mcfg.ctx_len, mcfg.n_layer

    sample = torch.tensor(
        np.asarray(stream[:2 * ctx], dtype=np.int64).reshape(2, ctx), device=device)
    ref_logits = logits_for(model, sample, "sequential")
    ref_logprob = F.log_softmax(ref_logits, dim=-1)
    ref_prob = ref_logprob.exp()

    seq_bpb = bpb_for(model, stream, byte_table, ctx, "sequential", device,
                      4 * args.max_windows)
    print(f"n_layers={L}  modes={len(MODES)}  program space = {len(MODES)}^{L} "
          f"= {len(MODES) ** L:.3e}")
    print(f"sequential baseline bpb = {seq_bpb:.4f}\n")

    # ---- 1. single-layer probes: L x 5
    print(f"=== single-layer probes ({L} layers x {len(ALT_MODES)} modes "
          f"= {L * len(ALT_MODES)}) ===", flush=True)
    probes = []
    for l in range(L):
        for m in ALT_MODES:
            mask = ["sequential"] * L
            mask[l] = m
            b = bpb_for(model, stream, byte_table, ctx, mask, device,
                        4 * args.max_windows)
            a, k = agree_kl(model, sample, mask, ref_logits, ref_logprob, ref_prob)
            probes.append({"layer": l, "mode": m, "bpb": b,
                           "delta_bpb": b - seq_bpb,
                           "argmax_agreement": a, "symmetric_kl": k})
        print(f"  layer {l:2d}: " + "  ".join(
            f"{r['mode'][:4]}={r['delta_bpb']:+.4f}"
            for r in probes[-len(ALT_MODES):]), flush=True)

    by_mode = {m: [r["delta_bpb"] for r in probes if r["mode"] == m] for m in ALT_MODES}
    print("\n  per-mode mean single-layer cost (ΔBPB):")
    for m in ALT_MODES:
        v = np.array(by_mode[m])
        print(f"    {m:12s} mean={v.mean():+.4f}  median={np.median(v):+.4f}  max={v.max():+.4f}")

    # ---- 2. random masks over 6^L
    rng = np.random.default_rng(args.seed)
    masks = [[MODES[i] for i in rng.integers(0, len(MODES), size=L)]
             for _ in range(args.n_masks)]

    print(f"\n=== {args.n_masks} random masks from 6^{L} ===", flush=True)
    rows = []
    for i, mask in enumerate(masks):
        b = bpb_for(model, stream, byte_table, ctx, mask, device, args.max_windows)
        a, k = agree_kl(model, sample, mask, ref_logits, ref_logprob, ref_prob)
        rows.append({"mask": mask, "bpb": b, "delta_bpb": b - seq_bpb,
                     "argmax_agreement": a, "symmetric_kl": k,
                     **{f"n_{m}": mask.count(m) for m in MODES}})
        if (i + 1) % 200 == 0:
            d = [r["delta_bpb"] for r in rows]
            ag = [r["argmax_agreement"] for r in rows]
            print(f"  [{i+1}/{args.n_masks}] mean ΔBPB={np.mean(d):+.4f} "
                  f"mean agree={np.mean(ag):.4f}", flush=True)

    # ---- 3. additive composition surrogate, calibration/holdout split
    n_cal = len(rows) // 2
    order = rng.permutation(len(rows))
    cal, hold = order[:n_cal], order[n_cal:]
    fit_results = {}
    for target_name in ("delta_bpb", "symmetric_kl"):
        y = [rows[i][target_name] for i in range(len(rows))]
        coef, X = fit_additive([rows[i]["mask"] for i in range(len(rows))], y, L)
        # refit on calibration only, evaluate on holdout
        coef_cal, _ = fit_additive([rows[i]["mask"] for i in cal],
                                   [y[i] for i in cal], L)
        pred_hold = X[hold] @ coef_cal
        true_hold = np.asarray([y[i] for i in hold])
        fit_results[target_name] = {
            "holdout_pearson": pearson(pred_hold, true_hold),
            "holdout_spearman": spearman(pred_hold, true_hold),
            "holdout_rmse": float(np.sqrt(np.mean((pred_hold - true_hold) ** 2))),
            "n_calibration": int(len(cal)), "n_holdout": int(len(hold)),
            "n_coefficients": int(L * len(ALT_MODES)),
        }
        # probe-only surrogate: use the measured single-layer costs directly
        probe_cost = {(r["layer"], r["mode"]): r[
            "delta_bpb" if target_name == "delta_bpb" else "symmetric_kl"]
            for r in probes}
        pred_probe = np.asarray([
            sum(probe_cost[(l, m)] for l, m in enumerate(rows[i]["mask"])
                if m != "sequential") for i in hold])
        fit_results[target_name]["probe_only_pearson"] = pearson(pred_probe, true_hold)
        fit_results[target_name]["probe_only_spearman"] = spearman(pred_probe, true_hold)

    deltas = np.array([r["delta_bpb"] for r in rows])
    agrees = np.array([r["argmax_agreement"] for r in rows])
    kls = np.array([r["symmetric_kl"] for r in rows])
    summary = {
        "mean_delta_bpb": float(deltas.mean()),
        "median_delta_bpb": float(np.median(deltas)),
        "p95_delta_bpb": float(np.percentile(deltas, 95)),
        "mean_agreement": float(agrees.mean()),
        "median_agreement": float(np.median(agrees)),
        "p05_agreement": float(np.percentile(agrees, 5)),
        "mean_symmetric_kl": float(kls.mean()),
        "median_symmetric_kl": float(np.median(kls)),
        "frac_agree_gt_090": float((agrees > 0.90).mean()),
        "frac_agree_gt_080": float((agrees > 0.80).mean()),
    }

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:24s} {v:+.4f}")
    print("\n=== additive composition (L x 5 coefficients) ===")
    for t, r in fit_results.items():
        print(f"  target={t}")
        print(f"    fitted   holdout pearson={r['holdout_pearson']:.4f} "
              f"spearman={r['holdout_spearman']:.4f} rmse={r['holdout_rmse']:.4f}")
        print(f"    probe-only holdout pearson={r['probe_only_pearson']:.4f} "
              f"spearman={r['probe_only_spearman']:.4f}")

    # per-mode-count breakdown on the dominant axis
    by_count = []
    for m in ALT_MODES:
        for c in range(L + 1):
            sel = [r for r in rows if r[f"n_{m}"] == c]
            if len(sel) < 20:
                continue
            by_count.append({
                "mode": m, "count": c, "n": len(sel),
                "mean_delta_bpb": float(np.mean([r["delta_bpb"] for r in sel])),
                "mean_agreement": float(np.mean([r["argmax_agreement"] for r in sel])),
            })

    # Perturbation-count curve: the sampling-invariant quantity. The native mean
    # depends on the mode distribution (uniform 1/6 gives E[perturbed]=0.833L,
    # vs ternary's 0.6L), so means are NOT comparable across families -- this curve
    # is. See scripts/compare_mask_families.py.
    by_perturb = []
    for k in range(L + 1):
        sel = [r for r in rows if (L - r["n_sequential"]) == k]
        if len(sel) < 5:
            continue
        by_perturb.append({
            "n_perturbed": k, "n": len(sel),
            "mean_agreement": float(np.mean([r["argmax_agreement"] for r in sel])),
            "mean_delta_bpb": float(np.mean([r["delta_bpb"] for r in sel])),
            "mean_symmetric_kl": float(np.mean([r["symmetric_kl"] for r in sel])),
        })

    result = {
        "checkpoint": args.ckpt, "config": args.config,
        "sampling": "uniform 1/6 per mode; E[perturbed] = 0.833*L",
        "by_perturb_count": by_perturb,
        "n_layers": L, "modes": MODES, "program_space": f"{len(MODES)}^{L}",
        "program_space_value": float(len(MODES) ** L),
        "n_masks": len(rows), "sequential_bpb": seq_bpb,
        "single_layer_probes": probes,
        "per_mode_single_layer": {
            m: {"mean": float(np.mean(by_mode[m])),
                "median": float(np.median(by_mode[m])),
                "max": float(np.max(by_mode[m]))} for m in ALT_MODES},
        "summary": summary,
        "composition": fit_results,
        "by_mode_count": by_count,
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
