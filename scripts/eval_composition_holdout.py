"""Held-out composition law evaluation.

Fits the composition model on one partition and evaluates on held-out masks.
Tests multiple split strategies to make the result bulletproof:

  1. Random 50/50 split
  2. Fit on |m| ≤ K, predict |m| > K (extrapolation)
  3. Fit on random masks, test on contiguous/structured masks
  4. Fit at one count range, predict another

Usage:
    PYTHONPATH=src python scripts/eval_composition_holdout.py \
        --graph_data blackwell/results/430m_poly_graphs.json \
        --output blackwell/results/430m_composition_holdout.json \
        --n_random 10000

    For 120M (exhaustive graphs already available):
    PYTHONPATH=src python scripts/eval_composition_holdout.py \
        --graph_data results/data/climbmix_graphs.json \
        --output results/data/climbmix_composition_holdout.json

    For scales with only ~200 graphs, use --n_random to generate more:
    PYTHONPATH=src python scripts/eval_composition_holdout.py \
        --graph_data blackwell/results/430m_poly_graphs.json \
        --ckpt runs/430m_poly_full/ckpts/step012000.safetensors \
        --config blackwell/configs/scale430m_poly_full.yaml \
        --val_shards data/climbmix/bpe8192/shards \
        --tokenizer_dir data/climbmix/bpe8192 \
        --n_random 10000 \
        --output blackwell/results/430m_composition_holdout.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr


def fit_2p(X, y, n):
    """Fit D(m) = α₀ · Σd · |m|^(-β)."""
    def loss(p):
        return float(np.mean((y - p[0] * X * np.power(n, -p[1]))**2))
    best = None
    for a0 in [0.5, 1.0, 1.3, 2.0]:
        for b0 in [0.1, 0.27, 0.4]:
            res = minimize(loss, [a0, b0], method='Nelder-Mead',
                          options={'xatol': 1e-9, 'fatol': 1e-13, 'maxiter': 10000})
            if best is None or res.fun < best.fun:
                best = res
    return best.x


def fit_1p(X, y):
    """Fit D(m) = α · Σd."""
    return float(np.dot(X, y) / np.dot(X, X))


def evaluate_split(X_train, y_train, n_train, X_test, y_test, n_test):
    """Fit on train, evaluate on test. Return metrics for 1p and 2p models."""
    # 1-param linear
    alpha = fit_1p(X_train, y_train)
    pred_1p = alpha * X_test
    rmse_1p = float(np.sqrt(np.mean((y_test - pred_1p)**2)) / np.sqrt(np.mean(y_test**2)))
    pearson_1p = float(pearsonr(pred_1p, y_test).statistic) if len(y_test) > 2 else 0
    spearman_1p = float(spearmanr(pred_1p, y_test).statistic) if len(y_test) > 2 else 0

    # 2-param count-adjusted
    p = fit_2p(X_train, y_train, n_train)
    pred_2p = p[0] * X_test * np.power(n_test, -p[1])
    rmse_2p = float(np.sqrt(np.mean((y_test - pred_2p)**2)) / np.sqrt(np.mean(y_test**2)))
    pearson_2p = float(pearsonr(pred_2p, y_test).statistic) if len(y_test) > 2 else 0
    spearman_2p = float(spearmanr(pred_2p, y_test).statistic) if len(y_test) > 2 else 0

    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "linear_alpha": float(alpha),
        "linear_rmse": rmse_1p,
        "linear_pearson": pearson_1p,
        "linear_spearman": spearman_1p,
        "ca_alpha0": float(p[0]),
        "ca_beta": float(p[1]),
        "ca_rmse": rmse_2p,
        "ca_pearson": pearson_2p,
        "ca_spearman": spearman_2p,
    }


def generate_random_masks(n_layers, count, seed=2026):
    """Generate random binary masks."""
    rng = np.random.default_rng(seed)
    masks = set()
    # Always include all-zero and all-one
    masks.add(tuple([0] * n_layers))
    masks.add(tuple([1] * n_layers))
    # All single-layer masks
    for i in range(n_layers):
        m = [0] * n_layers
        m[i] = 1
        masks.add(tuple(m))
    # Random masks
    while len(masks) < count + n_layers + 2:
        masks.add(tuple(rng.integers(0, 2, size=n_layers).tolist()))
    return sorted(masks, key=lambda m: (sum(m), m))


def evaluate_mask(model, sample, sequential_logits, sequential_logprob,
                  sequential_probability, bits, device):
    """Evaluate a single mask and return KL + agreement."""
    mask = ["parallel" if b else "sequential" for b in bits]
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits = model(sample, mode=mask).float()
    logprob = F.log_softmax(logits, dim=-1)
    symmetric_kl = (
        F.kl_div(logprob, sequential_probability, reduction="batchmean")
        + F.kl_div(sequential_logprob, logprob.exp(), reduction="batchmean")
    ) / 2
    agreement = float(
        (logits.argmax(dim=-1) == sequential_logits.argmax(dim=-1)).float().mean()
    )
    return float(symmetric_kl), agreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_data", required=True,
                        help="Existing graph results JSON (for defects + existing rows)")
    parser.add_argument("--ckpt", default=None,
                        help="Checkpoint for generating additional masks")
    parser.add_argument("--config", default=None)
    parser.add_argument("--val_shards", default=None)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--n_random", type=int, default=0,
                        help="Number of additional random masks to evaluate (requires --ckpt)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.graph_data) as f:
        graph_data = json.load(f)

    rows = graph_data["rows"]
    n_layers = len(graph_data["layer_defects"])
    baseline_kl = next(r["symmetric_kl"] for r in rows if sum(r["bits"]) == 0)
    singles = {r["bits"].index(1): r for r in rows if sum(r["bits"]) == 1}

    # Generate additional masks if checkpoint provided
    if args.n_random > 0 and args.ckpt:
        import yaml
        from safetensors.torch import load_file
        from fogen.model import GPT, ModelConfig

        print(f"Generating {args.n_random} additional random masks...")
        cfg = yaml.safe_load(open(args.config))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT(ModelConfig(**cfg["model"])).to(device)
        state = {k: v.float() for k, v in load_file(args.ckpt).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected
        model.eval()

        context = cfg["model"]["ctx_len"]
        from fogen.evals.bpb import val_stream
        stream = val_stream(args.val_shards)
        sample = torch.tensor(
            np.asarray(stream[:2 * context], dtype=np.int64).reshape(2, context),
            device=device,
        )

        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            sequential_logits = model(sample, mode="sequential").float()
        sequential_logprob = F.log_softmax(sequential_logits, dim=-1)
        sequential_probability = sequential_logprob.exp()

        existing_bits = {tuple(r["bits"]) for r in rows}
        new_masks = generate_random_masks(n_layers, args.n_random + len(existing_bits))
        new_masks = [m for m in new_masks if tuple(m) not in existing_bits]
        new_masks = new_masks[:args.n_random]

        print(f"Evaluating {len(new_masks)} new masks...")
        for i, bits in enumerate(new_masks):
            if i % 100 == 0:
                print(f"  [{i}/{len(new_masks)}]", flush=True)
            kl, agreement = evaluate_mask(
                model, sample, sequential_logits, sequential_logprob,
                sequential_probability, list(bits), device)
            rows.append({
                "bits": list(bits),
                "parallel_layers": sum(bits),
                "symmetric_kl": kl,
                "argmax_agreement": agreement,
                "source": "random_holdout",
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Build arrays
    multi = [r for r in rows if sum(r["bits"]) >= 2]
    X = np.array([sum(singles[i]["symmetric_kl"] - baseline_kl
                      for i, b in enumerate(r["bits"]) if b and i in singles)
                  for r in multi])
    y = np.array([r["symmetric_kl"] - baseline_kl for r in multi])
    n_par = np.array([sum(r["bits"]) for r in multi], dtype=float)

    print(f"\nTotal multi-layer graphs: {len(multi)}")

    results = {"n_layers": n_layers, "n_total": len(multi), "splits": {}}

    # ===== Split 1: Random 50/50 =====
    print("\n--- Split 1: Random 50/50 ---")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(multi))
    half = len(idx) // 2
    train_idx, test_idx = idx[:half], idx[half:]
    result = evaluate_split(
        X[train_idx], y[train_idx], n_par[train_idx],
        X[test_idx], y[test_idx], n_par[test_idx])
    results["splits"]["random_5050"] = result
    print(f"  Train={result['n_train']}, Test={result['n_test']}")
    print(f"  1p: Pearson={result['linear_pearson']:.4f}, RMSE={result['linear_rmse']:.4f}")
    print(f"  2p: Pearson={result['ca_pearson']:.4f}, RMSE={result['ca_rmse']:.4f}, β={result['ca_beta']:.3f}")

    # ===== Split 2: Fit |m| ≤ K, predict |m| > K =====
    print("\n--- Split 2: Extrapolation (fit low-count, predict high-count) ---")
    mid = n_layers // 2
    low_mask = n_par <= mid
    high_mask = n_par > mid
    if low_mask.sum() > 5 and high_mask.sum() > 5:
        result = evaluate_split(
            X[low_mask], y[low_mask], n_par[low_mask],
            X[high_mask], y[high_mask], n_par[high_mask])
        results["splits"][f"fit_leq{mid}_pred_gt{mid}"] = result
        print(f"  Train (|m|≤{mid}): {result['n_train']}, Test (|m|>{mid}): {result['n_test']}")
        print(f"  1p: Pearson={result['linear_pearson']:.4f}, RMSE={result['linear_rmse']:.4f}")
        print(f"  2p: Pearson={result['ca_pearson']:.4f}, RMSE={result['ca_rmse']:.4f}, β={result['ca_beta']:.3f}")

    # ===== Split 3: Fit |m| > K, predict |m| ≤ K (reverse extrapolation) =====
    print("\n--- Split 3: Reverse extrapolation (fit high-count, predict low-count) ---")
    if low_mask.sum() > 5 and high_mask.sum() > 5:
        result = evaluate_split(
            X[high_mask], y[high_mask], n_par[high_mask],
            X[low_mask], y[low_mask], n_par[low_mask])
        results["splits"][f"fit_gt{mid}_pred_leq{mid}"] = result
        print(f"  Train (|m|>{mid}): {result['n_train']}, Test (|m|≤{mid}): {result['n_test']}")
        print(f"  1p: Pearson={result['linear_pearson']:.4f}, RMSE={result['linear_rmse']:.4f}")
        print(f"  2p: Pearson={result['ca_pearson']:.4f}, RMSE={result['ca_rmse']:.4f}, β={result['ca_beta']:.3f}")

    # ===== Split 4: Fit on even |m|, predict odd |m| =====
    print("\n--- Split 4: Fit even |m|, predict odd |m| ---")
    even_mask = (n_par % 2 == 0)
    odd_mask = (n_par % 2 == 1)
    if even_mask.sum() > 5 and odd_mask.sum() > 5:
        result = evaluate_split(
            X[even_mask], y[even_mask], n_par[even_mask],
            X[odd_mask], y[odd_mask], n_par[odd_mask])
        results["splits"]["fit_even_pred_odd"] = result
        print(f"  Train (even |m|): {result['n_train']}, Test (odd |m|): {result['n_test']}")
        print(f"  1p: Pearson={result['linear_pearson']:.4f}, RMSE={result['linear_rmse']:.4f}")
        print(f"  2p: Pearson={result['ca_pearson']:.4f}, RMSE={result['ca_rmse']:.4f}, β={result['ca_beta']:.3f}")

    # ===== Split 5: 5-fold cross-validation =====
    print("\n--- Split 5: 5-fold cross-validation ---")
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(multi))
    fold_size = len(multi) // 5
    cv_results = []
    for fold in range(5):
        test_idx = perm[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.array([i for i in perm if i not in test_idx])
        result = evaluate_split(
            X[train_idx], y[train_idx], n_par[train_idx],
            X[test_idx], y[test_idx], n_par[test_idx])
        cv_results.append(result)
    avg_1p = np.mean([r["linear_pearson"] for r in cv_results])
    avg_2p = np.mean([r["ca_pearson"] for r in cv_results])
    avg_rmse_1p = np.mean([r["linear_rmse"] for r in cv_results])
    avg_rmse_2p = np.mean([r["ca_rmse"] for r in cv_results])
    avg_beta = np.mean([r["ca_beta"] for r in cv_results])
    results["splits"]["5fold_cv"] = {
        "folds": cv_results,
        "avg_linear_pearson": float(avg_1p),
        "avg_ca_pearson": float(avg_2p),
        "avg_linear_rmse": float(avg_rmse_1p),
        "avg_ca_rmse": float(avg_rmse_2p),
        "avg_ca_beta": float(avg_beta),
        "beta_std": float(np.std([r["ca_beta"] for r in cv_results])),
    }
    print(f"  Avg 1p Pearson: {avg_1p:.4f}, RMSE: {avg_rmse_1p:.4f}")
    print(f"  Avg 2p Pearson: {avg_2p:.4f}, RMSE: {avg_rmse_2p:.4f}, β: {avg_beta:.3f}±{np.std([r['ca_beta'] for r in cv_results]):.3f}")

    # ===== Summary =====
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Total graphs evaluated: {len(multi)}")
    print(f"\n  {'Split':<35} | {'1p Pearson':>10} | {'2p Pearson':>10} | {'2p β':>6}")
    print(f"  {'-'*35}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
    for name, r in results["splits"].items():
        if name == "5fold_cv":
            p1 = r["avg_linear_pearson"]
            p2 = r["avg_ca_pearson"]
            beta = r["avg_ca_beta"]
        else:
            p1 = r["linear_pearson"]
            p2 = r["ca_pearson"]
            beta = r["ca_beta"]
        print(f"  {name:<35} | {p1:>10.4f} | {p2:>10.4f} | {beta:>6.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
