"""Evaluate held-out ternary masks on ternary-trained vs binary-only models.

Samples thousands of random {sequential, parallel, skip}^L masks and
measures BPB degradation for each, comparing a ternary-trained model
against a binary-only (seq/par) polymorphic model.

Usage:
  python scripts/eval_ternary_masks.py \
      --ternary_ckpt runs/120m_ternary_poly/ckpts/step003500.safetensors \
      --binary_ckpt runs/120m_poly_cw01/ckpts/step003500.safetensors \
      --ternary_config configs/scale120m_ternary_polymorphic.yaml \
      --binary_config configs/scale120m_polymorphic_cw01.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --n_masks 5000 \
      --output blackwell/results/ternary_mask_evaluation.json
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
from fogen.evals.bpb import evaluate_bpb, val_stream
from fogen.model import GPT, ModelConfig

MODE_NAMES = ["sequential", "parallel", "skip"]


def load_model(ckpt, config_path, device):
    cfg = yaml.safe_load(open(config_path))
    mcfg = ModelConfig(**cfg["model"])
    model = GPT(mcfg).to(device)
    state = {k: v.float() for k, v in load_file(ckpt).items()}
    missing, _ = model.load_state_dict(state, strict=False)
    assert all(k.startswith("rope_") for k in missing)
    return model.eval(), mcfg


def eval_bpb(model, stream, token_bytes, ctx_len, mask, device, max_windows=32):
    old = model.cfg.execution_mode
    model.cfg.execution_mode = mask
    result = evaluate_bpb(model, stream, token_bytes, ctx_len,
                          batch_size=8, max_windows=max_windows,
                          device=str(device))
    model.cfg.execution_mode = old
    return result["val_bpb"]


def sample_ternary_masks(n_layers, n_masks, rng, probs=(0.4, 0.4, 0.2)):
    """Sample random ternary masks with given mode probabilities."""
    masks = []
    cumprobs = [probs[0], probs[0] + probs[1]]
    for _ in range(n_masks):
        r = rng.random(n_layers)
        mask = []
        for v in r:
            if v < cumprobs[0]:
                mask.append("sequential")
            elif v < cumprobs[1]:
                mask.append("parallel")
            else:
                mask.append("skip")
        masks.append(mask)
    return masks


def mask_stats(mask):
    return {
        "n_seq": mask.count("sequential"),
        "n_par": mask.count("parallel"),
        "n_skip": mask.count("skip"),
    }


def _logits(model, sample, mask):
    with torch.no_grad(), torch.autocast(
        device_type=sample.device.type, dtype=torch.bfloat16,
        enabled=sample.device.type == "cuda",
    ):
        return model(sample, mode=mask).float()


def agreement_and_kl(model, sample, mask, ref_logits, ref_logprob, ref_prob):
    """Matches the protocol in eval_graph_rewrites.py so numbers are comparable."""
    logits = _logits(model, sample, mask)
    logprob = F.log_softmax(logits, dim=-1)
    symmetric_kl = (
        F.kl_div(logprob, ref_prob, reduction="batchmean")
        + F.kl_div(ref_logprob, logprob.exp(), reduction="batchmean")
    ) / 2
    agreement = (logits.argmax(dim=-1) == ref_logits.argmax(dim=-1)).float().mean()
    return float(agreement), float(symmetric_kl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ternary_ckpt", required=True)
    parser.add_argument("--binary_ckpt", required=True)
    parser.add_argument("--ternary_config", required=True)
    parser.add_argument("--binary_config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--n_masks", type=int, default=5000)
    parser.add_argument("--skip_prob", type=float, default=0.2,
                        help="Per-layer skip probability for the mask sampling "
                             "distribution (parallel is held at 0.4).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading ternary model...", flush=True)
    ternary_model, mcfg = load_model(args.ternary_ckpt, args.ternary_config, device)
    print("Loading binary model...", flush=True)
    binary_model, _ = load_model(args.binary_ckpt, args.binary_config, device)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    stream = val_stream(args.val_shards)
    token_bytes = torch.tensor(
        [len(tokenizer.decode([i]).encode("utf-8")) for i in range(mcfg.vocab_size)],
        dtype=torch.long)

    n_layers = mcfg.n_layer

    # Baselines
    print("\n=== Baselines ===", flush=True)
    tern_seq = eval_bpb(ternary_model, stream, token_bytes, mcfg.ctx_len,
                        "sequential", device, max_windows=64)
    tern_par = eval_bpb(ternary_model, stream, token_bytes, mcfg.ctx_len,
                        "parallel", device, max_windows=64)
    bin_seq = eval_bpb(binary_model, stream, token_bytes, mcfg.ctx_len,
                       "sequential", device, max_windows=64)
    bin_par = eval_bpb(binary_model, stream, token_bytes, mcfg.ctx_len,
                       "parallel", device, max_windows=64)
    print(f"  Ternary: seq={tern_seq:.4f}  par={tern_par:.4f}")
    print(f"  Binary:  seq={bin_seq:.4f}  par={bin_par:.4f}")

    # Sequential reference logits for agreement/KL (each model vs its own seq)
    sample = torch.tensor(
        np.asarray(stream[:2 * mcfg.ctx_len], dtype=np.int64).reshape(2, mcfg.ctx_len),
        device=device,
    )
    refs = {}
    for name, m in (("ternary", ternary_model), ("binary", binary_model)):
        rl = _logits(m, sample, "sequential")
        rlp = F.log_softmax(rl, dim=-1)
        refs[name] = (rl, rlp, rlp.exp())

    # Mask sampling distribution: parallel fixed at 0.4, skip from --skip_prob
    eval_probs = (1.0 - 0.4 - args.skip_prob, 0.4, args.skip_prob)
    assert eval_probs[0] >= 0, f"skip_prob too large: {args.skip_prob}"
    print(f"\nMask sampling: seq={eval_probs[0]:.2f} par={eval_probs[1]:.2f} "
          f"skip={eval_probs[2]:.2f} (mean {args.skip_prob * n_layers:.1f} skips)",
          flush=True)
    rng = np.random.default_rng(42)
    masks = sample_ternary_masks(n_layers, args.n_masks, rng, probs=eval_probs)

    print(f"\n=== Evaluating {args.n_masks} ternary masks ===", flush=True)
    rows = []
    for i, mask in enumerate(masks):
        stats = mask_stats(mask)
        tern_bpb = eval_bpb(ternary_model, stream, token_bytes, mcfg.ctx_len,
                            mask, device)
        bin_bpb = eval_bpb(binary_model, stream, token_bytes, mcfg.ctx_len,
                           mask, device)
        tern_agree, tern_kl = agreement_and_kl(
            ternary_model, sample, mask, *refs["ternary"])
        bin_agree, bin_kl = agreement_and_kl(
            binary_model, sample, mask, *refs["binary"])
        rows.append({
            "mask": mask,
            **stats,
            "ternary_bpb": float(tern_bpb),
            "binary_bpb": float(bin_bpb),
            "ternary_degradation": float(tern_bpb - tern_seq),
            "binary_degradation": float(bin_bpb - bin_seq),
            "ternary_argmax_agreement": tern_agree,
            "ternary_symmetric_kl": tern_kl,
            "binary_argmax_agreement": bin_agree,
            "binary_symmetric_kl": bin_kl,
        })
        if (i + 1) % 100 == 0:
            tern_degs = [r["ternary_degradation"] for r in rows]
            bin_degs = [r["binary_degradation"] for r in rows]
            print(f"  [{i+1}/{args.n_masks}] tern mean Δ={np.mean(tern_degs):.4f}  "
                  f"bin mean Δ={np.mean(bin_degs):.4f}", flush=True)

    # Summary statistics
    tern_degs = np.array([r["ternary_degradation"] for r in rows])
    bin_degs = np.array([r["binary_degradation"] for r in rows])

    print(f"\n=== Summary ===")
    print(f"{'':>25} {'Ternary':>10} {'Binary':>10}")
    print(f"{'Mean Δ':>25} {np.mean(tern_degs):>10.4f} {np.mean(bin_degs):>10.4f}")
    print(f"{'Median Δ':>25} {np.median(tern_degs):>10.4f} {np.median(bin_degs):>10.4f}")
    print(f"{'95th pct Δ':>25} {np.percentile(tern_degs, 95):>10.4f} {np.percentile(bin_degs, 95):>10.4f}")
    print(f"{'Max Δ':>25} {np.max(tern_degs):>10.4f} {np.max(bin_degs):>10.4f}")
    print(f"{'Frac Δ < 0.01':>25} {np.mean(tern_degs < 0.01):>10.3f} {np.mean(bin_degs < 0.01):>10.3f}")
    print(f"{'Frac Δ < 0.05':>25} {np.mean(tern_degs < 0.05):>10.3f} {np.mean(bin_degs < 0.05):>10.3f}")
    print(f"{'Frac Δ < 0.10':>25} {np.mean(tern_degs < 0.10):>10.3f} {np.mean(bin_degs < 0.10):>10.3f}")

    tern_agr = np.array([r["ternary_argmax_agreement"] for r in rows])
    bin_agr = np.array([r["binary_argmax_agreement"] for r in rows])
    tern_kls = np.array([r["ternary_symmetric_kl"] for r in rows])
    bin_kls = np.array([r["binary_symmetric_kl"] for r in rows])
    print(f"{'Mean agreement':>25} {np.mean(tern_agr):>10.4f} {np.mean(bin_agr):>10.4f}")
    print(f"{'Median agreement':>25} {np.median(tern_agr):>10.4f} {np.median(bin_agr):>10.4f}")
    print(f"{'5th pct agreement':>25} {np.percentile(tern_agr, 5):>10.4f} {np.percentile(bin_agr, 5):>10.4f}")
    print(f"{'Mean sym KL':>25} {np.mean(tern_kls):>10.4f} {np.mean(bin_kls):>10.4f}")
    print(f"{'Median sym KL':>25} {np.median(tern_kls):>10.4f} {np.median(bin_kls):>10.4f}")

    # Breakdown by skip count
    print(f"\n=== By skip count ===")
    print(f"{'n_skip':>8} {'n':>6} {'tern mean Δ':>12} {'bin mean Δ':>12} "
          f"{'tern agree':>11} {'bin agree':>11} {'tern KL':>10}")
    by_skip = []
    for ns in range(n_layers + 1):
        idx = [i for i, r in enumerate(rows) if r["n_skip"] == ns]
        if len(idx) < 5:
            continue
        rec = {
            "n_skip": ns, "n": len(idx),
            "ternary_mean_degradation": float(np.mean(tern_degs[idx])),
            "binary_mean_degradation": float(np.mean(bin_degs[idx])),
            "ternary_mean_agreement": float(np.mean(tern_agr[idx])),
            "binary_mean_agreement": float(np.mean(bin_agr[idx])),
            "ternary_mean_symmetric_kl": float(np.mean(tern_kls[idx])),
            "binary_mean_symmetric_kl": float(np.mean(bin_kls[idx])),
        }
        by_skip.append(rec)
        print(f"{ns:>8} {len(idx):>6} {rec['ternary_mean_degradation']:>12.4f} "
              f"{rec['binary_mean_degradation']:>12.4f} "
              f"{rec['ternary_mean_agreement']:>11.4f} "
              f"{rec['binary_mean_agreement']:>11.4f} "
              f"{rec['ternary_mean_symmetric_kl']:>10.3f}")

    # Perturbation-count curve: sampling-invariant, so comparable across mask

    # families (binary/ternary/6-mode all sample different distributions).

    # See scripts/compare_mask_families.py.

    by_perturb = []

    for k in range(n_layers + 1):

        sel = [r for r in rows if (n_layers - r["n_seq"]) == k]

        if len(sel) < 5:

            continue

        by_perturb.append({

            "n_perturbed": k, "n": len(sel),

            "ternary_mean_agreement": float(np.mean(

                [r["ternary_argmax_agreement"] for r in sel])),

            "binary_mean_agreement": float(np.mean(

                [r["binary_argmax_agreement"] for r in sel])),

            "ternary_mean_degradation": float(np.mean(

                [r["ternary_degradation"] for r in sel])),

        })


    result = {
        "by_perturb_count": by_perturb,
        "ternary_checkpoint": args.ternary_ckpt,
        "binary_checkpoint": args.binary_ckpt,
        "n_layers": n_layers,
        "n_masks": len(rows),
        "eval_skip_prob": args.skip_prob,
        "eval_probs": {"sequential": eval_probs[0], "parallel": eval_probs[1],
                       "skip": eval_probs[2]},
        "baselines": {
            "ternary_seq": float(tern_seq),
            "ternary_par": float(tern_par),
            "binary_seq": float(bin_seq),
            "binary_par": float(bin_par),
        },
        "summary": {
            "ternary_mean_degradation": float(np.mean(tern_degs)),
            "ternary_median_degradation": float(np.median(tern_degs)),
            "ternary_p95_degradation": float(np.percentile(tern_degs, 95)),
            "ternary_max_degradation": float(np.max(tern_degs)),
            "binary_mean_degradation": float(np.mean(bin_degs)),
            "binary_median_degradation": float(np.median(bin_degs)),
            "binary_p95_degradation": float(np.percentile(bin_degs, 95)),
            "binary_max_degradation": float(np.max(bin_degs)),
            "ternary_frac_lt_001": float(np.mean(tern_degs < 0.01)),
            "ternary_frac_lt_005": float(np.mean(tern_degs < 0.05)),
            "ternary_frac_lt_010": float(np.mean(tern_degs < 0.10)),
            "binary_frac_lt_001": float(np.mean(bin_degs < 0.01)),
            "binary_frac_lt_005": float(np.mean(bin_degs < 0.05)),
            "binary_frac_lt_010": float(np.mean(bin_degs < 0.10)),
            "ternary_mean_agreement": float(np.mean(tern_agr)),
            "ternary_median_agreement": float(np.median(tern_agr)),
            "ternary_p05_agreement": float(np.percentile(tern_agr, 5)),
            "binary_mean_agreement": float(np.mean(bin_agr)),
            "binary_median_agreement": float(np.median(bin_agr)),
            "binary_p05_agreement": float(np.percentile(bin_agr, 5)),
            "ternary_mean_symmetric_kl": float(np.mean(tern_kls)),
            "ternary_median_symmetric_kl": float(np.median(tern_kls)),
            "binary_mean_symmetric_kl": float(np.mean(bin_kls)),
            "binary_median_symmetric_kl": float(np.median(bin_kls)),
        },
        "by_skip_count": by_skip,
        "rows": rows,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
