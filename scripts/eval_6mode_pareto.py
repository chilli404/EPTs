"""Joint speed/quality measurement for the 6-mode program space.

eval_6mode_latency.py measures latency only; eval_6mode_masks.py measures
quality only, on a different (randomly sampled) set of masks. Neither can
answer "what is the best mask trading off speed against accuracy?" because
speed and quality are never measured on the *same* mask. This script fixes
that: for each mask it reports val_bpb, argmax_agreement, symmetric_kl AND
latency_ms in one row, then computes the Pareto-optimal subset (masks not
dominated on both axes by any other mask).

Mask set: the six pure modes, canonical skip/cheap-mode sweeps (as in
eval_6mode_latency.py), plus `--n_random` masks sampled the same way training
does (random_execution_mask_6mode), so the frontier includes masks
representative of what the model was actually trained under.

Usage:
  python scripts/eval_6mode_pareto.py \
      --ckpt .../1b_6mode_lt_sym/ckpts/step012000.safetensors \
      --config configs/scale1b_6mode_losstarget.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --n_random 200 \
      --output blackwell/results/6mode_pareto_1b_lt_sym.json
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
from fogen.model import GPT, ModelConfig
from fogen.training.train import random_execution_mask_6mode

MODES = ["sequential", "parallel", "skip", "reverse", "attn_only", "ffn_only"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_forward(model, x, mode, warmup=3, repeats=20):
    with torch.no_grad():
        for _ in range(warmup):
            model(x, mode=mode)
        _sync(x.device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x, mode=mode)
            _sync(x.device)
    return (time.perf_counter() - t0) / repeats


def pareto_front(rows, cost_key="latency_ms", quality_key="argmax_agreement"):
    """Rows not dominated: no other row is both cheaper and >= as accurate."""
    front = []
    for r in rows:
        dominated = any(
            o is not r and o[cost_key] <= r[cost_key]
            and o[quality_key] >= r[quality_key]
            and (o[cost_key] < r[cost_key] or o[quality_key] > r[quality_key])
            for o in rows
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r[cost_key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_random", type=int, default=40)
    parser.add_argument("--max_windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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

    with torch.no_grad():
        sequential_logits = model(sample, mode="sequential").float()
    sequential_logprob = F.log_softmax(sequential_logits, dim=-1)
    sequential_probability = sequential_logprob.exp()

    masks = {"all_" + m: [m] * n_layers for m in MODES}
    for n in range(1, min(n_layers, 11)):
        mask = ["sequential"] * n_layers
        for i in range(n_layers - n, n_layers):
            mask[i] = "skip"
        masks[f"skip_{n}_last"] = mask
    for n in [k for k in (2, 4, 6, 8) if k <= n_layers]:
        for cheap in ("attn_only", "ffn_only"):
            mask = ["sequential"] * n_layers
            for i in range(n_layers - n, n_layers):
                mask[i] = cheap
            masks[f"{cheap}_{n}_last"] = mask

    generator = torch.Generator().manual_seed(args.seed)
    for i in range(args.n_random):
        masks[f"random_{i}"] = random_execution_mask_6mode(
            n_layers, cfg.get("execution_training", {}).get("mode_probabilities"),
            generator)

    rows = []
    for name, mask in masks.items():
        model.cfg.execution_mode = mask  # evaluate_bpb reads this, not a mode= arg
        bpb = evaluate_bpb(model, stream, byte_table, context, batch_size=32,
                           max_windows=args.max_windows, device=str(device))["val_bpb"]
        with torch.no_grad():
            logits = model(sample, mode=mask).float()
        logprob = F.log_softmax(logits, dim=-1)
        symmetric_kl = (
            F.kl_div(logprob, sequential_probability, reduction="batchmean")
            + F.kl_div(sequential_logprob, logprob.exp(), reduction="batchmean")
        ) / 2
        t = time_forward(model, sample, mask)
        rows.append({
            "name": name,
            "mask_counts": {m: mask.count(m) for m in MODES},
            "val_bpb": bpb,
            "argmax_agreement": float(
                (logits.argmax(dim=-1) == sequential_logits.argmax(dim=-1))
                .float().mean()),
            "symmetric_kl": float(symmetric_kl),
            "latency_ms": t * 1000,
        })
        print(f"  {name:<16} bpb={bpb:.4f} agree={rows[-1]['argmax_agreement']:.4f} "
              f"lat={t*1000:.2f}ms", flush=True)

    seq_row = next(r for r in rows if r["name"] == "all_sequential")
    for r in rows:
        r["speedup"] = seq_row["latency_ms"] / r["latency_ms"]
        r["bpb_degradation"] = r["val_bpb"] - seq_row["val_bpb"]

    frontier = pareto_front(rows)

    output = {
        "checkpoint": args.ckpt,
        "n_layers": n_layers,
        "seq_bpb": seq_row["val_bpb"],
        "seq_latency_ms": seq_row["latency_ms"],
        "rows": rows,
        "pareto_front": frontier,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nPareto front ({len(frontier)}/{len(rows)} masks):")
    for r in frontier:
        print(f"  {r['name']:<16} speedup={r['speedup']:.3f}x  "
              f"agree={r['argmax_agreement']:.4f}  bpb_deg={r['bpb_degradation']:+.4f}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
