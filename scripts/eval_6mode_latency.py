"""Measure latency under 6-mode execution masks.

Profiles wall-clock time for each pure mode (sequential, parallel, skip,
reverse, attn_only, ffn_only) and representative mixed masks, so a speedup
claim for the 6-mode program space can be checked the same way
eval_ternary_latency.py checks it for ternary.

Usage:
  python scripts/eval_6mode_latency.py \
      --ckpt /mnt/mldata/checkpoints/execution_graph_transformers/1b_6mode_lt_sym/ckpts/step012000.safetensors \
      --config blackwell/configs/scale1b_6mode_losstarget.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/6mode_latency_1b_lt_sym.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

from fogen.data import ShardedLoader
from fogen.model import GPT, ModelConfig

MODES = ["sequential", "parallel", "skip", "reverse", "attn_only", "ffn_only"]


def _sync(x):
    # perf_counter around an async device call under-measures unless the
    # queue is drained first -- true for both CUDA and MPS.
    if x.is_cuda:
        torch.cuda.synchronize()
    elif x.device.type == "mps":
        torch.mps.synchronize()


def time_forward(model, x, mode, warmup=3, repeats=20):
    for _ in range(warmup):
        model(x, mode=mode)
    _sync(x)
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x, mode=mode)
        _sync(x)
    return (time.perf_counter() - t0) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = ModelConfig(**cfg["model"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # bf16 matmul is not supported on MPS; float32 stays the correct default there.
    param_dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    model = GPT(mcfg).to(device=device, dtype=param_dtype)
    state = {k: v.to(param_dtype) for k, v in load_file(args.ckpt).items()}
    missing, _ = model.load_state_dict(state, strict=False)
    assert all(k.startswith("rope_") for k in missing)
    model.eval()

    loader = ShardedLoader(args.val_shards, 8, mcfg.ctx_len, seed=42, device=device)
    x, _ = loader.next_batch()
    n_layers = mcfg.n_layer

    results = []

    with torch.no_grad():
        t_seq = time_forward(model, x, "sequential")
        results.append({"mode": "all_sequential", "latency_ms": t_seq * 1000,
                         "speedup": 1.0})
        print(f"  all_sequential: {t_seq*1000:.2f}ms (baseline)")

        for mode in MODES[1:]:
            t = time_forward(model, x, mode)
            results.append({"mode": f"all_{mode}", "latency_ms": t * 1000,
                             "speedup": t_seq / t})
            print(f"  all_{mode:<10}: {t*1000:.2f}ms ({t_seq/t:.2f}x)")

        # Skip N layers (last-first), the axis that gave real speedup for
        # ternary/binary at other scales.
        for n_skip in range(1, min(n_layers, 11)):
            mask = ["sequential"] * n_layers
            for i in range(n_layers - n_skip, n_layers):
                mask[i] = "skip"
            t = time_forward(model, x, mask)
            results.append({"mode": f"skip_{n_skip}_last", "n_skip": n_skip,
                             "latency_ms": t * 1000, "speedup": t_seq / t})
            print(f"  skip {n_skip:>2} (last): {t*1000:.2f}ms ({t_seq/t:.2f}x)")

        # attn_only/ffn_only drop one sublayer per layer -- the 6-mode-specific
        # cheap-mode axis with no ternary analogue.
        for n_cheap in [n for n in (2, 4, 6, 8) if n <= n_layers]:
            for cheap_mode in ("attn_only", "ffn_only"):
                mask = ["sequential"] * n_layers
                for i in range(n_layers - n_cheap, n_layers):
                    mask[i] = cheap_mode
                t = time_forward(model, x, mask)
                results.append({
                    "mode": f"{cheap_mode}_{n_cheap}_last", "n_cheap": n_cheap,
                    "latency_ms": t * 1000, "speedup": t_seq / t})
                print(f"  {cheap_mode} {n_cheap:>2} (last): {t*1000:.2f}ms "
                      f"({t_seq/t:.2f}x)")

    output = {
        "checkpoint": args.ckpt,
        "n_layers": n_layers,
        "batch_size": 8,
        "ctx_len": mcfg.ctx_len,
        "baseline_ms": t_seq * 1000,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
