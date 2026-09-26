"""Measure real (bpb, latency, agreement) for a fixed list of specific
masks on a given checkpoint -- used to seed a new checkpoint's archive with
a handful of real latency-labeled entries when no search history exists
yet for it (e.g. reusing known-fast mask structures found on a different
checkpoint, but measuring them for real on THIS checkpoint's own weights).

Usage:
  python scripts/eval_specific_masks.py --ckpt ... --config ... \
      --masks_file masks.json --output out.json
  # masks_file: JSON list of lists, one mask (list of mode strings) per line
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from fogen.data import load_tokenizer
from fogen.evals.bpb import evaluate_bpb, token_byte_table, val_stream
from fogen.model import GPT, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--masks_file", required=True)
    parser.add_argument("--max_windows", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"=== {Path(__file__).name} config ===")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 40, flush=True)

    masks = json.load(open(args.masks_file))

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))
    mcfg = ModelConfig(**cfg["model"])
    model = GPT(mcfg).to(device)
    state = {k: v.float() for k, v in load_file(args.ckpt).items()}
    model.load_state_dict(state, strict=False)
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

    def time_forward(mode, warmup=3, repeats=10):
        with torch.no_grad():
            for _ in range(warmup):
                model(sample, mode=mode)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                model(sample, mode=mode)
                if device.type == "cuda":
                    torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats * 1000

    with torch.no_grad():
        sequential_logits = model(sample, mode="sequential").float()
    sequential_argmax = sequential_logits.argmax(dim=-1)
    print(f"seq_bpb={seq_bpb:.4f}")

    results = []
    for mask in masks:
        model.cfg.execution_mode = mask
        real_bpb = evaluate_bpb(model, stream, byte_table, context, batch_size=32,
                                max_windows=args.max_windows, device=str(device))["val_bpb"]
        real_lat = time_forward(mask)
        with torch.no_grad():
            logits = model(sample, mode=mask).float()
        real_agree = float((logits.argmax(dim=-1) == sequential_argmax).float().mean())
        results.append({"mask": mask, "bpb_degradation": real_bpb - seq_bpb,
                        "latency_ms": real_lat, "argmax_agreement": real_agree})
        print(f"  bpb={real_bpb-seq_bpb:+.4f} lat={real_lat:.2f}ms agree={real_agree:.4f}", flush=True)

    output = {"checkpoint": args.ckpt, "seq_bpb": seq_bpb, "all_evals": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w"), indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
