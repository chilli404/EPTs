"""Re-evaluate given masks' BPB on VALIDATION WINDOWS DISJOINT from whatever
was used during search -- evaluate_bpb always windows from index 0 of
whatever stream array it's given, so search (windows 0..max_windows-1) and
any later re-check using the same max_windows default silently reuse the
identical windows. This offsets the stream by `--window_offset` windows
before evaluating, giving a genuinely independent BPB estimate -- the
correct way to check whether a winning mask's near-budget margin survives
on data it wasn't (even implicitly) selected against.

Usage:
  python scripts/eval_disjoint_windows.py \
      --ckpt .../1b_6mode_gradnorm_cap10/ckpts/step024000.safetensors \
      --config blackwell/configs/scale1b_6mode_losstarget.yaml \
      --val_shards data/climbmix/bpe8192/shards --tokenizer_dir data/climbmix/bpe8192 \
      --masks_json /tmp/confound5_winners.json \
      --window_offset 64 --max_windows 64 \
      --output blackwell/results/confound5_disjoint_recheck.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--masks_json", required=True,
                        help="JSON dict {name: mask} of masks to re-check")
    parser.add_argument("--window_offset", type=int, default=64,
                        help="number of ctx_len-sized windows to skip from "
                             "the start of the stream, to avoid overlap "
                             "with whatever windows search used")
    parser.add_argument("--max_windows", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))
    mcfg = ModelConfig(**cfg["model"])
    ctx_len = mcfg.ctx_len

    model = GPT(mcfg).to(device)
    state = {k: v.float() for k, v in load_file(args.ckpt).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer_dir)
    full_stream = val_stream(args.val_shards)
    byte_table = token_byte_table(tokenizer)

    offset_tokens = args.window_offset * ctx_len
    disjoint_stream = full_stream[offset_tokens:]
    print(f"Full stream length: {len(full_stream)} tokens; "
          f"skipping first {args.window_offset} windows "
          f"({offset_tokens} tokens); disjoint stream length: "
          f"{len(disjoint_stream)} tokens", flush=True)

    masks = json.load(open(args.masks_json))

    model.cfg.execution_mode = "sequential"
    seq_result = evaluate_bpb(model, disjoint_stream, byte_table, ctx_len,
                              batch_size=32, max_windows=args.max_windows,
                              device=str(device))
    seq_bpb = seq_result["val_bpb"]
    print(f"all-sequential BPB on disjoint windows: {seq_bpb:.4f} "
          f"({seq_result['windows']} windows)", flush=True)

    results = {"window_offset": args.window_offset,
              "max_windows": args.max_windows,
              "seq_bpb_disjoint": seq_bpb,
              "per_mask": {}}
    for name, mask in masks.items():
        model.cfg.execution_mode = mask
        r = evaluate_bpb(model, disjoint_stream, byte_table, ctx_len,
                        batch_size=32, max_windows=args.max_windows,
                        device=str(device))
        degradation = r["val_bpb"] - seq_bpb
        print(f"  {name}: disjoint_bpb={r['val_bpb']:.4f} "
              f"degradation={degradation:+.4f}", flush=True)
        results["per_mask"][name] = {
            "mask": mask, "disjoint_bpb": r["val_bpb"],
            "disjoint_degradation": degradation, "windows": r["windows"],
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.output, "w"), indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
