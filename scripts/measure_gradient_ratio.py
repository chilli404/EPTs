"""Measure consistency-to-LM gradient ratio across model scales.

For each checkpoint, computes:
  R = ||∇L_consistency|| / ||∇L_LM||

where L_LM = (1/2)(L_seq + L_par) and L_consistency is the centered-logit MSE.
Also measures these separately for KL consistency to explain the KL failure.

Usage:
  python scripts/measure_gradient_ratio.py \
      --ckpt runs/430m_poly_full/ckpts/step012000.safetensors \
      --config blackwell/configs/scale430m_poly_full.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output results/gradient_ratio_430m.json

  # Or measure at multiple checkpoints during training:
  python scripts/measure_gradient_ratio.py \
      --ckpt_dir runs/430m_poly_full/ckpts \
      --config blackwell/configs/scale430m_poly_full.yaml \
      --val_shards data/climbmix/bpe8192/shards \
      --tokenizer_dir data/climbmix/bpe8192 \
      --output results/gradient_ratios_430m_over_training.json
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.data import ShardedLoader, load_tokenizer
from fogen.model import GPT, ModelConfig


def grad_norm_of(params, loss):
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.detach().float().norm().item() ** 2
    return total ** 0.5


def _forward_and_losses(model, batch_x, batch_y, device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        seq_logits = model(batch_x, mode="sequential")
        par_logits = model(batch_x, mode="parallel")

        seq_loss = F.cross_entropy(
            seq_logits.view(-1, seq_logits.size(-1)), batch_y.reshape(-1))
        par_loss = F.cross_entropy(
            par_logits.view(-1, par_logits.size(-1)), batch_y.reshape(-1))
        lm_loss = 0.5 * (seq_loss + par_loss)

        seq_centered = seq_logits - seq_logits.mean(dim=-1, keepdim=True)
        par_centered = par_logits - par_logits.mean(dim=-1, keepdim=True)
        mse_con = F.mse_loss(par_centered, seq_centered)

        seq_lp = F.log_softmax(seq_logits, dim=-1)
        par_lp = F.log_softmax(par_logits, dim=-1)
        kl_con = (
            F.kl_div(par_lp, seq_lp.exp(), reduction="batchmean")
            + F.kl_div(seq_lp, par_lp.exp(), reduction="batchmean")
        ) / 2

    return lm_loss, mse_con, kl_con


def measure_ratios(model, batch_x, batch_y, device):
    params = [p for p in model.parameters() if p.requires_grad]

    # Compute each gradient separately to avoid retain_graph OOM
    lm_loss, mse_con, kl_con = _forward_and_losses(model, batch_x, batch_y, device)
    lm_val, mse_val, kl_val = float(lm_loss), float(mse_con), float(kl_con)

    g_lm = grad_norm_of(params, lm_loss)
    del lm_loss, mse_con, kl_con
    model.zero_grad(set_to_none=True)

    _, mse_con, _ = _forward_and_losses(model, batch_x, batch_y, device)
    g_mse = grad_norm_of(params, mse_con)
    del mse_con
    model.zero_grad(set_to_none=True)

    _, _, kl_con = _forward_and_losses(model, batch_x, batch_y, device)
    g_kl = grad_norm_of(params, kl_con)
    del kl_con
    model.zero_grad(set_to_none=True)

    return {
        "lm_loss": lm_val,
        "mse_consistency": mse_val,
        "kl_consistency": kl_val,
        "grad_norm_lm": g_lm,
        "grad_norm_mse": g_mse,
        "grad_norm_kl": g_kl,
        "ratio_mse_over_lm": g_mse / max(g_lm, 1e-12),
        "ratio_kl_over_lm": g_kl / max(g_lm, 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_batches", type=int, default=4)
    parser.add_argument("--bf16", action="store_true",
                        help="Load model in bf16 (needed for 7B on 96GB)")
    parser.add_argument("--batch_seqs", type=int, default=None,
                        help="Override batch size (default: min(config, 8))")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = ModelConfig(**cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_dtype = torch.bfloat16 if args.bf16 else torch.float32

    batch = args.batch_seqs or min(cfg.get("batch_seqs", 8), 8)
    loader = ShardedLoader(
        args.val_shards, batch,
        mcfg.ctx_len, seed=999, device=device)

    if args.ckpt_dir:
        ckpts = sorted(Path(args.ckpt_dir).glob("step*.safetensors"))
    elif args.ckpt:
        ckpts = [Path(args.ckpt)]
    else:
        raise ValueError("Provide --ckpt or --ckpt_dir")

    results = []
    for ckpt_path in ckpts:
        step = int(ckpt_path.stem.replace("step", ""))
        print(f"\n=== {ckpt_path.name} (step {step}) ===")

        model = GPT(mcfg).to(device=device, dtype=param_dtype)
        state = load_file(str(ckpt_path))
        missing, unexpected = model.load_state_dict(
            {k: v.to(param_dtype) for k, v in state.items()}, strict=False)
        assert not unexpected
        model.train()

        batch_results = []
        for i in range(args.n_batches):
            x, y = loader.next_batch()
            ratios = measure_ratios(model, x, y, device)
            batch_results.append(ratios)

        import numpy as np
        avg = {}
        for key in batch_results[0]:
            vals = [r[key] for r in batch_results]
            avg[key] = float(np.mean(vals))
            avg[f"{key}_std"] = float(np.std(vals))

        avg["step"] = step
        avg["checkpoint"] = str(ckpt_path)
        avg["n_params"] = model.num_params()
        avg["d_model"] = mcfg.d_model
        avg["n_layer"] = mcfg.n_layer
        results.append(avg)

        print(f"  ||∇LM||={avg['grad_norm_lm']:.4f}  "
              f"||∇MSE||={avg['grad_norm_mse']:.4f}  "
              f"||∇KL||={avg['grad_norm_kl']:.4f}")
        print(f"  R_mse={avg['ratio_mse_over_lm']:.4f}  "
              f"R_kl={avg['ratio_kl_over_lm']:.4f}")

        del model

    output = {
        "config": args.config,
        "n_batches": args.n_batches,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
