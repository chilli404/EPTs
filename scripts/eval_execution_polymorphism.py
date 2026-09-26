import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from fogen.model import GPT, ModelConfig


def load_model(checkpoint, config, device):
    cfg = yaml.safe_load(open(config))
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    state = {key: value.float() for key, value in load_file(checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(key.startswith("rope_") for key in missing)
    return model.eval(), cfg


def benchmark(model, inputs, mode, warmup, repeats):
    with torch.no_grad(), torch.autocast(
        device_type=inputs.device.type,
        dtype=torch.bfloat16,
        enabled=inputs.device.type == "cuda",
    ):
        for _ in range(warmup):
            model(inputs, mode=mode)
        if inputs.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(inputs, mode=mode)
        if inputs.device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sequence_lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, args.config, device)
    torch.manual_seed(2026)
    ctx_len = cfg["model"]["ctx_len"]
    sequence_lengths = [l for l in args.sequence_lengths if l <= ctx_len]
    print(f"Model: {model.num_params()/1e6:.1f}M on {device}, ctx={ctx_len}", flush=True)
    print(f"Sequence lengths: {sequence_lengths}", flush=True)
    latency = {}
    for length in sequence_lengths:
        inputs = torch.randint(
            0,
            cfg["model"]["vocab_size"],
            (args.batch_size, length),
            device=device,
        )
        sequential = benchmark(model, inputs, "sequential", args.warmup, args.repeats)
        parallel = benchmark(model, inputs, "parallel", args.warmup, args.repeats)
        parallel_cuda = benchmark(model, inputs, "parallel_cuda", args.warmup, args.repeats)
        parallel_fused = benchmark(model, inputs, "parallel_fused", args.warmup, args.repeats)
        latency[str(length)] = {
            "sequential_seconds": sequential,
            "parallel_seconds": parallel,
            "parallel_cuda_seconds": parallel_cuda,
            "parallel_fused_seconds": parallel_fused,
            "parallel_speedup": sequential / parallel,
            "parallel_cuda_speedup": sequential / parallel_cuda,
            "parallel_fused_speedup": sequential / parallel_fused,
        }
        print(f"  T={length}: seq={sequential*1000:.2f}ms par={parallel*1000:.2f}ms "
              f"fused={parallel_fused*1000:.2f}ms ({sequential/parallel_fused:.3f}x)",
              flush=True)
    agreement_inputs = torch.randint(
        0,
        cfg["model"]["vocab_size"],
        (args.batch_size, min(512, cfg["model"]["ctx_len"])),
        device=device,
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        sequential_logits = model(agreement_inputs, mode="sequential").float()
        parallel_logits = model(agreement_inputs, mode="parallel").float()
        parallel_cuda_logits = model(agreement_inputs, mode="parallel_cuda").float()
        parallel_fused_logits = model(agreement_inputs, mode="parallel_fused").float()
    sequential_logprob = F.log_softmax(sequential_logits, dim=-1)
    parallel_logprob = F.log_softmax(parallel_logits, dim=-1)
    sequential_probability = sequential_logprob.exp()
    parallel_probability = parallel_logprob.exp()
    symmetric_kl = (
        F.kl_div(parallel_logprob, sequential_probability, reduction="batchmean")
        + F.kl_div(sequential_logprob, parallel_probability, reduction="batchmean")
    ) / 2
    agreement = float(
        (sequential_logits.argmax(dim=-1) == parallel_logits.argmax(dim=-1)).float().mean())
    print(f"Agreement: {agreement:.4f} | KL: {float(symmetric_kl):.4f}", flush=True)
    result = {
        "checkpoint": args.ckpt,
        "device": str(device),
        "batch_size": args.batch_size,
        "latency": latency,
        "argmax_agreement": agreement,
        "parallel_cuda_max_logit_difference": float(
            torch.max(torch.abs(parallel_cuda_logits - parallel_logits))
        ),
        "parallel_fused_max_logit_difference": float(
            torch.max(torch.abs(parallel_fused_logits - parallel_logits))
        ),
        "symmetric_kl": float(symmetric_kl),
        "centered_logit_rmse": float(torch.sqrt(torch.mean(
            (
                sequential_logits - sequential_logits.mean(dim=-1, keepdim=True)
                - parallel_logits + parallel_logits.mean(dim=-1, keepdim=True)
            ) ** 2
        ))),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
