import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file
from scipy.stats import spearmanr

from fogen.data import load_tokenizer
from fogen.evals.bpb import evaluate_bpb, token_byte_table, val_stream
from fogen.model import GPT, ModelConfig


def load_model(checkpoint, config, device):
    cfg = yaml.safe_load(open(config))
    model = GPT(ModelConfig(**cfg["model"])).to(device)
    state = {key: value.float() for key, value in load_file(checkpoint).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert all(key.startswith("rope_") for key in missing)
    return model.eval(), cfg


def graph_masks(n_layers, layer_defects, num_random):
    if 2**n_layers <= 256:
        return list(itertools.product((0, 1), repeat=n_layers))
    masks = {tuple([0] * n_layers), tuple([1] * n_layers)}
    for layer in range(n_layers):
        masks.add(tuple(int(index == layer) for index in range(n_layers)))
        masks.add(tuple(int(index != layer) for index in range(n_layers)))
    masks.add(tuple(index % 2 for index in range(n_layers)))
    masks.add(tuple((index + 1) % 2 for index in range(n_layers)))
    order = np.argsort(layer_defects)
    for count in range(1, n_layers):
        low = np.zeros(n_layers, dtype=int)
        high = np.zeros(n_layers, dtype=int)
        low[order[:count]] = 1
        high[order[-count:]] = 1
        masks.add(tuple(low.tolist()))
        masks.add(tuple(high.tolist()))
        masks.add(tuple([1] * count + [0] * (n_layers - count)))
        masks.add(tuple([0] * (n_layers - count) + [1] * count))
    rng = np.random.default_rng(2026)
    for _ in range(num_random):
        masks.add(tuple(rng.integers(0, 2, size=n_layers).tolist()))
    return sorted(masks, key=lambda mask: (sum(mask), mask))


def _sync(device):
    # perf_counter around an async device call under-measures unless the
    # queue is drained first -- true for both CUDA and MPS. The CUDA-only
    # check here previously made every MPS latency number silently wrong.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def latency(model, inputs, mask, warmup=5, repeats=20):
    with torch.no_grad(), torch.autocast(
        device_type=inputs.device.type,
        dtype=torch.bfloat16,
        enabled=inputs.device.type == "cuda",
    ):
        for _ in range(warmup):
            model(inputs, mode=mask)
        _sync(inputs.device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(inputs, mode=mask)
        _sync(inputs.device)
    return (time.perf_counter() - started) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_windows", type=int, default=128)
    parser.add_argument("--num_random_graphs", type=int, default=64)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model, cfg = load_model(args.ckpt, args.config, device)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    stream = val_stream(args.val_shards)
    byte_table = token_byte_table(tokenizer)
    context = cfg["model"]["ctx_len"]
    sample = torch.tensor(
        np.asarray(stream[:2 * context], dtype=np.int64).reshape(2, context),
        device=device,
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        sequential_logits = model(sample, mode="sequential").float()
        layer_defects = model.layer_defects(sample).float().cpu().numpy()
    sequential_logprob = F.log_softmax(sequential_logits, dim=-1)
    sequential_probability = sequential_logprob.exp()
    rows = []
    masks = graph_masks(
        cfg["model"]["n_layer"], layer_defects, args.num_random_graphs)
    print(f"Evaluating {len(masks)} graph configurations...", flush=True)
    for graph_idx, bits in enumerate(masks):
        mask = ["parallel" if bit else "sequential" for bit in bits]
        model.cfg.execution_mode = mask
        bpb = evaluate_bpb(
            model,
            stream,
            byte_table,
            context,
            batch_size=32,
            max_windows=args.max_windows,
            device=str(device),
        )["val_bpb"]
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(sample, mode=mask).float()
        logprob = F.log_softmax(logits, dim=-1)
        # Per-token symmetric KL in nats (batchmean = sum over vocab, mean over positions)
        symmetric_kl = (
            F.kl_div(logprob, sequential_probability, reduction="batchmean")
            + F.kl_div(sequential_logprob, logprob.exp(), reduction="batchmean")
        ) / 2
        selected_defects = layer_defects[np.asarray(bits, dtype=bool)]
        predicted_defect = float(np.sqrt(np.sum(selected_defects**2)))
        row = {
            "bits": list(bits),
            "mask": mask,
            "parallel_layers": int(sum(bits)),
            "predicted_defect": predicted_defect,
            "val_bpb": bpb,
            "symmetric_kl": float(symmetric_kl),
            "argmax_agreement": float(
                (logits.argmax(dim=-1) == sequential_logits.argmax(dim=-1)).float().mean()
            ),
            "latency_seconds": latency(model, sample, mask),
        }
        rows.append(row)
        print(f"  [{graph_idx+1}/{len(masks)}] {sum(bits)}/L par | "
              f"bpb={bpb:.4f} kl={float(symmetric_kl):.4f} "
              f"agree={row['argmax_agreement']:.3f}", flush=True)
    sequential_bpb = next(row["val_bpb"] for row in rows if sum(row["bits"]) == 0)
    for row in rows:
        row["bpb_degradation"] = row["val_bpb"] - sequential_bpb
    nonzero = [row for row in rows if row["parallel_layers"] > 0]
    result = {
        "checkpoint": args.ckpt,
        "layer_defects": layer_defects.tolist(),
        "rows": rows,
        "spearman_defect_vs_bpb": float(spearmanr(
            [row["predicted_defect"] for row in nonzero],
            [row["bpb_degradation"] for row in nonzero],
        ).statistic),
        "spearman_defect_vs_kl": float(spearmanr(
            [row["predicted_defect"] for row in nonzero],
            [row["symmetric_kl"] for row in nonzero],
        ).statistic),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
