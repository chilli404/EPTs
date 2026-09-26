"""Bits-per-byte evaluation, matching nanochat's evaluate_bpb definition:
sum of per-token NLL (nats) over targets with byte length > 0, divided by
ln(2) * total UTF-8 bytes of those targets. Special tokens (byte count 0)
are masked from numerator and denominator."""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def token_byte_table(tokenizer) -> torch.Tensor:
    n = tokenizer.get_vocab_size()
    tb = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        # ByteLevel decoder maps each token to its exact byte string;
        # special tokens decode to "" (skip_special_tokens default) -> 0
        tb[i] = len(tokenizer.decode([i]).encode("utf-8"))
    return tb


def val_stream(shard_dir: str | Path) -> np.ndarray:
    shards = sorted(Path(shard_dir).glob("shard_*.bin"))
    if not shards:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    return np.concatenate([np.fromfile(p, dtype=np.uint16) for p in shards])


@torch.no_grad()
def evaluate_bpb(model, stream: np.ndarray, token_bytes: torch.Tensor,
                 ctx_len: int, batch_size: int = 16,
                 max_windows: int | None = None, device: str = "cpu") -> dict:
    """Sequential non-overlapping windows over the token stream (deterministic)."""
    n_win = (len(stream) - 1) // ctx_len
    if max_windows:
        n_win = min(n_win, max_windows)
    _is_cuda = device.startswith("cuda")
    token_bytes = token_bytes.to(device)
    total_nats = torch.zeros((), dtype=torch.float64)
    total_bytes = torch.zeros((), dtype=torch.long)
    total_tokens = 0
    for start in range(0, n_win, batch_size):
        idxs = range(start, min(start + batch_size, n_win))
        wins = np.stack([stream[i * ctx_len: i * ctx_len + ctx_len + 1]
                         for i in idxs]).astype(np.int64)
        t = torch.from_numpy(wins).to(device)
        x, y = t[:, :-1], t[:, 1:]
        with torch.autocast(device_type=device.split(":")[0],
                            dtype=torch.bfloat16, enabled=_is_cuda):
            logits = model(x)
        nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                              y.reshape(-1), reduction="none")
        nb = token_bytes[y.reshape(-1)]
        total_nats += (nll * (nb > 0)).sum().cpu().double()
        total_bytes += nb.sum().cpu()
        total_tokens += int((nb > 0).sum())
    bpb = total_nats.item() / (math.log(2) * total_bytes.item())
    return {"val_bpb": bpb, "windows": n_win, "tokens_scored": total_tokens,
            "bytes": int(total_bytes.item()),
            "mean_nll_nats": total_nats.item() / max(total_tokens, 1)}


@torch.no_grad()
def evaluate_bpb_docs(model, stream: np.ndarray, token_bytes: torch.Tensor,
                      ctx_len: int, batch_size: int = 16,
                      max_docs: int | None = None, eot_id: int = 0,
                      device: str = "cpu") -> dict:
    """One document per window starting at position 0, EOT-padded.

    In-distribution protocol for one-doc-trained models (packed windows put
    docs at arbitrary positions after EOTs such models never saw). EOT
    targets have 0 bytes, so padding drops out of both sums automatically.
    Docs longer than ctx_len+1 are truncated.
    """
    ends = np.where(stream == eot_id)[0]
    starts = np.concatenate(([0], ends[:-1] + 1))
    if max_docs:
        starts, ends = starts[:max_docs], ends[:max_docs]
    _is_cuda = device.startswith("cuda")
    token_bytes = token_bytes.to(device)
    total_nats = torch.zeros((), dtype=torch.float64)
    total_bytes = torch.zeros((), dtype=torch.long)
    total_tokens = 0
    for b0 in range(0, len(starts), batch_size):
        rows = np.full((min(batch_size, len(starts) - b0), ctx_len + 1),
                       eot_id, dtype=np.int64)
        for i in range(rows.shape[0]):
            s, e = starts[b0 + i], ends[b0 + i]
            seg = stream[s:e + 1][:ctx_len + 1].astype(np.int64)
            rows[i, :len(seg)] = seg
        t = torch.from_numpy(rows).to(device)
        x, y = t[:, :-1], t[:, 1:]
        with torch.autocast(device_type=device.split(":")[0],
                            dtype=torch.bfloat16, enabled=_is_cuda):
            logits = model(x)
        nll = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                              y.reshape(-1), reduction="none")
        nb = token_bytes[y.reshape(-1)]
        total_nats += (nll * (nb > 0)).sum().cpu().double()
        total_bytes += nb.sum().cpu()
        total_tokens += int((nb > 0).sum())
    bpb = total_nats.item() / (math.log(2) * total_bytes.item())
    return {"val_bpb": bpb, "protocol": "doc", "docs": len(starts),
            "tokens_scored": total_tokens, "bytes": int(total_bytes.item()),
            "mean_nll_nats": total_nats.item() / max(total_tokens, 1)}
