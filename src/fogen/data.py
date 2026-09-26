"""Corpus pipeline: BPE tokenizer training, uint16 shard writing, batch loading.

Artifacts land in S3 under data/<corpus>/<tokenizer_rev>/ with a manifest
recording sha256 hashes (prereg requires data provenance).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def train_tokenizer(text_iter, vocab_size: int, out_dir: str | Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(text_iter, trainer)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(path) / "tokenizer.json"))


def write_shards(doc_iter, tokenizer, out_dir: str | Path,
                 shard_tokens: int = 50_000_000, eot_id: int = 0) -> dict:
    """Tokenize docs (appending <|endoftext|>), write uint16 .bin shards + manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buf, shard_idx, total, hashes = [], 0, 0, []

    def flush():
        nonlocal buf, shard_idx
        if not buf:
            return
        arr = np.asarray(buf, dtype=np.uint16)
        p = out_dir / f"shard_{shard_idx:05d}.bin"
        arr.tofile(p)
        hashes.append({"file": p.name, "tokens": len(arr),
                       "sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
        buf, shard_idx = [], shard_idx + 1

    for doc in doc_iter:
        ids = tokenizer.encode(doc).ids
        buf.extend(ids)
        buf.append(eot_id)
        total += len(ids) + 1
        if len(buf) >= shard_tokens:
            flush()
    flush()
    manifest = {"total_tokens": total, "shards": hashes, "dtype": "uint16"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class ShardedLoader:
    """Deterministic random-offset batch sampler over memmapped uint16 shards.

    one_doc_per_seq=True reproduces the naive "one story per context window"
    regime (hypothesized v1 setup): each row starts at a document boundary,
    runs to the document's <|endoftext|>, and the rest of the window is EOT
    padding which IS included in the loss.

    mask_padding=True (H1b) sets padding targets after the first EOT to -100
    so cross_entropy ignores them; the first EOT is still a target (story end).

    max_tokens caps the corpus at the first N tokens of the concatenated
    stream (the data-budget axis of the phase diagram: same step budget over
    less unique data = more epochs).
    """

    def __init__(self, shard_dir: str | Path, batch_size: int, ctx_len: int,
                 seed: int, device: str = "cpu", one_doc_per_seq: bool = False,
                 mask_padding: bool = False, max_tokens: int | None = None,
                 eot_id: int = 0):
        shard_dir = Path(shard_dir)
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r")
                       for p in sorted(shard_dir.glob("shard_*.bin"))]
        if not self.shards:
            raise FileNotFoundError(f"no shards in {shard_dir}")
        self.sizes = np.array([len(s) for s in self.shards], dtype=np.int64)
        if max_tokens is not None:
            budget = max_tokens
            for i, n in enumerate(self.sizes):
                self.sizes[i] = min(int(n), max(0, budget))
                budget -= self.sizes[i]
            keep = self.sizes > ctx_len + 1
            self.shards = [s for s, k in zip(self.shards, keep) if k]
            self.sizes = self.sizes[keep]
            if not self.shards:
                raise ValueError(f"max_tokens={max_tokens} too small")
        self.batch_size, self.ctx_len = batch_size, ctx_len
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.one_doc_per_seq, self.eot_id = one_doc_per_seq, eot_id
        self.mask_padding = mask_padding
        if one_doc_per_seq:
            self.doc_starts = []
            for arr, size in zip(self.shards, self.sizes):
                a = np.asarray(arr)
                starts = np.concatenate(([0], np.where(a == eot_id)[0] + 1))
                self.doc_starts.append(starts[starts < size - 1])

    def _doc_row(self, out: np.ndarray) -> int:
        si = self.rng.choice(len(self.shards),
                             p=self.sizes / self.sizes.sum())
        s, starts = self.shards[si], self.doc_starts[si]
        off = starts[self.rng.integers(0, len(starts))]
        seg = np.asarray(s[off:off + self.ctx_len + 1], dtype=np.int64)
        ends = np.where(seg == self.eot_id)[0]
        n = (ends[0] + 1) if len(ends) else len(seg)
        out[:] = self.eot_id
        out[:n] = seg[:n]
        return n

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        xs = np.empty((self.batch_size, self.ctx_len + 1), dtype=np.int64)
        if self.one_doc_per_seq:
            ns = [self._doc_row(xs[i]) for i in range(self.batch_size)]
            if self.mask_padding:
                ys = xs[:, 1:].copy()
                for i, n in enumerate(ns):
                    ys[i, max(0, n - 1):] = -100
                x = torch.from_numpy(xs[:, :-1]).to(self.device, non_blocking=True)
                y = torch.from_numpy(ys).to(self.device, non_blocking=True)
                return x, y
        else:
            probs = self.sizes / self.sizes.sum()
            for i in range(self.batch_size):
                si = self.rng.choice(len(self.shards), p=probs)
                s, size = self.shards[si], self.sizes[si]
                off = self.rng.integers(0, size - self.ctx_len - 1)
                xs[i] = s[off:off + self.ctx_len + 1]
        t = torch.from_numpy(xs).to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]
