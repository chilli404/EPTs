"""Forced-choice minimal-pair scoring.

For each item we compute the continuation logprob of `correct` and
`distractor` given `prefix`, yielding:
  - argmax_acc: 1[logp(correct) > logp(distractor)]   (sequence-level)
  - logprob_diff: length-normalized logp(correct) - logp(distractor)

Model-agnostic: works with fogen GPT or any HF causal LM via a uniform
(encode, logits) interface, so our sweep and Pythia/OLMo use identical code.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class Scorer:
    encode: callable          # str -> list[int]
    logits_fn: callable       # LongTensor (B,T) -> FloatTensor (B,T,V)
    device: str = "cpu"
    batch_size: int = 128

    @torch.no_grad()
    def _cont_logprobs(self, pairs: list[tuple[list[int], list[int]]]) -> list[float]:
        """pairs: (prefix_ids, cont_ids) -> total logprob of cont given prefix."""
        out = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            seqs = [p + c for p, c in chunk]
            maxlen = max(len(s) for s in seqs)
            ids = torch.zeros(len(seqs), maxlen, dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s)
            logits = self.logits_fn(ids.to(self.device)).float()
            logprobs = F.log_softmax(logits, dim=-1)
            for j, (p, c) in enumerate(chunk):
                lp = 0.0
                for k, tok in enumerate(c):
                    lp += logprobs[j, len(p) + k - 1, tok].item()
                out.append(lp)
        return out

    def score_items(self, items: list[dict]) -> list[dict]:
        pairs, meta = [], []
        for it in items:
            p = self.encode(it["prefix"])
            c, d = self.encode(it["correct"]), self.encode(it["distractor"])
            if len(c) == 0 or len(d) == 0 or len(p) == 0:
                continue  # tokenizer produced empty continuation; skip + count
            pairs += [(p, c), (p, d)]
            meta.append((it, len(c), len(d)))
        lps = self._cont_logprobs(pairs)
        rows = []
        for j, (it, lc, ld) in enumerate(meta):
            lp_c, lp_d = lps[2 * j], lps[2 * j + 1]
            rows.append({
                **{k: it[k] for k in ("probe", "category", "item_id",
                                      "template_id", "split")},
                "lp_correct": lp_c, "lp_distractor": lp_d,
                "len_correct": lc, "len_distractor": ld,
                "argmax_acc": int(lp_c > lp_d),
                "logprob_diff": lp_c / lc - lp_d / ld,
            })
        return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Per (probe, split) aggregates."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["probe"], r["split"])].append(r)
    out = []
    for (probe, split), rs in sorted(groups.items()):
        n = len(rs)
        out.append({
            "probe": probe, "split": split, "n": n,
            "category": rs[0]["category"],
            "argmax_acc": sum(r["argmax_acc"] for r in rs) / n,
            "logprob_diff": sum(r["logprob_diff"] for r in rs) / n,
        })
    return out


def load_battery(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fogen_scorer(model, tokenizer, device="cpu", batch_size=128) -> Scorer:
    model.eval().to(device)
    return Scorer(encode=lambda s: tokenizer.encode(s).ids,
                  logits_fn=lambda ids: model(ids),
                  device=device, batch_size=batch_size)


def hf_scorer(model, tokenizer, device="cpu", batch_size=64) -> Scorer:
    model.eval().to(device)
    return Scorer(encode=lambda s: tokenizer(s, add_special_tokens=False)["input_ids"],
                  logits_fn=lambda ids: model(ids).logits,
                  device=device, batch_size=batch_size)
