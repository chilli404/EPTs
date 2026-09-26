"""One active-learning round: trust-region search around real archive
winners + boundary/exploration diversity, scored by the conservative
(q0+q95) surrogate, then a single batch of real GPU validation. Tests
directly whether closing the loop even once -- generating proposals from
where real data already exists, rather than blind random restarts or one
big offline surrogate search -- finds something genuinely better than the
plain black-box search's own best, using far fewer real evaluations.

Batch composition (per the adopted design):
  50% exploitation -- mutations (1-3 layers) around the archive's current
    best feasible masks, ranked by conservative-gated predicted speedup.
  25% boundary -- candidates near the feasibility edge (conservative
    prediction close to the floor) with the largest mean/q95 gap (a proxy
    for "the surrogate is least sure here") -- exactly where more real data
    is most valuable to acquire.
  25% exploration -- genuinely random masks, for diversity / to catch
    anything structurally novel the trust regions would never generate.

Usage:
  python scripts/eval_6mode_active_loop.py \
      --archive blackwell/results/6mode_archive_430m.json \
      --surrogate blackwell/results/6mode_surrogate_430m_v2.pkl \
      --bpb_floor 0.02 --batch_size 24 \
      --ckpt .../430m_6mode_lt/ckpts/step012000.safetensors \
      --config configs/scale430m_6mode_losstarget.yaml \
      --val_shards data/climbmix/bpe8192/shards --tokenizer_dir data/climbmix/bpe8192 \
      --output blackwell/results/6mode_active_loop_430m.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

MODES = ["sequential", "parallel", "skip", "reverse", "attn_only", "ffn_only"]


def mask_to_features(mask):
    n_layers = len(mask)
    onehot = np.zeros(n_layers * len(MODES))
    counts = np.zeros(len(MODES))
    for l, m in enumerate(mask):
        idx = MODES.index(m)
        onehot[l * len(MODES) + idx] = 1.0
        counts[idx] += 1
    return np.concatenate([onehot, counts])


def score_batch(surrogate, masks):
    from fogen.execution_graph import evaluate_true_cost
    X = np.array([mask_to_features(m) for m in masks])
    q0 = np.array([evaluate_true_cost(m, surrogate["q0_base_costs"], surrogate["alt_modes"],
                                      surrogate["q0_alpha"], surrogate["q0_beta"], surrogate["q0_gamma"],
                                      seq_penalty=surrogate["q0_seq_penalty"]) for m in masks])
    mean_pred = q0 + surrogate["bpb_residual_model"].predict(X)
    safe_pred = q0 + surrogate["bpb_residual_q95_model"].predict(X)
    lat_pred = surrogate["latency_model"].predict(X)
    return mean_pred, safe_pred, lat_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--surrogate", required=True)
    parser.add_argument("--objective", choices=["speed", "quality"], default="speed",
                        help="'speed' (default): seed from the archive's "
                             "FASTEST feasible real masks, rank candidates by "
                             "predicted latency subject to the bpb_floor. "
                             "'quality': the opposite objective -- seed from "
                             "the archive's BEST-QUALITY (most negative "
                             "bpb_degradation) real masks instead, rank "
                             "candidates by predicted BPB IMPROVEMENT "
                             "(most negative predicted bpb) with no latency "
                             "constraint -- directly answers 'how much can "
                             "quality improve locally around real winners', "
                             "the same bug fixed in eval_6mode_blackbox_"
                             "search.py's --objective quality applied to (C)")
    parser.add_argument("--bpb_floor", type=float, default=None,
                        help="required for --objective speed; ignored for "
                             "--objective quality (no floor -- quality IS "
                             "the objective, not a constraint)")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--mutations_per_seed", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val_shards", required=True)
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--max_windows", type=int, default=64)
    parser.add_argument("--reference_bpb", type=float, default=None,
                        help="real val_bpb of a DIFFERENT reference checkpoint "
                             "(e.g. the dedicated sequential specialist). If "
                             "given together with --known_seq_bpb, --bpb_floor "
                             "is applied to degradation vs THIS value instead "
                             "of the search checkpoint's own sequential mode "
                             "(per C24/C25: comparing against a checkpoint's "
                             "own possibly-degraded sequential mode is not the "
                             "same question as comparing against a properly-"
                             "trained specialist) -- also always reported per-"
                             "result as real_bpb_degradation_vs_reference.")
    parser.add_argument("--known_seq_bpb", type=float, default=None,
                        help="the search checkpoint's own real sequential-mode "
                             "val_bpb, measured previously -- required together "
                             "with --reference_bpb to shift the floor (the "
                             "archive/surrogate work in own-sequential-relative "
                             "units natively and don't store an absolute "
                             "seq_bpb, so this must be supplied explicitly "
                             "rather than re-derived from the archive).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"=== {Path(__file__).name} config ===")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=" * 40, flush=True)

    if args.objective == "speed" and args.bpb_floor is None:
        raise ValueError("--bpb_floor is required for --objective speed")
    if (args.reference_bpb is None) != (args.known_seq_bpb is None):
        raise ValueError("--reference_bpb and --known_seq_bpb must be given together")

    # bpb_degradation_vs_reference = bpb_degradation + (known_seq_bpb - reference_bpb)
    # (both measure the same absolute val_bpb, just relative to a different
    # baseline) -- so an equivalent floor in the surrogate/archive's native
    # own-sequential-relative units is bpb_floor - known_seq_bpb + reference_bpb.
    # Shifting the floor instead of the predictions avoids needing the
    # surrogate to predict anything it wasn't trained to predict.
    effective_bpb_floor = args.bpb_floor
    if args.reference_bpb is not None and args.bpb_floor is not None:
        effective_bpb_floor = args.bpb_floor - args.known_seq_bpb + args.reference_bpb
        print(f"--reference_bpb given: shifting bpb_floor from {args.bpb_floor} "
              f"to {effective_bpb_floor:.4f} in the search checkpoint's own "
              f"(own-sequential-relative) units")

    with open(args.surrogate, "rb") as f:
        surrogate = pickle.load(f)
    n_layers = surrogate["n_layers"]
    archive = json.load(open(args.archive))
    rng = np.random.default_rng(args.seed)

    if args.objective == "quality":
        # Opposite of the speed objective: seed from the BEST-QUALITY real
        # masks (most negative bpb_degradation), not the fastest feasible
        # ones -- there is no floor to respect here, quality improvement IS
        # the objective, not a constraint.
        with_bpb = [e for e in archive["entries"] if e["bpb_degradation"] is not None]
        with_bpb.sort(key=lambda e: e["bpb_degradation"])
        seeds = [e["mask"] for e in with_bpb[:args.n_seeds]]
        print(f"Trust-region seeds ({len(seeds)}, best-quality real masks, "
              f"from {len(with_bpb)} archive entries with real BPB):")
        for e in with_bpb[:args.n_seeds]:
            print(f"  bpb_deg={e['bpb_degradation']:+.4f}")
    else:
        # Trust-region seeds: current best-known REAL feasible masks.
        feasible_real = [e for e in archive["entries"]
                         if e["bpb_degradation"] is not None and e["bpb_degradation"] <= effective_bpb_floor
                         and e.get("latency_ms") is not None]
        feasible_real.sort(key=lambda e: e["latency_ms"])
        seeds = [e["mask"] for e in feasible_real[:args.n_seeds]]
        print(f"Trust-region seeds ({len(seeds)}, from {len(feasible_real)} feasible archive entries):")
        for e in feasible_real[:args.n_seeds]:
            print(f"  latency={e['latency_ms']:.2f}ms bpb_deg={e['bpb_degradation']:+.4f}")
        if not seeds:
            # Real, informative case (not a bug): if the checkpoint's own
            # gap to the reference is already larger than the nominal
            # floor, the shifted effective_bpb_floor goes negative --
            # meaning no mask can be feasible without ALREADY being a
            # genuine improvement over the checkpoint's own sequential mode
            # deeper than the shift requires. Report this cleanly instead
            # of crashing on an empty candidate batch downstream.
            if args.reference_bpb is not None:
                reason = (f"Either effective_bpb_floor<=0 (the checkpoint's own "
                         f"gap to the reference, "
                         f"{args.known_seq_bpb - args.reference_bpb:.4f}, exceeds "
                         f"the requested floor -- no mask could possibly be "
                         f"feasible without already improving on the checkpoint's "
                         f"own sequential mode), OR the archive's latency-labeled "
                         f"subset (which can be much sparser than its full real-"
                         f"measurement set) simply doesn't happen to cover this "
                         f"specific range.")
            else:
                reason = (f"The archive's latency-labeled subset (which can be "
                         f"much sparser than its full real-measurement set) "
                         f"simply doesn't happen to have any entry with "
                         f"bpb_degradation <= {effective_bpb_floor:.4f} in the "
                         f"checkpoint's own units -- check n entries with "
                         f"latency vs n total in the archive.")
            print(f"\nNo seeds found -- effective_bpb_floor={effective_bpb_floor:.4f} "
                  f"(own-sequential-relative units). {reason}")
            output = {"checkpoint": args.ckpt, "bpb_floor": args.bpb_floor,
                      "effective_bpb_floor": effective_bpb_floor,
                      "known_seq_bpb": args.known_seq_bpb,
                      "reference_bpb": args.reference_bpb, "results": [],
                      "infeasible_reason": "checkpoint's own gap to reference exceeds requested floor"}
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            json.dump(output, open(args.output, "w"), indent=2)
            print(f"Saved (empty) result to {args.output}")
            return

    # 50% exploitation: mutations around seeds.
    exploit_candidates = []
    for seed_mask in seeds:
        for _ in range(args.mutations_per_seed):
            m = list(seed_mask)
            n_mut = rng.integers(1, 4)
            for layer in rng.choice(n_layers, size=n_mut, replace=False):
                m[layer] = rng.choice(MODES)
            exploit_candidates.append(m)

    mean_pred, safe_pred, lat_pred = score_batch(surrogate, exploit_candidates)
    n_exploit = args.batch_size // 2
    n_boundary = args.batch_size // 4

    if args.objective == "quality":
        # Rank by predicted BPB improvement directly (most negative mean
        # prediction first) -- no feasibility gate, no latency involved.
        exploit_order = np.argsort(mean_pred)
        exploit_batch = [exploit_candidates[i] for i in exploit_order[:n_exploit]]
        # Boundary: largest (safe_pred - mean_pred) gap among the most
        # promising (lowest mean-predicted) candidates -- same "surrogate
        # least sure here" proxy, restricted to the region we actually care
        # about (candidates that look like real improvements).
        promising = mean_pred <= np.percentile(mean_pred, 50)
        uncertainty = safe_pred - mean_pred
        boundary_order = np.argsort(-np.where(promising, uncertainty, -np.inf))
        boundary_batch = [exploit_candidates[i] for i in boundary_order[:n_boundary]]
    else:
        feasible_mask = safe_pred <= effective_bpb_floor
        n_conservatively_feasible = int(feasible_mask.sum())
        if n_conservatively_feasible > 0:
            exploit_order = np.argsort(np.where(feasible_mask, lat_pred, np.inf))
            exploit_batch = [exploit_candidates[i] for i in exploit_order[:n_exploit]]
        else:
            # Real finding this session: with the properly-fixed conservative
            # gate, small mutations around known-good real seeds can ALL come
            # back predicted-infeasible -- falling back to an all-inf ranking
            # would pick an arbitrary, uninformative subset. Instead fall back
            # to "closest to the boundary" (least-infeasible), which is both a
            # sensible exploitation attempt AND the most informative thing to
            # measure if the conservative bound turns out to be too loose here.
            print(f"  [exploit] 0/{len(exploit_candidates)} conservatively feasible near seeds -- "
                  f"falling back to closest-to-boundary instead of an arbitrary pick")
            exploit_order = np.argsort(safe_pred)  # smallest (least infeasible) first
            exploit_batch = [exploit_candidates[i] for i in exploit_order[:n_exploit]]

        # 25% boundary: from the SAME exploit pool, largest (safe_pred - mean_pred)
        # gap among near-feasible candidates -- proxy for "surrogate least sure here".
        near_floor = np.abs(safe_pred - effective_bpb_floor) < 0.5 * abs(effective_bpb_floor)
        uncertainty = safe_pred - mean_pred
        boundary_order = np.argsort(-np.where(near_floor, uncertainty, -np.inf))
        boundary_batch = [exploit_candidates[i] for i in boundary_order[:n_boundary]]

    # 25% exploration: genuinely random.
    n_explore = args.batch_size - n_exploit - n_boundary
    explore_batch = [[rng.choice(MODES) for _ in range(n_layers)] for _ in range(n_explore)]

    batch = exploit_batch + boundary_batch + explore_batch
    print(f"\nBatch: {len(exploit_batch)} exploitation + {len(boundary_batch)} boundary + "
          f"{len(explore_batch)} exploration = {len(batch)} candidates for real validation")

    # Real validation.
    import torch
    import yaml
    from safetensors.torch import load_file
    from fogen.data import load_tokenizer
    from fogen.evals.bpb import evaluate_bpb, token_byte_table, val_stream
    from fogen.model import GPT, ModelConfig

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

    baseline_ms = time_forward("sequential")
    with torch.no_grad():
        sequential_logits = model(sample, mode="sequential").float()
    sequential_argmax = sequential_logits.argmax(dim=-1)
    print(f"\nseq_bpb={seq_bpb:.4f} baseline_ms={baseline_ms:.3f}")

    results = []
    for i, mask in enumerate(batch):
        bucket = "exploit" if i < len(exploit_batch) else ("boundary" if i < len(exploit_batch) + len(boundary_batch) else "explore")
        model.cfg.execution_mode = mask
        real_bpb = evaluate_bpb(model, stream, byte_table, context, batch_size=32,
                                max_windows=args.max_windows, device=str(device))["val_bpb"]
        real_lat = time_forward(mask)
        with torch.no_grad():
            logits = model(sample, mode=mask).float()
        real_agree = float((logits.argmax(dim=-1) == sequential_argmax).float().mean())
        real_deg = real_bpb - seq_bpb
        speedup = baseline_ms / real_lat
        entry = {"mask": mask, "bucket": bucket, "real_bpb_absolute": real_bpb,
                "real_bpb_degradation": real_deg,
                "real_latency_ms": real_lat, "real_speedup": speedup, "real_agreement": real_agree}
        msg = f"  [{bucket:>8}] real: bpb={real_deg:+.4f} lat={real_lat:.2f}ms speedup={speedup:.3f}x agree={real_agree:.4f}"
        if args.reference_bpb is not None:
            deg_vs_ref = real_bpb - args.reference_bpb
            entry["real_bpb_degradation_vs_reference"] = deg_vs_ref
            msg += f"  |  vs specialist: bpb={deg_vs_ref:+.4f}"
        results.append(entry)
        print(msg, flush=True)

    print(f"\n=== Comparison ===")
    if args.objective == "quality":
        best_new = min(results, key=lambda r: r["real_bpb_degradation"])
        best_archive_bpb = with_bpb[0]["bpb_degradation"] if with_bpb else None
        if best_archive_bpb is not None:
            print(f"  archive's best real BPB (before this round): {best_archive_bpb:+.4f}")
        print(f"  best NEW real bpb this round: {best_new['real_bpb_degradation']:+.4f} "
              f"speedup={best_new['real_speedup']:.3f}x agree={best_new['real_agreement']:.4f} "
              f"(bucket={best_new['bucket']})")
        if best_archive_bpb is not None:
            print(f"  IMPROVED ARCHIVE: {best_new['real_bpb_degradation'] < best_archive_bpb}")
        if args.reference_bpb is not None:
            print(f"  best NEW absolute bpb={best_new['real_bpb_absolute']:.4f} vs "
                  f"specialist={args.reference_bpb:.4f}: "
                  f"{best_new['real_bpb_degradation_vs_reference']:+.4f} "
                  f"({'BEATS' if best_new['real_bpb_degradation_vs_reference'] < 0 else 'still behind'} specialist)")
        n_conservative_correct = None  # no feasibility gate in quality mode
    else:
        feasible_results = [r for r in results if r["real_bpb_degradation"] <= effective_bpb_floor]
        best_new = max(feasible_results, key=lambda r: r["real_speedup"]) if feasible_results else None
        best_archive = feasible_real[0] if feasible_real else None
        if best_archive:
            print(f"  archive's best (before this round): speedup={baseline_ms/best_archive['latency_ms']:.3f}x "
                  f"bpb_deg={best_archive['bpb_degradation']:+.4f}")
        if best_new:
            print(f"  best NEW real-feasible candidate this round: speedup={best_new['real_speedup']:.3f}x "
                  f"bpb_deg={best_new['real_bpb_degradation']:+.4f} (bucket={best_new['bucket']})")
            improved = best_archive is None or best_new["real_speedup"] > baseline_ms / best_archive["latency_ms"]
            print(f"  IMPROVED ARCHIVE: {improved}")
            if args.reference_bpb is not None:
                print(f"  best NEW absolute bpb={best_new['real_bpb_absolute']:.4f} vs "
                      f"specialist={args.reference_bpb:.4f}: "
                      f"{best_new['real_bpb_degradation_vs_reference']:+.4f} "
                      f"({'BEATS' if best_new['real_bpb_degradation_vs_reference'] < 0 else 'still behind'} specialist)")
        else:
            print("  no real-feasible candidate found in this batch")
        n_conservative_correct = sum(1 for r in results[:len(exploit_batch)] if r["real_bpb_degradation"] <= effective_bpb_floor)
        print(f"\n  exploitation batch: {n_conservative_correct}/{len(exploit_batch)} genuinely met the "
              f"real BPB floor (checks whether the conservative gate fix actually worked)")

    output = {"checkpoint": args.ckpt, "archive": args.archive, "surrogate": args.surrogate, "objective": args.objective,
              "bpb_floor": args.bpb_floor, "effective_bpb_floor": effective_bpb_floor, "seq_bpb": seq_bpb, "baseline_ms": baseline_ms, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w"), indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
