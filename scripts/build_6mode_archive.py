"""Program archive: amortize the expensive quality search into a validated,
reusable set of real (mask -> bpb_degradation) measurements, decoupled from
any specific hardware. BPB quality is hardware-independent; latency is not
(established repeatedly this session, e.g. F63 onward) -- so the expensive
part (finding which masks are actually good) only needs doing once per
checkpoint, and picking the fastest one for a NEW GPU only needs a latency
re-measurement on the archive's masks, not a full re-search.

This reuses build_6mode_surrogate.py's aggregation step (same data, same
de-duplication), but the deliverable here is a filtered, ranked, reusable
archive -- not a trained model.

Usage:
  python scripts/build_6mode_archive.py --scale 430m --n_layers 20 \
      --results_dir blackwell/results --bpb_floor 0.02 \
      --output blackwell/results/6mode_archive_430m.json

  # then, on any hardware with latency data for archive masks (or after a
  # fresh re-measurement pass):
  python scripts/build_6mode_archive.py --select_best \
      --archive blackwell/results/6mode_archive_430m.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from build_6mode_surrogate import extract_rows
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", required=False)
    parser.add_argument("--n_layers", type=int, required=False)
    parser.add_argument("--checkpoint", default=None,
                        help="exact source checkpoint path -- files whose "
                             "recorded checkpoint differs are excluded (real "
                             "bug found and fixed this session: filename-"
                             "pattern matching silently mixed measurements "
                             "from different checkpoints sharing a scale "
                             "name, e.g. 430m_6mode_lt vs 430m_6mode_ramp)")
    parser.add_argument("--results_dir", default="results/six_mode")
    parser.add_argument("--bpb_floor", type=float, default=None,
                        help="only keep masks with real bpb_degradation <= this")
    parser.add_argument("--output")
    parser.add_argument("--select_best", action="store_true",
                        help="load an existing archive (--archive) and print "
                             "the best (lowest latency) feasible entry -- the "
                             "cheap hardware-adaptation step, no new search")
    parser.add_argument("--archive")
    parser.add_argument("--select_bpb_floor", type=float, default=None,
                        help="tighter floor for selection than the archive's "
                             "own build-time floor, if desired")
    args = parser.parse_args()

    if args.select_best:
        archive = json.load(open(args.archive))
        entries = archive["entries"]
        floor = args.select_bpb_floor if args.select_bpb_floor is not None else archive["bpb_floor"]
        if floor is None:
            raise ValueError("--select_bpb_floor required: the archive was built without --bpb_floor, "
                             "so there is no default quality budget to select against")
        feasible = [e for e in entries if e["bpb_degradation"] is not None and e["bpb_degradation"] <= floor
                    and e.get("latency_ms") is not None]
        if not feasible:
            print(f"No archived entries with real latency data and bpb_degradation <= {floor}")
            return
        best = min(feasible, key=lambda e: e["latency_ms"])
        mc = {m: best["mask"].count(m) for m in ["sequential","parallel","skip","reverse","attn_only","ffn_only"] if best["mask"].count(m) > 0}
        print(f"Best archived entry (n_feasible_with_latency={len(feasible)}/{len(entries)}):")
        print(f"  bpb_degradation={best['bpb_degradation']:+.4f} latency_ms={best['latency_ms']:.2f} "
              f"agreement={best.get('argmax_agreement')} modes={mc}")
        return

    if not (args.scale and args.n_layers and args.output):
        raise ValueError("--scale, --n_layers, and --output required unless --select_best")

    pattern = f"{args.results_dir}/6mode_*{args.scale}*.json"
    files = sorted(glob.glob(pattern))
    print(f"Scanning {len(files)} files matching '{pattern}'...")

    all_rows = []
    seen = {}  # mask tuple -> index into all_rows
    for f in files:
        for mask, bpb_deg, lat, agree in extract_rows(f, args.n_layers, checkpoint=args.checkpoint):
            key = tuple(mask)
            if key in seen:
                # Enrich, don't skip -- real bug found and fixed this
                # session: a later occurrence of the same mask with
                # latency data was silently dropped because de-dup kept
                # only the first-seen (latency-less) entry. Fill in any
                # field the existing entry is missing.
                existing = all_rows[seen[key]]
                if existing["latency_ms"] is None and lat is not None:
                    existing["latency_ms"] = lat
                if existing["argmax_agreement"] is None and agree is not None:
                    existing["argmax_agreement"] = agree
                continue
            seen[key] = len(all_rows)
            all_rows.append({"mask": mask, "bpb_degradation": bpb_deg, "latency_ms": lat, "argmax_agreement": agree})

    print(f"Total unique real masks: {len(all_rows)}")
    if args.bpb_floor is not None:
        kept = [r for r in all_rows if r["bpb_degradation"] is not None and r["bpb_degradation"] <= args.bpb_floor]
        print(f"Quality-feasible (bpb_degradation <= {args.bpb_floor}): {len(kept)}")
    else:
        kept = all_rows
    n_with_lat = sum(1 for r in kept if r["latency_ms"] is not None)
    print(f"  of which have real latency measured: {n_with_lat}")

    output = {"scale": args.scale, "n_layers": args.n_layers, "bpb_floor": args.bpb_floor,
              "n_source_files": len(files), "entries": kept}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w"), indent=2)
    print(f"Saved archive to {args.output}")


if __name__ == "__main__":
    main()
