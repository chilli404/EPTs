"""3-panel 3D density-landscape figure (speedup vs BPB degradation) across
Blackwell/H100/L40S for the 1B primary checkpoint, using the full 3000-mask
uniform-random corpus (measured for real on Blackwell) with per-platform
latency estimated from each platform's own real per-mode marginal benefit
(from the pure-mode microbenchmark), since only Blackwell has a full 3000-mask
real per-mask latency measurement. BPB/degradation is checkpoint-only and
identical across platforms; only the speedup axis differs per platform.

Usage:
  python scripts/plot_hardware_landscape_3panel.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

CORPUS = "blackwell/results/6mode_masks_1b_gradnorm_cap10_24k.json"
BENCH_FILES = {
    "Blackwell": "blackwell/results/6mode_latency_1b_gradnorm_cap10_24k_blackwell.json",
    "H100": "blackwell/results/remote_gpu_comparison/6mode_latency_1b_gradnorm_cap10_24k_h100.json",
    "L40S": "blackwell/results/remote_gpu_comparison/6mode_latency_1b_gradnorm_cap10_24k_l40s.json",
}
PANEL_LABELS = ["(a)", "(b)", "(c)"]
N_LAYERS = 28

FONT_TITLE = 20
FONT_LABEL = 16
FONT_TICK = 12
FONT_PANEL = 22


def per_mode_benefit(path):
    d = json.load(open(path))
    baseline = d["baseline_ms"]
    r = {x["mode"]: x for x in d["results"]}
    skip_ns = [(x["n_skip"], baseline - x["latency_ms"]) for x in d["results"] if "n_skip" in x]
    attn_ns = [(x["n_cheap"], baseline - x["latency_ms"]) for x in d["results"]
              if x["mode"].startswith("attn_only") and "n_cheap" in x]
    skip_slope = np.polyfit(*zip(*skip_ns), 1)[0]
    attn_slope = np.polyfit(*zip(*attn_ns), 1)[0]
    return {
        "baseline": baseline,
        "skip": skip_slope,
        "attn_only": attn_slope,
        "parallel": (baseline - r["all_parallel"]["latency_ms"]) / N_LAYERS,
        "reverse": (baseline - r["all_reverse"]["latency_ms"]) / N_LAYERS,
        "ffn_only": (baseline - r["all_ffn_only"]["latency_ms"]) / N_LAYERS,
    }


def mask_latency(mask, benefit):
    total = benefit["baseline"]
    for m in mask:
        if m in benefit:
            total -= benefit[m]
    return total


corpus = json.load(open(CORPUS))
rows = corpus["rows"]
degradations_full = np.array([row["delta_bpb"] for row in rows])
keep = degradations_full <= 0.40
rows = [r for r, k in zip(rows, keep) if k]
degradations = degradations_full[keep]

fig = plt.figure(figsize=(21, 8))

for i, (platform, bench_path) in enumerate(BENCH_FILES.items()):
    benefit = per_mode_benefit(bench_path)
    latencies = np.array([mask_latency(row["mask"], benefit) for row in rows])
    speedups = benefit["baseline"] / latencies

    ax = fig.add_subplot(1, 3, i + 1, projection="3d")

    xy = np.vstack([speedups, degradations])
    kde = gaussian_kde(xy, bw_method=0.35)
    xg = np.linspace(speedups.min(), speedups.max(), 80)
    yg = np.linspace(degradations.min(), degradations.max(), 80)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    ax.plot_surface(Xg, Yg, Zg, cmap="jet", linewidth=0.2, antialiased=True,
                    edgecolor="k", alpha=1.0)
    ax.contour(Xg, Yg, Zg, zdir="z", offset=0, cmap="jet", levels=25)
    ax.view_init(elev=22, azim=-55)
    ax.invert_xaxis()

    ax.set_xlabel("Speedup", fontsize=FONT_LABEL, labelpad=14)
    ax.set_ylabel("BPB degradation", fontsize=FONT_LABEL, labelpad=14)
    ax.set_zlabel("Density", fontsize=FONT_LABEL, labelpad=10)
    ax.tick_params(axis="both", labelsize=FONT_TICK)

    ax.set_title(f"{platform}, 1B primary", fontsize=FONT_TITLE, pad=18)
    ax.text2D(
        0.02, 0.98, PANEL_LABELS[i], transform=ax.transAxes,
        fontsize=FONT_PANEL, fontweight="bold", va="top", ha="left",
    )

fig.suptitle("Speed/accuracy landscape across hardware, 1B primary",
             fontsize=FONT_TITLE + 2, y=1.02)
fig.tight_layout()
fig.savefig("blackwell/results/plots/landscape_final_3panel_1b_bigfont.png",
           dpi=200, bbox_inches="tight")
print("saved blackwell/results/plots/landscape_final_3panel_1b_bigfont.png")
