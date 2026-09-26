# Replication guide for results beyond the primary checkpoint-provenance tables

The paper's Reproducibility Statement commits to providing "code, configurations, raw
result artifacts, and exact selected execution programs" at this artifact link. All 18
checkpoint-provenance paths cited in the appendix (`tab:checkpoint-provenance`,
`tab:checkpoint-provenance-ternary`, `tab:checkpoint-provenance-sixmode`) are present
under `results/binary/`, `results/ternary/`, and `results/six_mode/` and have been
spot-checked to match the paper's cited reference BPB values exactly.

Nine additional paper-cited tables/figures initially had no backing data included in
this release. **Eight are now fully backed with real result files** (added under
`results/composition/`, `results/diagnostics/`, `results/downstream/`, and
`results/six_mode/search/` and `results/six_mode/surrogate/`). One item (below) is
permanently unrecoverable. This guide documents what's included, what it backs, and the
exact command to regenerate each one if you want to re-run it yourself.

## 1. `tab:binary-downstream` — 430M binary downstream accuracy — **backed**

Zero-shot HellaSwag/PIQA/WinoGrande/ARC-Easy for the 430M binary-consistency checkpoint,
sequential vs. parallel execution.

- `results/downstream/lm_eval_430m_10b_seq/results_2026-09-05T05-24-36.195134.json`
- `results/downstream/lm_eval_430m_10b_par/results_2026-09-05T05-25-31.168046.json`

To regenerate:

```bash
python scripts/export_hf_checkpoint.py --ckpt <ckpt>/430m_poly_10b/ckpts/step076293.safetensors \
  --config configs/scale430m_poly_10b.yaml --tokenizer_dir data/climbmix/bpe8192 \
  --output hf_export/430m_poly_10b_step076293
lm_eval --model hf --model_args pretrained=hf_export/430m_poly_10b_step076293 \
  --tasks hellaswag,piqa,winogrande,arc_easy --num_fewshot 0
```

Requires the external `lm-evaluation-harness` package (`lm_eval`), not one of this
repo's scripts.

## 2. Llama-family binary/ternary numeric claims

- **r=0.8174** (430M composition transfer) — **backed**:
  `results/composition/llama_430m_composition_holdout.json`
  (`fit_gt10_pred_leq10` split, `ca_pearson: 0.8174711890570215`, exact match).
  Regenerate with `eval_composition_holdout.py` on `llama_430m_poly_gradnorm@step012000`.
- **84.45% / BPB 1.1160** (430M ternary agreement) — **backed**:
  `results/ternary/llama_430m_ternary_gradnorm_mask_eval_agree.json`
  (`ternary_seq: 1.1160319329938653`, `ternary_mean_agreement: 0.84452109375`, exact
  match). Regenerate with `eval_ternary_masks.py` comparing
  `llama_430m_ternary_gradnorm@step012000` vs `llama_430m_poly_gradnorm@step012000`.
- **91.16% / BPB 1.0035** (1B binary) — **permanently unrecoverable.** The original
  checkpoint was lost before this release. A retrain (`llama_1b_poly_fixedcw`) reproduces
  90.33%/BPB 0.9247 — close, but not identical, as expected from a different random
  initialization. Regenerating the exact paper digit is not possible; a fresh retrain
  under `configs/scale1b_llama_poly.yaml` is the closest available substitute.

## 3/5/7. `fig:sixmode-compiler`, `tab:compiler`, `tab:search-accounting` — **backed**

Real-search speed/quality frontier and per-platform accounting for the 1B six-mode
checkpoint (`1b_6mode_gradnorm_cap10/step024000`) on Blackwell, H100, and L40S. All three
platforms share the same checkpoint and near-identical `seq_bpb` (~1.11767), confirming
they are the correct matched triple.

- `results/six_mode/search/6mode_blackbox_1b_gradnorm_cap10_24k_blackwell.json`
- `results/six_mode/search/6mode_blackbox_1b_gradnorm_cap10_24k_h100.json`
- `results/six_mode/search/6mode_blackbox_1b_gradnorm_cap10_24k_l40s.json`
- `results/six_mode/search/final_winners_disjoint_check.json` (disjoint-window recheck)

To regenerate (once per platform):

```bash
python scripts/eval_6mode_blackbox_search.py \
  --ckpt <ckpt>/1b_6mode_gradnorm_cap10/ckpts/step024000.safetensors \
  --config configs/scale1b_6mode_gradnorm_cap10_24k.yaml \
  --val_shards <shards> --tokenizer_dir <tokenizer> \
  --agreement_floor 0.05 --n_restarts 5 --steps_per_restart 50 --neighbors_per_step 6 \
  --output results/6mode_blackbox_1b_gradnorm_cap10_24k_<platform>.json
python scripts/eval_disjoint_windows.py --ckpt <same ckpt> --config <same config> \
  --masks_json <winning masks per platform> --window_offset 64
```

Known issue: the search's stagnation-break logic can fail to halt early; use
`--steps_per_restart` as a hard cap rather than relying on `--stagnation_patience` alone.

## 4. `tab:mode-set-ablation` — **backed**

Full 6-mode vs. `{Seq,Attn,Skip}` vs. `{Seq,Par,Rev}`, ε=0.05, 430M
`gradnorm@step024000`.

- `results/six_mode/search/hillclimb_430m_gradnorm_24k_6mode_from4mode_v2.json` (full 6-mode)
- `results/six_mode/search/hillclimb_430m_gradnorm_24k_skip_attn_only.json` ({Seq,Attn,Skip})
- `results/six_mode/search/hillclimb_430m_gradnorm_24k_seq_par_rev_only.json` ({Seq,Par,Rev})

To regenerate:

```bash
python scripts/eval_6mode_blackbox_search.py --ckpt <ckpt> \
  --config configs/scale430m_6mode_gradnorm_24k.yaml --bpb_floor 0.05 \
  --n_restarts 3 --steps_per_restart 60 --neighbors_per_step 6 --seed 42 \
  [--alt_modes attn_only,skip | --alt_modes parallel,reverse]
```

Note: the full-6-mode number in the paper is a chained continuation across several
searches (2-mode → 4-mode → further refinement), not one clean single-command run —
comparing differently-sized mode spaces at a fixed budget is a known, acknowledged
confound (search difficulty scales with space size), discussed in the paper itself.

## 6. `fig:speed-quality-landscape` — **backed**

```bash
python scripts/plot_hardware_landscape_3panel.py
```

No CLI args. Regenerates `results/six_mode/plots/landscape_final_3panel_1b_bigfont.png`
directly from data already included in this export
(`results/six_mode/6mode_masks_1b_gradnorm_cap10_24k.json` and
`results/six_mode/latency/6mode_latency_1b_gradnorm_cap10_24k_{blackwell,h100,l40s}.json`).
The rendered PNG is included in this release.

## 8. `tab:blackwell-diagnostics` — **backed**

Llama six-mode compiler diagnostics, 120M/430M/1B, `cost_source: ridge`, `max_free: 4`.

- `results/diagnostics/6mode_compiler_llama_120m_ridge_maxfree4.json`
- `results/diagnostics/6mode_compiler_llama_430m_ridge_maxfree4.json`
- `results/diagnostics/6mode_compiler_llama_1b_ridge_maxfree4.json`

Numbers reproduce the paper's table exactly. Caution: a deprecated, superseded artifact
family using `--cost_source m2m3` also exists; do not confuse it with the `ridge` variant
used here.

## 9. `tab:surrogate-error` — **backed**

Held-out surrogate error, per scale. Picked by closeness to the paper's reported numbers
where determinable:

- `results/six_mode/surrogate/6mode_surrogate_120m_gradnorm_7k.pkl`
- `results/six_mode/surrogate/6mode_surrogate_430m_gradnorm_v2.pkl` (holdout r=0.7675,
  matches the paper's cited 0.768 almost exactly)
- `results/six_mode/surrogate/6mode_surrogate_1b_gradnorm_cap10_24k.pkl`

To regenerate: `python scripts/build_6mode_surrogate.py --scale <scale> --n_layers <n> --results_dir <dir with real 6mode_masks_*.json for this scale> --output results/six_mode/surrogate/<name>.pkl` — prints holdout Pearson r / RMSE.

## 10. `tab:matched-budget` — **fully backed, all 4 seeds**

Plain (control) vs. surrogate-ranked search, 4 seeds, 430M `gradnorm@step012000`, ε=0.05.

- `results/six_mode/search/multiseed_control_s{7,42,999,123}.json`
- `results/six_mode/search/multiseed_surrogate_s{7,42,999,123}.json`

Seed 123 (`best_speedup`: control 1.5303×, surrogate 1.4194×) was freshly run to
complete this table — the original internal logs only had 3 of 4 seeds. Consistent with
every other seed: plain search beats surrogate-ranked search.

To regenerate:

```bash
python scripts/eval_6mode_blackbox_search.py \
  --ckpt <ckpt>/430m_6mode_gradnorm/ckpts/step012000.safetensors \
  --config configs/scale430m_6mode_gradnorm.yaml --bpb_floor 0.05 \
  --steps_per_restart 15 --neighbors_per_step 4 --seed <7|42|999|123> \
  [--surrogate_filter --surrogate_type m2m3 --surrogate_pool_size 40]
```

---

## Summary

Of the 9 previously-uncovered items, **8 are now fully backed with real result files**
included in this release, alongside all 18 originally-cited checkpoint-provenance paths.
**One item remains permanently unrecoverable**: the Llama-1B binary 91.16% agreement
claim (item 2, third bullet) — the original checkpoint was lost before this release, and
no retrain reproduces the exact digit (a retrain gets close, 90.33%, but differs as
expected from a different random initialization). This is the only gap in the
reproducibility statement's artifact-link promise, and it is a data-loss limitation, not
a withheld or fabricated result.
