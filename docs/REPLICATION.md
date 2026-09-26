# Replication guide for results not shipped with raw data

Checkpoints and most raw search/measurement logs are intentionally not included in this
release (large binaries, or exploratory artifacts outside the paper's final scope). The
following paper-cited tables and figures do not have backing data files in `results/`.
This guide gives the exact script and arguments needed to regenerate each one, with an
honest confidence level per item — several hyperparameters below are recovered from
internal experiment logs rather than a single literally-logged command, and are noted
as such.

## 1. `tab:binary-downstream` — 430M binary downstream accuracy

Zero-shot HellaSwag/PIQA/WinoGrande/ARC-Easy for the 430M binary-consistency checkpoint,
sequential vs. parallel execution.

```bash
python scripts/export_hf_checkpoint.py --ckpt <ckpt>/430m_poly_10b/ckpts/step076293.safetensors \
  --config configs/scale430m_poly_10b.yaml --tokenizer_dir data/climbmix/bpe8192 \
  --output hf_export/430m_poly_10b_step076293
lm_eval --model hf --model_args pretrained=hf_export/430m_poly_10b_step076293 \
  --tasks hellaswag,piqa,winogrande,arc_easy --num_fewshot 0
```

**Confidence: high** — checkpoint and step confirmed exactly. Requires the external
`lm-evaluation-harness` package (`lm_eval`), not one of this repo's scripts.

## 2. Llama-family binary/ternary numeric claims

- **r=0.8174** (430M composition transfer): `eval_composition_holdout.py` on
  `llama_430m_poly_gradnorm@step012000`. **High confidence.**
- **84.45% / BPB 1.1160** (430M ternary agreement): `eval_ternary_masks.py` comparing
  `llama_430m_ternary_gradnorm@step012000` vs `llama_430m_poly_gradnorm@step012000`.
  **High confidence.**
- **91.16% / BPB 1.0035** (1B binary): **could not determine — the original checkpoint
  is lost.** A retrain (`llama_1b_poly_fixedcw`) reproduces 90.33%/BPB 0.9247, close but
  not identical to the paper's reported figure. If exact reproduction matters, this
  number needs a fresh retrain of the 1B Llama binary checkpoint under
  `configs/scale1b_llama_poly.yaml`, which will not reproduce the paper's exact digits
  (different random initialization).

## 3. `fig:sixmode-compiler` — real-search speed/quality frontier, 1B, 3 platforms

```bash
python scripts/eval_6mode_blackbox_search.py \
  --ckpt <ckpt>/1b_6mode_gradnorm_cap10/ckpts/step024000.safetensors \
  --config configs/scale1b_6mode_gradnorm_cap10_24k.yaml \
  --val_shards <shards> --tokenizer_dir <tokenizer> \
  --agreement_floor 0.05 --n_restarts 5 --steps_per_restart 50 --neighbors_per_step 6 \
  --output results/6mode_blackbox_1b_gradnorm_cap10_24k_<platform>.json
```

Run once per GPU platform, then re-check selected programs with:

```bash
python scripts/eval_disjoint_windows.py --ckpt <same ckpt> --config <same config> \
  --masks_json <winning masks per platform> --window_offset 64
```

**Confidence: high** for checkpoint/config/reported speedups. **Medium** for
`--n_restarts`/`--steps_per_restart` — inferred from nearby runs, not one literally
quoted invocation. Known issue: the search's stagnation-break logic can fail to halt
early; use `--steps_per_restart` as a hard cap rather than relying on
`--stagnation_patience` alone.

## 4. `tab:mode-set-ablation` — restricted mode-family search, 430M

Full 6-mode vs. `{Seq,Attn,Skip}` vs. `{Seq,Par,Rev}`, ε=0.05, `gradnorm@step024000`.

```bash
python scripts/eval_6mode_blackbox_search.py --ckpt <ckpt> \
  --config configs/scale430m_6mode_gradnorm_24k.yaml --bpb_floor 0.05 \
  --n_restarts 3 --steps_per_restart 60 --neighbors_per_step 6 --seed 42 \
  [--alt_modes attn_only,skip | --alt_modes parallel,reverse]
```

**Confidence: medium.** Speedups match exactly, but the full-6-mode number in the paper
is a chained continuation across several searches (2-mode → 4-mode → further
refinement), not one clean single-command run — comparing differently-sized mode
spaces at a fixed budget is a known, acknowledged confound (search difficulty scales
with space size), discussed in the paper itself.

## 5. `tab:compiler` — compiler results, 1B, 3 platforms

```bash
python scripts/eval_6mode_compiler.py \
  --ckpt <ckpt>/1b_6mode_gradnorm_cap10/ckpts/step024000.safetensors \
  --config configs/scale1b_6mode_losstarget.yaml \
  --val_shards <shards> --tokenizer_dir <tokenizer> \
  --probes_json results/six_mode/6mode_masks_1b_gradnorm_cap10_24k.json \
  --budgets 0.0005,0.002,0.01,0.03 \
  --output results/6mode_compiler_1b_gradnorm_cap10_24k_<platform>.json
```

**Confidence: medium-high.** Blackwell's checkpoint/config/winning-mode-tuple confirmed
exact. H100/L40S nearby values found don't exactly match the paper (close but not
identical), and the underlying search was actually run via
`eval_6mode_blackbox_search.py` per internal logs — `eval_6mode_compiler.py` is the
analytic successor path, not necessarily what produced this exact table.

## 6. `fig:speed-quality-landscape` — 3-panel density landscape, 1B, 3 platforms

```bash
python scripts/plot_hardware_landscape_3panel.py
```

No CLI args — requires `results/six_mode/6mode_masks_1b_gradnorm_cap10_24k.json` and
`results/six_mode/latency/6mode_latency_1b_gradnorm_cap10_24k_{blackwell,h100,l40s}.json`,
both of which are already included in this export.

**Confidence: medium.** Script, corpus, and checkpoint confirmed to match the figure's
description; the exact original output filename could not be pinned down among ~20
similarly-named same-day variants in the internal logs.

## 7. `tab:search-accounting` — per-platform search accounting, 1B

Evaluation counts, unique programs, in/out-of-budget counts, and a disjoint-window
recheck. Same command as item 3, plus:

```bash
python scripts/eval_disjoint_windows.py --ckpt <ckpt> --config <config> \
  --masks_json <winning masks> --window_offset 64 --max_windows 64
```

**Confidence: high** for the evaluation/unique/in-budget/out-of-budget counts (exact
matches found for all three platforms). **Medium** for the disjoint-window ΔBPB
column — the closest matching artifact found is close but not an exact match.

## 8. `tab:blackwell-diagnostics` — Llama six-mode compiler diagnostics, 120M/430M/1B

```bash
python scripts/eval_6mode_compiler.py \
  --ckpt <ckpt>/llama_1b_6mode_gradnorm/ckpts/step012000.safetensors \
  --config configs/scale1b_llama_6mode_gradnorm.yaml \
  --probes_json <6mode_masks output for this checkpoint> \
  --cost_source ridge --max_free 4 --budgets 0.0,0.0005,0.002,0.005,0.02 \
  --output results/6mode_compiler_llama_1b_ridge_maxfree4.json
```

Repeat per scale with the matching config/checkpoint (120M/430M use
`--budgets 0.0,0.005,0.02`).

**Confidence: high** — numbers reproduce the paper's table exactly. Caution: a
deprecated, superseded artifact family using `--cost_source m2m3` also exists
internally; do not confuse it with the `ridge` variant used here.

## 9. `tab:surrogate-error` — held-out surrogate error, per scale

```bash
python scripts/build_6mode_surrogate.py --scale 430m --n_layers 20 \
  --results_dir <results dir with real 6mode_masks_*.json for this scale> \
  --output results/6mode_surrogate_430m.pkl
```

Repeat per scale. The script itself prints the holdout Pearson r / RMSE.

**Confidence: medium.** Mechanism (train/test split, holdout Pearson/RMSE) confirmed
exactly. Nearby internal values are close but not identical to the paper's reported
numbers (e.g. 430M r≈0.82 internally vs. 0.768 in the paper) — several versioned
surrogate rebuilds exist internally and the exact final one used for the paper's table
isn't uniquely pinned down.

## 10. `tab:matched-budget` — plain vs. surrogate-ranked search, 4 seeds, 430M

```bash
python scripts/eval_6mode_blackbox_search.py \
  --ckpt <ckpt>/430m_6mode_gradnorm/ckpts/step012000.safetensors \
  --config configs/scale430m_6mode_gradnorm.yaml --bpb_floor 0.05 \
  --steps_per_restart 15 --neighbors_per_step 4 --seed <123|7|42|999> \
  [--surrogate_filter --surrogate_type m2m3 --surrogate_pool_size 40]
```

**Confidence: high** for checkpoint/floor/step-count/seeds — recovered from internal
metadata, and speedups match the paper exactly. **Medium** for the exact starting-mask
argument (`--seed_from_corpus` or equivalent) — described qualitatively internally, not
logged as a literal command.

---

## Summary of gaps

- **Item 2's 91.16% Llama-1B binary claim cannot be exactly reproduced** — the original
  checkpoint was lost before this release; a retrain gets close (90.33%) but not
  identical, as expected from a different random initialization.
- **Items 4, 5, 6, 9, 10** have confirmed scripts and checkpoints, but some
  hyperparameters or exact output filenames are inferred from nearby internal runs
  rather than a single literally-logged invocation — flagged as "medium confidence"
  above. Re-running with the given arguments should reproduce results close to, but not
  necessarily bit-identical to, the paper's exact reported digits.
