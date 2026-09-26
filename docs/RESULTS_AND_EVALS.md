# Results and Evaluation Tools

## Headline results

| Scale | Agreement | Sym KL | ΔBPB | Architecture |
|-------|-----------|--------|------|--------------|
| 120M | 95.4% | 5.86 | 0.0014 | Primary |
| 430M | 96.5% | 4.68 | 0.0016 | Primary |
| 1B | 96.1% | 4.10 | 0.0012 | Primary |
| 3B (gradnorm) | 98.3% | 0.92 | 0.0012 | Primary |
| 7B | 95.6% | 3.78 | 0.0017 | Primary |
| 430M | 93.6% | 10.4 | 0.003 | Llama |
| 1B | 91.2% | 26.1 | 0.002 | Llama |

- **Specialist collapse:** 430M specialist gets 54% agreement (KL=1273) under graph switch. Polymorphic model: 96.5% (KL=4.68).
- **Composition law:** single-layer defects predict 10k+ held-out mixed DAGs with Pearson 0.97 (primary) and 0.85 (Llama). Calibrated compiler reduces budget violations from 33–47% to <1%.
- **Hardware compilation:** L40S 1.20–1.54×, Apple M4 1.14×, 7B TP=2 1.20× from communication overlap.
- **Downstream parity:** 7B lm-eval (HellaSwag, PIQA, WinoGrande, ARC) — all execution graphs within 1 stderr.
- **Loss analysis:** MSE and KL both achieve graph consistency. KL gradients are 10–58× larger than LM gradients; gradient-normalized weighting resolves scale sensitivity.

## Result artifacts (`results/`)

- `results/binary/` — Binary (sequential/parallel) execution-graph results, 120M–7B
- `results/ternary/` — Ternary (seq/parallel/skip) consistent-mask evals, 120M–3B
- `results/six_mode/` — Six-mode mask evals, 120M–1B (primary + Llama), plus per-hardware latency benchmarks

## Evaluation and analysis tools (`scripts/`)

| Script | Produces |
|---|---|
| `eval_graph_rewrites.py` | Full binary graph-rewrite sweep (agreement, KL, BPB) |
| `eval_ternary_masks.py` | Ternary (seq/parallel/skip) mask evaluation |
| `eval_6mode_masks.py` | Six-mode random-mask sweep + composition law |
| `eval_execution_polymorphism.py` | Agreement/KL under sequential vs parallel execution |
| `eval_composition_holdout.py` | Held-out composition-law evaluation (multiple splits) |
| `eval_disjoint_windows.py` | Re-evaluate selected masks on disjoint validation windows |
| `eval_graph_compiler.py` | Composition-law-guided graph compiler (calibration/test split) |
| `eval_6mode_compiler.py` | Six-mode compiler (DP/greedy program selection) |
| `eval_6mode_blackbox_search.py` | Measurement-guided hill-climb search over six-mode programs |
| `eval_6mode_latency.py` | Per-mode latency measurement across hardware |
| `eval_6mode_active_loop.py` | Active-learning loop for six-mode surrogate/search |
| `eval_6mode_pareto.py` | Random-sampling Pareto baseline for six-mode search |
| `eval_specific_masks.py` | Real (bpb, latency, agreement) measurement for a fixed mask list |
| `build_6mode_surrogate.py` | Learned (GBM) surrogate over the six-mode program space |
| `build_6mode_archive.py` | Reusable, ranked archive of validated six-mode programs |
| `plot_hardware_landscape_3panel.py` | Speed/quality density landscape across GPUs |
| `measure_gradient_ratio.py` | ‖∇consistency‖ / ‖∇LM‖ measurement |
| `export_hf_checkpoint.py` | Export to HuggingFace format |

## Current status

All experiments complete through 7B (primary architecture) and 1B (Llama). Results include: scale series 120M–7B, second architecture validation, loss-objective Pareto sweeps at 430M and 1B, gradient ratio analysis, gradient-normalized training at 3B, pairwise mechanism analysis, compiler calibration, vLLM serving, TP=2 communication overlap, and downstream benchmarks.
