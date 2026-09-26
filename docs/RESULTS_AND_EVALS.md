# Results and Evaluation Tools

## Headline results

Binary (sequential/parallel) paired-consistency evidence, from `tab:binary-evidence`:

| Scale | Training | Agreement | ΔBPB |
|-------|----------|-----------|------|
| 120M | Sequential specialist (no consistency) | 37.11% | 0.5437 |
| 120M | Paired consistency (fixed λ=1) | 90.53% | 0.001 |
| 430M | Paired consistency (fixed λ=.1, stop-grad, 10B tok) | 92.24% | 0.00019 |
| 1B | Paired consistency (fixed λ=1, 197M tok) | 96.14% | 0.00120 |
| 3B | Paired consistency (GradNorm ρ=.2, 98M tok) | 98.10% | 0.00124 |
| 7B | Paired consistency (fixed λ=.1, stop-grad, 98M tok) | 95.56% | 0.00174 |

- **Specialist collapse:** the 120M sequential specialist (no consistency training) drops to 37.11% agreement (ΔBPB 0.5437) when forced to run under the parallel graph. Paired-consistency training at the same scale recovers 90.53% agreement at ΔBPB 0.001.
- **Composition law:** single-layer defects predict held-out mixed DAGs; the calibrated compiler narrows — but does not eliminate — budget violations. On disjoint validation windows, selected programs cost 0.0503–0.0516 BPB, slightly above the nominal 0.05 threshold (`tab:search-accounting`).
- **Hardware compilation:** one 1B six-mode checkpoint reaches 1.57–1.71× prefill speedup across Blackwell, H100, and L40S, near +0.05 BPB, using a different measured program on each GPU.
- **Downstream parity:** 7B lm-eval (HellaSwag, PIQA, WinoGrande, ARC) results are reported in the paper's downstream table.

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
