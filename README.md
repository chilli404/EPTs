# One Model, Many DAGs

Research code and collaborator materials for **execution-graph equivalent Transformers**: training one checkpoint that can run under sequential, parallel, and mixed Attention–FFN execution DAGs while preserving quality.

> **Status:** ICLR 2027 submission. All experiments complete (120M–7B primary architecture + Llama 430M/1B). See `results/` for the curated result artifacts referenced by the paper.

## Core idea

A standard Transformer block is sequential:

```text
x -> Attention -> FFN -> output
```

A parallel block lets Attention and FFN read the same normalized input:

```text
              -> Attention -
x -> Norm(x)                 + -> output
              -> FFN -------
```

Ordinary Transformer weights specialize to the graph used during training. Changing only the execution dependency can substantially degrade quality. We train with a graph-consistency objective so the same weights implement approximately equivalent functions across execution DAGs.

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

See `results/` for the underlying result artifacts and `docs/TERNARY_WRITEUP.md` for the ternary-mode writeup.

## Repository map

```text
src/fogen/
  model.py                Transformer with 4 execution modes (seq/par/fused/cuda)
                          Supports both primary (ReLU², QK-norm, value embeddings)
                          and Llama-style (RMSNorm, SwiGLU) architectures
  training/train.py       Config-driven training with polymorphic loss,
                          multiple consistency objectives, gradient-normalized cw
  execution_graph.py      Defect-budget graph compiler (greedy + DP)
  hf_model.py             Hugging Face AutoModel interface
  evals/                  BPB evaluation and forced-choice scoring

configs/                  Training and ablation configs, 120M through 7B scale
  pareto/                 430M and 1B loss-objective sweep configs
                          Naming: {scale}_{objective}_{temperature}_cw{weight}

scripts/                  Evaluation, analysis, and export tools that produced
                          the paper's cited results (pruned to only these)
  eval_graph_rewrites.py        Full binary graph-rewrite sweep (agreement, KL, BPB)
  eval_ternary_masks.py         Ternary (seq/parallel/skip) mask evaluation
  eval_6mode_masks.py           Six-mode random-mask sweep + composition law
  eval_execution_polymorphism.py  Agreement/KL under sequential vs parallel execution
  eval_composition_holdout.py   Held-out composition-law evaluation (multiple splits)
  eval_disjoint_windows.py      Re-evaluate selected masks on disjoint validation windows
  eval_graph_compiler.py        Composition-law-guided graph compiler (calibration/test split)
  eval_6mode_compiler.py        Six-mode compiler (DP/greedy program selection)
  eval_6mode_blackbox_search.py Measurement-guided hill-climb search over six-mode programs
  eval_6mode_latency.py         Per-mode latency measurement across hardware
  eval_6mode_active_loop.py     Active-learning loop for six-mode surrogate/search
  eval_6mode_pareto.py          Random-sampling Pareto baseline for six-mode search
  eval_specific_masks.py        Real (bpb, latency, agreement) measurement for a fixed mask list
  build_6mode_surrogate.py      Learned (GBM) surrogate over the six-mode program space
  build_6mode_archive.py        Reusable, ranked archive of validated six-mode programs
  plot_hardware_landscape_3panel.py  Speed/quality density landscape across GPUs
  measure_gradient_ratio.py     ||∇consistency|| / ||∇LM|| measurement
  export_hf_checkpoint.py       Export to HuggingFace format

docs/
  TERNARY_WRITEUP.md      Ternary (seq/parallel/skip) execution-mode writeup
  learning_guide.html     Self-contained walkthrough of the project

results/                  Curated result artifacts referenced by the paper
  binary/                 Binary (sequential/parallel) execution-graph results, 120M–7B
  ternary/                Ternary (seq/parallel/skip) consistent-mask evals, 120M–3B
  six_mode/               Six-mode mask evals, 120M–1B (primary + Llama)
```

## Suggested reading order

1. This README
2. `docs/TERNARY_WRITEUP.md` — ternary execution-mode methodology and results
3. `docs/learning_guide.html` — project walkthrough (problem, method, results, mechanism)

## Main research question

> Can a neural network learn an equivalence class of execution graphs, so deployment can compile one checkpoint into a hardware-specific DAG without retraining?

## Training objective

For sequential and parallel logits \(z_s,z_p\):

\[
\mathcal L
=\tfrac12\mathcal L_{\mathrm{seq}}
+\tfrac12\mathcal L_{\mathrm{par}}
+\lambda\|\bar z_s-\bar z_p\|_2^2,
\]

where centered logits remove vocabulary-wide shifts that do not affect softmax probabilities.

## Theory in one line

The local graph defect is exactly

\[
d_l(x)=g_l(x+a_l(x))-g_l(x)
=\int_0^1Jg_l(x+t a_l(x))a_l(x)\,dt.
\]

This connects Attention updates, FFN steering, execution-graph divergence, and defect-guided graph compilation.

## Current status

All experiments complete through 7B (primary architecture) and 1B (Llama). Results include: scale series 120M–7B, second architecture validation, loss-objective Pareto sweeps at 430M and 1B, gradient ratio analysis, gradient-normalized training at 3B, pairwise mechanism analysis, compiler calibration, vLLM serving, TP=2 communication overlap, and downstream benchmarks.

## Quick start

```bash
uv sync --extra dev
uv run pytest -q
uv run python scripts/eval_graph_rewrites.py --help
```

Or without uv:

```bash
python -m pip install -e ".[dev]"
pytest -q
python scripts/eval_graph_rewrites.py --help
```

Large training data and checkpoints are intentionally not committed. See `results/` for the curated result artifacts.
