# One Model, Many DAGs

Research code for **execution-graph equivalent Transformers**: training one checkpoint that can run under sequential, parallel, and mixed Attention–FFN execution DAGs while preserving quality.

> **Status:** ICLR 2027 submission. See `docs/RESULTS_AND_EVALS.md` for headline results and `results/` for the curated result artifacts referenced by the paper.

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

## Main research question

> Can a neural network learn an equivalence class of execution graphs, so deployment can compile one checkpoint into a hardware-specific DAG without retraining?

## Repository map

```text
src/fogen/                 Model, training, and evaluation library
configs/                    Training and ablation configs, 120M through 7B scale
scripts/                    Evaluation, analysis, and export tools
docs/                        Methodology writeups, results tables, and eval-tool reference
results/                    Curated result artifacts referenced by the paper
tests/                       Unit tests
```

See `docs/RESULTS_AND_EVALS.md` for a full breakdown of what each script produces and `docs/TERNARY_WRITEUP.md` / `docs/learning_guide.html` for methodology writeups.

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

Large training data and checkpoints are intentionally not committed.
