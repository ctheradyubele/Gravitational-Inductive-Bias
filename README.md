# Does a Gravitational Inductive Bias Improve Transformer Attention? A Controlled Study

Code and data for the manuscript submitted to *Machine Learning and Knowledge Extraction* (MDPI).

- `code/` — model, attention variants, training script (`scripts/train_mlm.py`), run matrix (`scripts/run_matrix.py`), protocol (`configs/protocol.yaml`), tests (`tests/`, run with `pytest`), and `DEVIATIONS.md` (departures from the protocol).
- `data/results/` — one JSON per run (`<variant>__s<seed>.json`): 68 runs at 128 tokens (Experiment A: 20, B: 36, D: 12).
- `data/results_C/` — Experiment C: 9 runs (A1, A2, A4 x seeds 0-2) at 512-token sequences.
- `data/diagnostics/` — per-head diagnostics logged every 500 steps for A2, A4, D1a, D1b, D3 and D4 (128 tokens).
- `data/diagnostics_C/` — the same diagnostics for Experiment C (A2 and A4 at 512 tokens, seeds 0-2).
- `analysis/` — scripts that produce every table and figure in the paper from the files above:
  `python analyse.py` (tables), `python diag_summary.py` (diagnostic tables), `python figures.py` (figures), `python analyse_C.py` and `python figure_C.py` (Experiment C).
  `tables/` and `figures/` hold the outputs used in the manuscript.

The WikiText-103 corpus is not included; the training script downloads it through the Hugging Face `datasets` library.
