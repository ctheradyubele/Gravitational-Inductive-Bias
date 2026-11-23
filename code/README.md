# Paper 2: Does a gravitational inductive bias improve Transformer attention?

Code for the from-scratch study that follows Paper 1 (GAM). Small Transformer encoders are trained from
random initialisation with masked language modelling on WikiText-103; only the attention module
changes between runs. The definitions follow the Phase 2 specification document.

## Layout

| Path | What it holds |
| --- | --- |
| `gib/attention.py` | One attention module for every configuration: standard, hybrid, pure; mass, kernel and mixing levels; calibration; per-head diagnostics (C, D, entropy, quantiles) |
| `gib/model.py` | Pre-LayerNorm MLM encoder; identical shared weights for every variant given a seed |
| `gib/variants.py` | The 20 named configurations of Experiments A, B and D (+ optional A5) |
| `gib/data.py` | WikiText-103 chunking (copied from Paper 1) and seeded BERT masking |
| `scripts/train_mlm.py` | Trains one variant and writes `results.json`, `diagnostics.jsonl`, `train_log.jsonl`, `final.pt` |
| `scripts/run_matrix.py` | Runs the pilot or an experiment, skipping finished runs |
| `configs/protocol.yaml` | Fixed settings, the pilot plan and provisional thresholds |
| `tests/test_attention.py` | 93 tests, including proofs that the collapsed grid cells really collapse |
| `notebooks/run_paper2_colab.ipynb` | Colab notebook: mount Drive, install, test, run the pilot |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                      # 93 tests, a few seconds on CPU
python scripts/run_matrix.py pilot --root runs/pilot --extra "--amp --cache_dir data_cache"
```

The first real run tokenises WikiText-103 once and caches it in `--cache_dir` (put this on Drive in
Colab so that later sessions skip it).

## What the code guarantees

- **Same starting point.** For a given seed every variant has identical embeddings, projections and
  feed-forward weights, and the same data order, masking and dropout stream (tested).
- **Collapsed cells are excluded on proof.** No mass + no distance equals the standard model; no mass +
  L2 equals a rescaled dot product plus a key-norm bias (both tested to 1e-9).
- **Full precision where it matters.** Scores, distances and masses run with autocast off, because
  ||q||^2 + ||k||^2 - 2 q.k cancels badly in bfloat16 exactly where the inverse-square kernel peaks.
- **Padding never leaks** into real tokens, for every variant (tested).
- **G is calibrated** with Paper 1's rule (median |g| = std of s_dot) and frozen by default; the
  pilot compares this with recalibration at the end of warmup (`--recalibrate_at 1000`) and with
  learned G (`--learn_g`).

## Pilot results (15 November 2026, seed 99, 2,000 steps)

- **G strategy:** frozen at step 0. A toy run suggested the gravitational contribution C would fade
  as q and k grow, but on WikiText-103 C rose from 0.036 to 0.252 in A2, driven by growing learned
  masses. Recalibrated (0.261) and learned G (0.257) ended in the same place.
- **lambda rose** from 0.10 to 0.16, with 1 of 48 heads negative.
- **Mass matters in the hybrid:** shuffling masses raised A2's validation loss from 4.023 to 4.175.
- **Speed (A100 40 GB):** 0.16 s/step standard, 0.21 s/step gravitational at micro-batch 64; about 7 GB
  peak, so micro-batch 128 is now the default.
- **Fixed after the pilot:** the floor variant used lambda = lambda_min + softplus(rho), which slowed
  lambda's effective learning rate about 12-fold. It now uses the same parameter as "learned",
  projected onto [lambda_min, inf) after each step (tested).
