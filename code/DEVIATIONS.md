# Deviations from the protocol and infrastructure events

The protocol (`configs/protocol.yaml`) is fixed after the pilot and before the confirmatory runs.
Record every later change here with its date and reason, and report it in the paper's appendix.

| Date | Item | Description and effect |
| --- | --- | --- |
| 2026-11-15 | Duplicate sessions | From A2 seed 0 onwards, two Colab sessions ran the same runs into the same folders about 4-13 minutes apart. Final results are unaffected (same configuration and seed; results.json written by whichever copy finished last), but diagnostics.jsonl and train_log.jsonl of the affected runs interleave records from both copies. Fixed by a run lock (scripts/train_mlm.py); affected diagnostics are de-duplicated by step for analysis. |
| 2026-11-20 | Seeds for B and D | Experiments B and D use three seeds (0-2) instead of five, because of compute limits; Experiment A keeps five. Decided before any B or D run started, so the choice cannot depend on their results. Total runs: 20 (A) + 36 (B) + 12 (D) = 68 instead of 100. |
| 2026-11-15 | Micro-batch for A1 and A3, seed 0 | These two runs used micro-batch 64 (the pilot setting) instead of 128. Gradient accumulation kept the effective batch at 256 sequences, so the optimisation is unchanged; only wall-clock time and peak memory differ. They are excluded from the cost comparison. |
| 2026-11-15 to 2026-11-22 | Mixed GPUs | Runs used NVIDIA A100 40 GB and 80 GB GPUs as Colab assigned them. Losses are unaffected; timing and memory are compared only between runs on the same GPU type with the same micro-batch. |
| 2026-11-23 | Experiment C added | Added after Experiments A, B and D were analysed, to test H3 (sequence length), which the original run plan left untested. Design fixed before any C run: A1, A2 and A4 at 512-token sequences, 64 sequences per step (the same 32,768 tokens per step, 20,000 steps and passes over the data as A, B and D), seeds 0-2, all other settings unchanged. Primary comparison: A2 - A1 at 512 tokens, set against A2 - A1 on seeds 0-2 at 128 tokens. Runs are written to runs/C_seq512/. Reported in the paper as an additional experiment added after the main analysis. |
