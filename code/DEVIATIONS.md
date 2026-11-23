# Deviations from the protocol and infrastructure events

The protocol (`configs/protocol.yaml`) is fixed after the pilot and before the confirmatory runs.
Record every later change here with its date and reason, and report it in the paper's appendix.

| Date | Item | Description and effect |
| --- | --- | --- |
| 2026-11-15 | Duplicate sessions | From A2 seed 0 onwards, two Colab sessions ran the same runs into the same folders about 4-13 minutes apart. Final results are unaffected (same configuration and seed; results.json written by whichever copy finished last), but diagnostics.jsonl and train_log.jsonl of the affected runs interleave records from both copies. Fixed by a run lock (scripts/train_mlm.py); affected diagnostics are de-duplicated by step for analysis. |
| 2026-11-20 | Seeds for B and D | Experiments B and D use three seeds (0-2) instead of five, because of compute limits; Experiment A keeps five. Decided before any B or D run started, so the choice cannot depend on their results. Total runs: 20 (A) + 36 (B) + 12 (D) = 68 instead of 100. |
