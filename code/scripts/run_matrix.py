"""Run (or print) the pilot or an experiment's runs in a fixed order, skipping finished runs.

    python scripts/run_matrix.py pilot --root /content/drive/MyDrive/GAM_Paper2/pilot --dry
    python scripts/run_matrix.py A --root /content/drive/MyDrive/GAM_Paper2/runs
    python scripts/run_matrix.py B --root ...      # seeds 0-2 (A: 0-4); override with --seeds
    python scripts/run_matrix.py all --root ...

Each run writes to <root>/<variant>/s<seed>/ (pilot: <root>/<run name>/). A run whose results.json
exists is skipped, so the script can simply be restarted after a Colab disconnect; an interrupted
run restarts from scratch.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gib import variants

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "train_mlm.py")
PILOT_SEED, PILOT_STEPS, WARMUP = 99, 2000, 1000
# Decided 20 Nov 2026, before any B or D run (compute limits): A keeps five seeds, B and D use three.
SEEDS = {"A": [0, 1, 2, 3, 4], "A_optional": [0, 1, 2, 3, 4], "B": [0, 1, 2], "D": [0, 1, 2]}


def pilot_runs(extra):
    base = ["--seed", str(PILOT_SEED), "--steps", str(PILOT_STEPS), "--eval_every", "500", "--diag_every", "250"]
    runs = [(v, base) for v in ("A1_standard", "A2_hybrid", "A3_pure", "D3_bounded",
                                "D1a_floor_0.02", "D1b_floor_0.05")]
    runs += [("A2_hybrid", base + ["--recalibrate_at", str(WARMUP)], "A2_hybrid_recal"),
             ("A2_hybrid", base + ["--learn_g"], "A2_hybrid_learned_g")]
    out = []
    for r in runs:
        v, args = r[0], r[1]
        name = r[2] if len(r) > 2 else v
        out.append((name, ["--variant", v] + args + extra))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["pilot", "A", "A_optional", "B", "D", "all"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; default per experiment (A: 0-4; B and D: 0-2, decided 20 Nov 2026)")
    ap.add_argument("--dry", action="store_true", help="print the commands only")
    ap.add_argument("--extra", default="--amp", help="extra arguments passed to every run")
    a = ap.parse_args()
    extra = a.extra.split()

    if a.which == "pilot":
        jobs = [(os.path.join(a.root, n), args) for n, args in pilot_runs(extra)]
    else:
        exps = ["A", "B", "D"] if a.which == "all" else [a.which]
        jobs = []
        for e in exps:
            seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else SEEDS[e]
            jobs += [(os.path.join(a.root, v, f"s{s}"), ["--variant", v, "--seed", str(s)] + extra)
                     for s in seeds for v in variants.EXPERIMENTS[e]]   # seed-major within each experiment

    for out, args in jobs:
        done = os.path.exists(os.path.join(out, "results.json"))
        cmd = [sys.executable, TRAIN, "--out", out] + args
        print(("[done] " if done else "") + " ".join(cmd), flush=True)
        if not done and not a.dry:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
