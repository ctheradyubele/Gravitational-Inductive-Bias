"""Paper 2 analysis: Experiments A, B and D from the 68 results.json files.

Primary outcome: test masked-LM loss after 20,000 steps. Comparisons are paired by seed (shared
initial weights, data order and masks). For each comparison: mean paired difference, 95% t-interval,
Cohen's d_z, paired t-test; Holm correction within each experiment. Experiment B: OLS of test loss on
mass, distance, mixing and their two-way interactions with seed as a block (type II ANOVA).
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "data", "results", "*.json"))):
        r = json.load(open(f))
        fin, cur = r["final"], {c["step"]: c["val_loss"] for c in r["curve"]}
        g = r.get("grav_final") or {}
        lam = [x for row in (g.get("lambda") or []) if row for x in row]
        rows.append({
            "variant": r["variant"], "seed": r["seed"], "test_loss": fin["test_loss"], "test_ppl": fin["test_ppl"],
            "val_loss": fin["val_loss"], "val_4k": cur.get(4000), "val_10k": cur.get(10000),
            "shuffled": fin.get("val_loss_mass_shuffled"), "uniform": fin.get("val_loss_mass_uniform"),
            "lam_mean": np.mean(lam) if lam else np.nan, "neg_heads": sum(x < 0 for x in lam) if lam else np.nan,
            "n_heads": len(lam) or np.nan, "minutes": r["train_time_s"] / 60, "gpu": r["env"]["gpu"],
            "peak_mem_mb": r["peak_mem_mb"], "tokens_per_s": r["tokens_per_s"],
            "clipped": r["counters"]["clipped_steps"], "nonfinite": r["counters"]["nonfinite_steps"],
            "spikes": r["counters"]["loss_spikes"], "mass": r["attn_config"]["mass"],
            "kernel": r["attn_config"]["kernel"], "mixing": r["attn_config"]["mixing"], "mode": r["attn_config"]["mode"]})
    return pd.DataFrame(rows)


def paired(df, a, b, seeds, col="test_loss"):
    x = df[(df.variant == a) & df.seed.isin(seeds)].set_index("seed")[col]
    y = df[(df.variant == b) & df.seed.isin(seeds)].set_index("seed")[col]
    d = (x - y).dropna()
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(x[d.index], y[d.index])
    return {"comparison": f"{a} - {b}", "n": n, "mean_diff": m, "sd_diff": sd, "ci_low": m - tcrit * se,
            "ci_high": m + tcrit * se, "dz": m / sd if sd > 0 else np.nan, "t": t, "p": p,
            "ppl_ratio": math.exp(m), "wins": int((d < 0).sum())}


def holm(rows):
    p = [r["p"] for r in rows]
    adj = multipletests(p, method="holm")[1]
    for r, q in zip(rows, adj):
        r["p_holm"] = q
    return rows


def main():
    df = load()
    df.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
    res = {}

    # ---------------- Experiment A (5 seeds)
    s5 = [0, 1, 2, 3, 4]
    A = ["A1_standard", "A2_hybrid", "A3_pure", "A4_distance_only"]
    summ = df[df.variant.isin(A)].groupby("variant").agg(
        test_mean=("test_loss", "mean"), test_sd=("test_loss", "std"), ppl_mean=("test_ppl", "mean"),
        val_4k=("val_4k", "mean"), val_10k=("val_10k", "mean"), minutes=("minutes", "mean"),
        mem=("peak_mem_mb", "mean"), tok_s=("tokens_per_s", "mean"), lam=("lam_mean", "mean"),
        neg=("neg_heads", "mean"), shuffled=("shuffled", "mean"), uniform=("uniform", "mean"),
        val=("val_loss", "mean")).reindex(A)
    summ.to_csv(os.path.join(OUT, "A_summary.csv"))
    compA = holm([paired(df, v, "A1_standard", s5) for v in A[1:]])
    compA.append(paired(df, "A2_hybrid", "A4_distance_only", s5))          # mass effect, reported, not in family
    pd.DataFrame(compA).to_csv(os.path.join(OUT, "A_paired.csv"), index=False)
    # data efficiency: paired differences in validation loss at 4k and 10k steps
    effA = [dict(paired(df, "A2_hybrid", "A1_standard", s5, c), step=c) for c in ("val_4k", "val_10k", "val_loss")]
    pd.DataFrame(effA).to_csv(os.path.join(OUT, "A_efficiency.csv"), index=False)
    res["A"] = {"summary": summ, "paired": compA, "efficiency": effA}

    # ---------------- Experiment B (3 seeds); A1, A2 and A4 seeds 0-2 included
    s3 = [0, 1, 2]
    core = df[(df["mode"] == "hybrid") & df.mass.isin(["learned", "norm"]) & df.kernel.isin(["inverse_square", "l2"])
              & df.seed.isin(s3) & df.variant.str.match(r"^(A2_hybrid|B_)")].copy()
    core = core[~core.variant.str.startswith("B_mass_only")]
    assert len(core) == 36, len(core)
    model = smf.ols("test_loss ~ (C(mass) + C(kernel) + C(mixing))**2 + C(seed)", data=core).fit()
    aov = anova_lm(model, typ=2)
    aov["eta_p2"] = aov["sum_sq"] / (aov["sum_sq"] + aov.loc["Residual", "sum_sq"])
    aov.to_csv(os.path.join(OUT, "B_anova.csv"))
    cells = core.groupby(["mass", "kernel", "mixing"]).agg(test_mean=("test_loss", "mean"), test_sd=("test_loss", "std"),
                                                         shuffled=("shuffled", "mean"), val=("val_loss", "mean"),
                                                         lam=("lam_mean", "mean"), neg=("neg_heads", "mean")).reset_index()
    cells.to_csv(os.path.join(OUT, "B_cells.csv"), index=False)
    names = sorted(core.variant.unique()) + ["B_mass_only", "A4_distance_only"]
    compB = holm([paired(df, v, "A1_standard", s3) for v in names])
    pd.DataFrame(compB).to_csv(os.path.join(OUT, "B_paired_vs_A1.csv"), index=False)
    marg = {f: core.groupby(f).test_loss.mean().to_dict() for f in ("mass", "kernel", "mixing")}
    res["B"] = {"anova": aov, "cells": cells, "paired": compB, "marginal": marg}

    # ---------------- Experiment D (3 seeds) vs A2
    D = ["D1a_floor_0.02", "D1b_floor_0.05", "D3_bounded", "D4_bounded_learned_sigma"]
    compD = holm([paired(df, v, "A2_hybrid", s3) for v in D])
    pd.DataFrame(compD).to_csv(os.path.join(OUT, "D_paired_vs_A2.csv"), index=False)
    res["D"] = {"paired": compD}

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
    print("=== A summary\n", summ)
    print("=== A paired vs A1 (Holm over A2,A3,A4)\n", pd.DataFrame(compA))
    print("=== A efficiency (A2-A1)\n", pd.DataFrame(effA)[["step", "mean_diff", "ci_low", "ci_high", "p", "wins"]])
    print("=== B ANOVA\n", aov)
    print("=== B marginal means\n", json.dumps(marg, indent=1))
    print("=== B cells\n", cells)
    print("=== B paired vs A1 (Holm over 14)\n", pd.DataFrame(compB)[["comparison", "mean_diff", "ci_low", "ci_high", "dz", "p", "p_holm", "wins"]])
    print("=== D paired vs A2 (Holm over 4)\n", pd.DataFrame(compD)[["comparison", "mean_diff", "ci_low", "ci_high", "dz", "p", "p_holm", "wins"]])
    print("=== stability\n", df[["nonfinite", "spikes"]].sum().to_dict(), "clipped share", round(df.clipped.mean() / 20000, 3))
    print("=== cost by variant group\n", df.groupby(df["mode"]).agg(min=("minutes", "mean"), mem=("peak_mem_mb", "mean")))
    by_gpu = df.groupby(["gpu", "mode"]).minutes.mean()
    print(by_gpu)


if __name__ == "__main__":
    main()
