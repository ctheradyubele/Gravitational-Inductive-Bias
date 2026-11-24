"""Experiment C: paired comparisons at 512 tokens and the change from 128 tokens (seeds 0-2).
Writes tables/C_runs.csv and tables/C_paired.csv."""
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES128 = os.path.join(HERE, "..", "data", "results") if os.path.isdir(os.path.join(HERE, "..", "data")) else os.path.join(HERE, "results")
RES512 = os.path.join(HERE, "..", "data", "results_C") if os.path.isdir(os.path.join(HERE, "..", "data")) else os.path.join(HERE, "results_C")

def load(folder):
    rows = []
    for f in glob.glob(os.path.join(folder, "*.json")):
        r = json.load(open(f))
        rows.append({"variant": r["variant"], "seed": r["seed"], "seq_len": r["args"]["seq_len"],
                     "test_loss": r["final"]["test_loss"], "val_loss": r["final"]["val_loss"],
                     "val_loss_mass_shuffled": r["final"].get("val_loss_mass_shuffled"),
                     "val_loss_mass_uniform": r["final"].get("val_loss_mass_uniform"),
                     "gpu": r["env"]["gpu"], "train_min": r["train_time_s"] / 60,
                     "tokens_per_s": r["tokens_per_s"], "peak_mem_gb": r["peak_mem_mb"] / 1024, **r["counters"]})
    return pd.DataFrame(rows)

def paired(x, name):
    x = np.asarray(x, float); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); h = stats.t.ppf(.975, len(x) - 1) * se
    return {"comparison": name, "mean": m, "ci_low": m - h, "ci_high": m + h, "d_z": m / x.std(ddof=1),
            "p": stats.ttest_1samp(x, 0).pvalue, "n_negative": int((x < 0).sum()), "per_seed": np.round(x, 4).tolist()}

def main():
    c = load(RES512); c.to_csv(os.path.join(HERE, "tables", "C_runs.csv"), index=False)
    p = c.pivot(index="seed", columns="variant", values="test_loss")
    a = load(RES128); q = a[a.seed.isin([0, 1, 2])].pivot(index="seed", columns="variant", values="test_loss")
    res = [paired(p.A2_hybrid - p.A1_standard, "A2-A1 @512"), paired(p.A4_distance_only - p.A1_standard, "A4-A1 @512"),
           paired(p.A2_hybrid - p.A4_distance_only, "A2-A4 @512"), paired(q.A2_hybrid - q.A1_standard, "A2-A1 @128 (seeds 0-2)"),
           paired((p.A2_hybrid - p.A1_standard) - (q.A2_hybrid - q.A1_standard), "change in A2-A1, 512 minus 128 (primary)"),
           paired((p.A4_distance_only - p.A1_standard) - (q.A4_distance_only - q.A1_standard), "change in A4-A1, 512 minus 128")]
    r = pd.DataFrame(res)
    ps = r.p[:3].values; order = np.argsort(ps); holm = np.zeros(3); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (3 - k) * ps[i])); holm[i] = run
    r["holm_p"] = list(holm) + [np.nan] * 3
    r.to_csv(os.path.join(HERE, "tables", "C_paired.csv"), index=False)
    print(r.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
