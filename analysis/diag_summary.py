"""Attention diagnostics: contribution C, KL shift D, suppression, lambda sign split, over training."""
import glob, json, os, statistics as st
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
rows, traj = [], []
for f in sorted(glob.glob(os.path.join(HERE, "..", "data", "diagnostics", "*.json"))):
    name = os.path.basename(f)[:-5]; v, s = name.split("__s"); recs = json.load(open(f))
    steps = {r["step"]: r for r in recs}
    print(v, s, 'missing steps', sorted(set(range(0, 20001, 500)) - set(steps))) if len(recs) < 41 else None
    for r in recs:
        L = r["layers"]
        C = [c for l in L for c in l["C"]]; D = [d for l in L for d in l["D"]]; lam = [x for l in L for x in l["lambda"]]
        traj.append({"variant": v, "seed": int(s), "step": r["step"], "C": np.mean(C), "D": np.mean(D),
                     "lam": np.mean(lam), "neg": sum(x < 0 for x in lam),
                     "suppressed": sum(c < 0.05 and d < 0.01 for c, d in zip(C, D)),
                     "max_score": max(x for l in L for x in l["max_score"]),
                     "mass": np.mean([l["mass_mean"] for l in L]), "entropy": np.mean([e for l in L for e in l["entropy"]])})
    fin = traj[-1]
    L = recs[-1]["layers"]
    rows.append(dict(fin, C_by_layer=[round(np.mean(l["C"]), 3) for l in L], lam_by_layer=[round(np.mean(l["lambda"]), 3) for l in L]))
T = pd.DataFrame(traj); F = pd.DataFrame(rows)
T.to_csv(os.path.join(HERE, "tables", "diag_trajectories.csv"), index=False)
F.to_csv(os.path.join(HERE, "tables", "diag_final.csv"), index=False)
pd.set_option("display.width", 220, "display.float_format", "{:.4f}".format)
g = F.groupby("variant")[["C", "D", "lam", "neg", "suppressed", "max_score", "mass", "entropy"]].agg(["mean", "std"])
print(g)
print("\nA2 trajectory (mean over 5 seeds)\n", T[T.variant == "A2_hybrid"].groupby("step")[["C", "D", "lam", "neg", "suppressed", "mass", "max_score"]].mean().iloc[[0, 1, 2, 4, 8, 12, 20, 30, 40]])
print("\nA2 C by layer (seed mean)", np.mean(np.array(F[F.variant == "A2_hybrid"].C_by_layer.tolist()), 0).round(3))
print("A2 lam by layer", np.mean(np.array(F[F.variant == "A2_hybrid"].lam_by_layer.tolist()), 0).round(3))
# first step at which mean C > 0.05 (A2)
t = T[T.variant == "A2_hybrid"].groupby("step").C.mean(); print("first step C>0.05:", t[t > 0.05].index.min())
