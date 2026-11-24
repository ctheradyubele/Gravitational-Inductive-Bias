"""Figure 5: Experiment C, paired differences from A1 at 128 and 512 tokens (seeds 0-2)."""
import json, glob, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data") if os.path.isdir(os.path.join(HERE, "..", "data")) else HERE

def load(folder):
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(DATA, folder, "*.json"))]
    d = pd.DataFrame([{"variant": r["variant"], "seed": r["seed"], "test": r["final"]["test_loss"]} for r in rows])
    return d[d.seed.isin([0, 1, 2])].pivot(index="seed", columns="variant", values="test")
p128, p512 = load("results"), load("results_C")
fig, ax = plt.subplots(figsize=(6.4, 2.6))
comps = [("A2_hybrid", "A2 hybrid − A1"), ("A4_distance_only", "A4 distance-only − A1")]
for i, (v, lab) in enumerate(comps):
    for j, (p, col, L) in enumerate([(p128, BLUE, "128 tokens"), (p512, ORANGE, "512 tokens")]):
        x = (p[v] - p["A1_standard"]).values
        y = i * 0.75 + (-0.15 if j == 0 else 0.15)
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); h = stats.t.ppf(.975, len(x) - 1) * se
        ax.errorbar(m, y, xerr=h, fmt="o", color=col, ecolor=col, elinewidth=2, capsize=0, ms=7, zorder=3,
                    label=L if i == 0 else None)
        ax.scatter(x, [y] * len(x), s=14, facecolor="white", edgecolor=col, zorder=4, lw=1.2)
        ax.text(0.042, y, f"{m:+.3f} [{m-h:+.3f}, {m+h:+.3f}]", va="center", fontsize=8, color=INK2)
ax.axvline(0, color=INK2, lw=1, ls="--")
ax.set_yticks([0, 0.75]); ax.set_yticklabels([c[1] for c in comps]); ax.invert_yaxis()
ax.set_xlim(-0.10, 0.11); ax.set_xlabel("Test-loss difference from A1 (nats); negative favours the gravitational variant")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=8)
ax.set_title("The hybrid's advantage triples at 512 tokens", loc="left", fontsize=10, color=INK)
fig.tight_layout(); save(fig, "fig5_expC_length")
