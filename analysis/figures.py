"""Paper 2 figures (PNG 300 dpi + PDF). Palette: validated reference slots 1-2 (blue, orange)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "tables")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False})
runs = pd.read_csv(os.path.join(T, "all_runs.csv"))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def forest(pairs, ref, seeds, labels, title, xlabel, name):
    """Paired difference vs a reference: mean + 95% CI, with the individual seed differences."""
    P = pd.read_csv(pairs).set_index("comparison")
    fig, ax = plt.subplots(figsize=(5.2, 0.55 * len(labels) + 0.9))
    for i, (v, lab) in enumerate(labels):
        r = P.loc[f"{v} - {ref}"]
        a = runs[(runs.variant == v) & runs.seed.isin(seeds)].set_index("seed").test_loss
        b = runs[(runs.variant == ref) & runs.seed.isin(seeds)].set_index("seed").test_loss
        d = (a - b).dropna()
        y = len(labels) - 1 - i
        ax.scatter(d.values, np.full(len(d), y) + np.linspace(-0.12, 0.12, len(d)), s=12, color=INK2, alpha=0.55, zorder=2, linewidths=0)
        ax.errorbar(r.mean_diff, y, xerr=[[r.mean_diff - r.ci_low], [r.ci_high - r.mean_diff]], fmt="o", ms=6,
                    color=BLUE, ecolor=BLUE, elinewidth=2, capsize=0, zorder=3)
        ax.text(1.03, y, f"{r.mean_diff:+.4f}  [{r.ci_low:+.4f}, {r.ci_high:+.4f}]", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([lab for _, lab in labels][::-1])
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.text(1.03, len(labels) - 0.45, "mean  [95% CI]", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2)
    save(fig, name)


forest(os.path.join(T, "A_paired.csv"), "A1_standard", [0, 1, 2, 3, 4],
       [("A2_hybrid", "A2 Hybrid"), ("A4_distance_only", "A4 Distance only"), ("A3_pure", "A3 Pure")],
       "Experiment A: test loss relative to standard attention (5 seeds)",
       "Paired difference in test loss vs A1 (lower is better)", "fig1_expA_paired")

forest(os.path.join(T, "D_paired_vs_A2.csv"), "A2_hybrid", [0, 1, 2],
       [("D1a_floor_0.02", "D1a  λ floor 0.02"), ("D1b_floor_0.05", "D1b  λ floor 0.05"),
        ("D3_bounded", "D3  Bounded kernel"), ("D4_bounded_learned_sigma", "D4  Bounded, learned σ")],
       "Experiment D: stabilisers relative to the hybrid A2 (3 seeds)",
       "Paired difference in test loss vs A2 (lower is better)", "fig4_expD_paired")

# ---- Figure 2: gravitational use over training in A2 (three panels, shared x; no dual axis)
tr = pd.read_csv(os.path.join(T, "diag_trajectories.csv"))
a2 = tr[tr.variant == "A2_hybrid"].groupby("step")[["lam", "C", "D"]].agg(["mean", "min", "max"])
fig, axs = plt.subplots(1, 3, figsize=(9, 2.6), sharex=True)
for ax, col, lab in zip(axs, ["lam", "C", "D"], ["Mixing weight λ (mean over heads)", "Contribution C", "Attention shift D (nats)"]):
    m = a2[col]
    ax.fill_between(m.index, m["min"], m["max"], color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(m.index, m["mean"], color=BLUE, lw=2)
    ax.set_title(lab, loc="left", fontsize=9, color=INK)
    ax.set_xlabel("Training step")
    ax.set_xticks([0, 10000, 20000])
    ax.set_xticklabels(["0", "10k", "20k"])
axs[1].axhline(0.05, color=INK2, lw=1, ls="--")
axs[1].annotate("suppression threshold", (20000, 0.05), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=7, color=INK2)
axs[2].axhline(0.01, color=INK2, lw=1, ls="--")
fig.suptitle("A2 hybrid: gravitational contribution grows throughout training (line: mean of 5 seeds; band: range)",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout()
save(fig, "fig2_A2_contribution")

# ---- Figure 3: Experiment B cell means
cells = pd.read_csv(os.path.join(T, "B_cells.csv"))
core = runs[runs.seed.isin([0, 1, 2])]
a1 = core[core.variant == "A1_standard"].test_loss.mean()
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True)
order = ["learned", "floor", "fixed"]
for ax, mass in zip(axs, ["learned", "norm"]):
    for j, (kern, col, lab) in enumerate([("inverse_square", BLUE, "Inverse-square"), ("l2", ORANGE, "L2 (linear)")]):
        for k, mix in enumerate(order):
            c = cells[(cells.mass == mass) & (cells.kernel == kern) & (cells.mixing == mix)].iloc[0]
            pts = core[(core["mode"] == "hybrid") & (core.mass == mass) & (core.kernel == kern) & (core.mixing == mix)
                       & core.variant.str.match(r"^(A2_hybrid|B_)") & ~core.variant.str.startswith("B_mass_only")].test_loss
            x = k + (j - 0.5) * 0.28
            ax.scatter(np.full(len(pts), x), pts, s=10, color=col, alpha=0.45, linewidths=0, zorder=2)
            ax.plot([x - 0.1, x + 0.1], [c.test_mean] * 2, color=col, lw=2.5, solid_capstyle="round", zorder=3,
                    label=lab if k == 0 else None)
    ax.axhline(a1, color=INK2, lw=1, ls="--", label="A1 standard (mean)" if mass == "learned" else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["learned λ", "λ floor", "fixed λ"])
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{mass.capitalize()} mass", loc="left", fontsize=9, color=INK)
axs[0].set_ylabel("Test loss (lower is better)")
axs[0].legend(loc="upper left", fontsize=8)
fig.suptitle("Experiment B: learned mass helps with either kernel; norm mass only with L2 (line: mean of 3 seeds; dots: seeds)",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.tight_layout()
save(fig, "fig3_expB_cells")
print(sorted(os.listdir(OUT)))
