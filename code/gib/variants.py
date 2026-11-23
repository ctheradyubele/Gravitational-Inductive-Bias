"""Named configurations from the Phase 2 specification.

Experiment A  A1 standard, A2 hybrid, A3 pure, A4 distance-only control, A5 Paper 1 L2 (optional)
Experiment B  12 core cells  mass {learned, norm} x kernel {inverse_square, l2} x mixing {learned, floor, fixed}
              plus the mass-only control (learned mass, no distance, learned lambda)
Experiment D  D1a/D1b lambda floor 0.02/0.05, D3 bounded kernel, D4 bounded kernel with learned length scale

B_learned_inverse_square_learned is A2 and B_none_inverse_square_learned is A4; they are aliases so
that the same run serves both experiments. Collapsed cells (no mass + no distance, no mass + L2) are
not defined here on purpose: tests/test_attention.py proves that they reduce to other models.
"""
from __future__ import annotations

from .attention import AttnConfig

HYB = AttnConfig(mode="hybrid", mass="learned", kernel="inverse_square", mixing="learned")

VARIANTS: dict = {
    "A1_standard": AttnConfig(mode="standard"),
    "A2_hybrid": HYB,
    "A3_pure": AttnConfig(mode="pure", mass="learned", kernel="inverse_square"),
    "A4_distance_only": HYB.but(mass="none"),
    "A5_l2_paper1": AttnConfig(mode="l2dot"),
    "B_mass_only": HYB.but(kernel="none"),
    "D1a_floor_0.02": HYB.but(mixing="floor", lambda_min=0.02),
    "D1b_floor_0.05": HYB.but(mixing="floor", lambda_min=0.05),
    "D3_bounded": HYB.but(kernel="bounded"),
    "D4_bounded_learned_sigma": HYB.but(kernel="bounded", learn_sigma=True),
}

for _mass in ("learned", "norm"):
    for _kernel in ("inverse_square", "l2"):
        for _mixing in ("learned", "floor", "fixed"):
            VARIANTS[f"B_{_mass}_{_kernel}_{_mixing}"] = HYB.but(mass=_mass, kernel=_kernel, mixing=_mixing)

ALIASES = {"B_learned_inverse_square_learned": "A2_hybrid"}

EXPERIMENTS = {
    "A": ["A1_standard", "A2_hybrid", "A3_pure", "A4_distance_only"],
    "A_optional": ["A5_l2_paper1"],
    "B": [n for n in VARIANTS if n.startswith("B_") and n not in ALIASES],
    "D": ["D1a_floor_0.02", "D1b_floor_0.05", "D3_bounded", "D4_bounded_learned_sigma"],
}


def get(name: str) -> AttnConfig:
    return VARIANTS[ALIASES.get(name, name)]


def unique_runs(include_optional: bool = False) -> list:
    names = EXPERIMENTS["A"] + EXPERIMENTS["B"] + EXPERIMENTS["D"]
    if include_optional:
        names += EXPERIMENTS["A_optional"]
    return names
