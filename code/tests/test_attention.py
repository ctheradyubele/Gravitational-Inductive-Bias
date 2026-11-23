"""Unit tests: the identities in the Phase 2 specification, numerical safety and wiring.

Run: python -m pytest -q tests
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from gib import variants
from gib.attention import AttnConfig, GravitationalSelfAttention, inv_softplus, pairwise_sq_dist
from gib.data import Masker
from gib.model import MLMEncoder, ModelConfig

D, H, T, B = 32, 4, 9, 3
torch.set_default_dtype(torch.float64)   # identities are checked tightly in float64


def make(cfg, seed=0):
    torch.manual_seed(seed)
    m = GravitationalSelfAttention(D, H, cfg).eval()
    if cfg.gravitational:
        with torch.no_grad():
            m.log_g.fill_(math.log(0.7))
    return m


def inputs(seed=1, pad=True):
    torch.manual_seed(seed)
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    if pad:
        mask[1, 6:] = False
        mask[2, 3:] = False
    return x, mask


def probs(m, x, mask):
    m.record = False
    captured = {}
    orig = torch.softmax

    def spy(s, dim):
        p = orig(s, dim=dim)
        captured["p"] = p
        return p
    torch.softmax = spy
    try:
        y = m(x, mask)
    finally:
        torch.softmax = orig
    return y, captured["p"]


def copy_shared(dst, src):
    dst.qkv.load_state_dict(src.qkv.state_dict())
    dst.out.load_state_dict(src.out.state_dict())


# ----------------------------------------------------------------------------- identities
def test_pairwise_distance_matches_cdist():
    q, k = torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    assert torch.allclose(pairwise_sq_dist(q, k), torch.cdist(q, k) ** 2, atol=1e-10)


def test_hybrid_with_lambda_zero_equals_standard():
    std = make(AttnConfig(mode="standard"))
    hyb = make(AttnConfig(mode="hybrid", lambda_init=0.0))
    copy_shared(hyb, std)
    x, mask = inputs()
    assert torch.allclose(std(x, mask), hyb(x, mask), atol=1e-10)


@pytest.mark.parametrize("lam", [0.1, 0.5, 3.0])
def test_collapse_no_mass_no_distance_is_standard(lam):
    """Specification: g = G is constant over keys and cancels in softmax."""
    std = make(AttnConfig(mode="standard"))
    cell = make(AttnConfig(mode="hybrid", mass="none", kernel="none", lambda_init=lam))
    copy_shared(cell, std)
    x, mask = inputs()
    assert torch.allclose(std(x, mask), cell(x, mask), atol=1e-9)


def test_collapse_no_mass_l2_is_rescaled_dot_plus_key_norm_bias():
    """Specification: s = (1/sqrt(dk) + 2 lam G) q.k - lam G ||k||^2 (the ||q||^2 term cancels)."""
    lam = 0.3
    cell = make(AttnConfig(mode="hybrid", mass="none", kernel="l2", lambda_init=lam))
    x, mask = inputs()
    _, p = probs(cell, x, mask)
    q, k, _ = cell._qkv(x)
    G = float(cell.G()[0])
    s = (1 / math.sqrt(cell.dk) + 2 * lam * G) * (q @ k.transpose(-1, -2)) - lam * G * k.pow(2).sum(-1)[:, :, None, :]
    s = s.masked_fill(~mask[:, None, None, :], torch.finfo(s.dtype).min)
    assert torch.allclose(p, torch.softmax(s, -1), atol=1e-9)


def test_query_mass_does_not_cancel():
    """Specification rule 2: m_i scales the whole row, so it changes attention (no log)."""
    cfg = AttnConfig(mode="hybrid", mass="learned", kernel="none", lambda_init=1.0)
    m = make(cfg)
    x, mask = inputs(pad=False)
    _, p1 = probs(m, x, mask)
    m.mass_override = lambda mass: mass * torch.linspace(0.5, 2.0, T)[None]   # rescale m_i per position
    _, p2 = probs(m, x, mask)
    assert not torch.allclose(p1, p2, atol=1e-6)


def test_mass_with_l2_does_not_collapse():
    cfg = AttnConfig(mode="hybrid", mass="learned", kernel="l2", lambda_init=0.3)
    cell, ref = make(cfg), make(AttnConfig(mode="hybrid", mass="none", kernel="l2", lambda_init=0.3))
    copy_shared(ref, cell)
    with torch.no_grad():
        cell.mass_head.weight.normal_(0, 0.5)
    x, mask = inputs()
    assert not torch.allclose(cell(x, mask), ref(x, mask), atol=1e-6)


def test_pure_has_no_lambda_and_hybrid_does():
    assert not hasattr(make(AttnConfig(mode="pure")), "lam_raw")
    assert hasattr(make(AttnConfig(mode="hybrid")), "lam_raw")


# ----------------------------------------------------------------------------- parameterisations
def test_floor_lambda_bounds_and_init():
    m = make(AttnConfig(mode="hybrid", mixing="floor", lambda_min=0.05, lambda_init=0.1))
    assert torch.allclose(m.lam(), torch.full((H,), 0.1), atol=1e-12)
    with torch.no_grad():
        m.lam_raw.fill_(-3.0)
    assert (m.lam() >= 0.05).all()
    m.project_()
    assert torch.allclose(m.lam_raw, torch.full((H,), 0.05))


def test_floor_learns_at_same_rate_as_learned_lambda():
    """Pilot fix: away from the bound, a floor variant's lambda must follow the learned variant exactly."""
    x, mask = inputs()
    out = []
    for mixing in ("learned", "floor"):
        m = make(AttnConfig(mode="hybrid", mixing=mixing, lambda_min=0.02)).train()
        opt = torch.optim.SGD([m.lam_raw], lr=1e-3)
        for _ in range(3):
            opt.zero_grad()
            m(x, mask)[mask].sum().backward()
            opt.step()
            m.project_()
        out.append(m.lam().detach().clone())
    assert (out[0] > 0.02).all() and not torch.allclose(out[0], torch.full((H,), 0.1))   # moved, off the bound
    assert torch.allclose(out[0], out[1], atol=1e-12)


def test_fixed_lambda_is_not_trainable():
    m = make(AttnConfig(mode="hybrid", mixing="fixed", lambda_fixed=0.5))
    assert torch.allclose(m.lam(), torch.full((H,), 0.5))
    assert all(p is not m.lam_const for p in m.parameters())


def test_g_frozen_by_default_and_learnable_when_asked():
    assert not make(AttnConfig(mode="hybrid")).log_g.requires_grad
    assert make(AttnConfig(mode="hybrid", freeze_g=False)).log_g.requires_grad


def test_bounded_kernel_in_unit_interval_and_finite():
    m = make(AttnConfig(mode="hybrid", kernel="bounded"))
    d = torch.tensor([0.0, 1e-9, 1.0, 1e6]).view(1, 1, 1, 4).expand(1, H, 1, 4)
    k = m.kernel(d)
    assert (k > 0).all() and (k <= 1).all() and torch.isfinite(k).all()
    assert torch.allclose(k[..., 0], torch.ones_like(k[..., 0]))


def test_learned_sigma_floor_and_start_at_median():
    m = make(AttnConfig(mode="hybrid", kernel="bounded", learn_sigma=True, sigma_min_frac=0.1))
    x, mask = inputs()
    out = m.calibrate(x, mask)
    assert torch.allclose(m.sigma(), torch.tensor(out["sigma"]), rtol=1e-6)
    with torch.no_grad():
        m.nu.fill_(-50.0)
    assert (m.sigma() >= 0.1 * m.sigma_ref - 1e-12).all()


def test_inv_softplus_roundtrip():
    for y in (1e-4, 0.08, 1.0, 30.0):
        assert math.isclose(float(torch.nn.functional.softplus(torch.tensor(inv_softplus(y)))), y, rel_tol=1e-9)


def test_learned_mass_starts_at_one():
    m = make(AttnConfig(mode="hybrid", mass="learned"))
    with torch.no_grad():
        m.mass_head.weight.zero_()
    assert torch.allclose(m.compute_mass(torch.randn(2, 5, D)), torch.ones(2, 5))


def test_norm_mass_is_scaled_by_sqrt_dim():
    m = make(AttnConfig(mode="hybrid", mass="norm"))
    x = torch.ones(1, 2, D)
    assert torch.allclose(m.compute_mass(x), torch.ones(1, 2))


# ----------------------------------------------------------------------------- calibration
@pytest.mark.parametrize("kernel", ["inverse_square", "l2", "bounded", "none"])
def test_calibration_matches_median_rule(kernel):
    cfg = AttnConfig(mode="hybrid", kernel=kernel)
    m = make(cfg)
    x, mask = inputs()
    m.calibrate(x, mask)
    r = m.raw_terms(x)
    pairs = (mask[:, :, None] & mask[:, None, :])[:, None].expand(B, H, T, T)
    g = (m.G().view(1, -1, 1, 1) * r["mm"] * m.kernel(r["d"])).abs()
    for h in range(H):
        med = g[:, h][pairs[:, h]].median()
        sd = r["s_dot"][:, h][pairs[:, h]].std()
        assert torch.isclose(med, sd, rtol=1e-6)
    assert bool(m.calibrated)


# ----------------------------------------------------------------------------- masking & safety
@pytest.mark.parametrize("name", sorted(variants.VARIANTS))
def test_padding_does_not_leak(name):
    cfg = variants.get(name)
    m = make(cfg)
    x, mask = inputs()
    y1 = m(x, mask)
    x2 = x.clone()
    x2[~mask] = torch.randn(int((~mask).sum()), D) * 100
    y2 = m(x2, mask)
    assert torch.allclose(y1[mask], y2[mask], atol=1e-8), name


@pytest.mark.parametrize("name", sorted(variants.VARIANTS))
def test_gradients_reach_gravitational_parameters(name):
    cfg = variants.get(name)
    m = make(cfg).train()
    x, mask = inputs()
    m(x, mask)[mask].pow(2).sum().backward()
    assert m.qkv.weight.grad is not None and torch.isfinite(m.qkv.weight.grad).all()
    for p in m.grav_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_inverse_square_identical_query_key_is_finite():
    m = make(AttnConfig(mode="hybrid", kernel="inverse_square"))
    with torch.no_grad():                         # make q == k exactly so d = 0
        w = m.qkv.weight
        w[D:2 * D] = w[:D]
        m.qkv.bias[D:2 * D] = m.qkv.bias[:D]
    x, mask = inputs()
    assert torch.isfinite(m(x, mask)).all()


# ----------------------------------------------------------------------------- diagnostics
def test_diagnostics_zero_contribution_when_lambda_zero():
    m = make(AttnConfig(mode="hybrid", lambda_init=0.0))
    m.record = True
    x, mask = inputs()
    m(x, mask)
    assert max(m.last["C"]) < 1e-12 and max(abs(v) for v in m.last["D"]) < 1e-9


def test_diagnostics_positive_contribution_when_lambda_large():
    m = make(AttnConfig(mode="hybrid", lambda_init=1.0))
    m.calibrate(*inputs())
    m.record = True
    m(*inputs())
    assert min(m.last["C"]) > 0.05 and min(m.last["D"]) > 0
    assert set(m.last) >= {"lambda", "G", "lambda_G", "C", "D", "entropy", "g_q", "d_q", "s_q", "mass_mean"}


# ----------------------------------------------------------------------------- variants & model
def test_variant_registry_matches_specification():
    assert len(variants.unique_runs()) == 20
    assert len(variants.EXPERIMENTS["B"]) == 12
    assert variants.get("B_learned_inverse_square_learned") == variants.get("A2_hybrid")
    names = set(variants.VARIANTS)
    assert not any("none_none" in n or "none_l2" in n for n in names)   # collapsed cells are absent


@pytest.mark.parametrize("name", variants.unique_runs(include_optional=True))
def test_model_forward_backward_every_variant(name):
    torch.manual_seed(0)
    mc = ModelConfig(vocab_size=50, max_len=16, dim=32, layers=2, heads=4, ffn=64, dropout=0.0, attn_dropout=0.0)
    model = MLMEncoder(mc, variants.get(name))
    ids = torch.randint(5, 50, (4, 16))
    ids[1, 12:] = 0
    model.calibrate(ids)
    labels = torch.full_like(ids, -100)
    labels[:, 3] = ids[:, 3]
    loss, logits = model(ids, labels)
    loss.backward()
    assert torch.isfinite(loss) and logits.shape == (4, 16, 50)
    groups = model.param_groups(1e-3, 1e-3)
    assert sum(len(g["params"]) for g in groups) == sum(1 for p in model.parameters() if p.requires_grad)


def test_masker_rate_specials_and_determinism():
    mk = Masker(vocab_size=100, mask_id=4, special_ids=[0, 1, 2, 3, 4])
    ids = torch.randint(5, 100, (64, 128))
    ids[:, 0], ids[:, -1] = 2, 3
    x1, y1 = mk(ids, torch.Generator().manual_seed(1234))
    x2, y2 = mk(ids, torch.Generator().manual_seed(1234))
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    rate = float((y1 != -100).float().mean())
    assert 0.13 < rate < 0.16
    assert (y1[:, 0] == -100).all() and (y1[:, -1] == -100).all()
    chosen = y1 != -100
    assert 0.75 < float((x1[chosen] == 4).float().mean()) < 0.85


def test_shared_weights_identical_across_variants():
    """Same seed -> identical embeddings, projections and feed-forward weights for every variant."""
    mc = ModelConfig(vocab_size=50, max_len=16, dim=32, layers=2, heads=4, ffn=64)

    def shared(name):
        torch.manual_seed(7)
        m = MLMEncoder(mc, variants.get(name))
        sd = m.state_dict()
        keep = {k: v for k, v in sd.items() if not any(s in k for s in ("mass_head", "log_g", "lam", "rho", "nu", "sigma", "calibrated"))}
        return keep, torch.rand(1)   # RNG state after construction must match too
    ref, r0 = shared("A1_standard")
    for name in variants.unique_runs(include_optional=True):
        sd, r = shared(name)
        assert sd.keys() == ref.keys(), name
        assert all(torch.equal(sd[k], ref[k]) for k in ref), name
        assert torch.equal(r, r0), name
