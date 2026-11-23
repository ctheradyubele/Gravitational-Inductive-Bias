"""Gravitational self-attention for Paper 2 (from-scratch study).

One module implements every configuration in the Phase 2 specification, so that the baseline, the
grid cells and the stabilisation variants all run through identical code.

Per layer l and head h, with hidden states x_i, queries q_i and keys k_j (d_k per head):

    s_dot_ij = q_i . k_j / sqrt(d_k)
    d_ij     = ||q_i - k_j||^2
    g_ij     = G_h * m_i * m_j * K(d_ij)

    standard : s = s_dot
    hybrid   : s = s_dot + lambda_h * g
    pure     : s = g
    l2dot    : s = -d / sqrt(d_k)             (Paper 1 "l2" control, optional A5)

Factor levels (specification, section "Factor levels"):

    mass    learned  m_i = softplus(w^T x_i + b), m_i = 1 at initialisation
            norm     m_i = ||x_i||_2 / sqrt(D)
            none     m_i = 1
    kernel  inverse_square  K = 1 / (d + eps)
            l2              K = -d
            bounded         K = 1 / (1 + d / sigma_h)     (D3; D4 learns sigma_h with a floor)
            none            K = 1
    mixing  learned  lambda_h unconstrained, init 0.1
            floor    lambda_h >= lambda_min, init 0.1: same parameter and learning rate as "learned", projected
             back onto [lambda_min, inf) after every optimiser step (a softplus reparameterisation was
             rejected after the pilot: it slowed lambda's effective learning rate about 12-fold)
            fixed    lambda_h = constant (0.5), not trained

G_h is set by `calibrate` (median g = std of s_dot over valid pairs, Paper 1's rule) and frozen by
default, so that lambda is the only mixing scale. All gravitational arithmetic runs in float32.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("standard", "hybrid", "pure", "l2dot")
MASSES = ("learned", "norm", "none")
KERNELS = ("inverse_square", "l2", "bounded", "none")
MIXINGS = ("learned", "floor", "fixed")

MASS_BIAS_INIT = math.log(math.e - 1.0)  # softplus(b) = 1


def inv_softplus(y: float) -> float:
    return y + math.log(-math.expm1(-y))  # log(exp(y) - 1), numerically stable


@dataclass(frozen=True)
class AttnConfig:
    mode: str = "standard"
    mass: str = "learned"
    kernel: str = "inverse_square"
    mixing: str = "learned"
    lambda_init: float = 0.1
    lambda_min: float = 0.02          # used by mixing="floor"
    lambda_fixed: float = 0.5         # used by mixing="fixed"
    learn_sigma: bool = False         # D4: learn the bounded-kernel length scale
    sigma_min_frac: float = 0.1       # D4 floor, as a fraction of the calibrated median distance
    eps: float = 1e-6
    freeze_g: bool = True

    def __post_init__(self):
        assert self.mode in MODES, self.mode
        assert self.mass in MASSES, self.mass
        assert self.kernel in KERNELS, self.kernel
        assert self.mixing in MIXINGS, self.mixing
        if self.mixing == "floor":
            assert self.lambda_init > self.lambda_min, "floor needs lambda_init > lambda_min"
        if self.learn_sigma:
            assert self.kernel == "bounded", "learn_sigma only applies to the bounded kernel"

    @property
    def gravitational(self) -> bool:
        return self.mode in ("hybrid", "pure")

    def to_dict(self):
        return asdict(self)

    def but(self, **kw) -> "AttnConfig":
        return replace(self, **kw)


def pairwise_sq_dist(q: torch.Tensor, k: torch.Tensor, qk: Optional[torch.Tensor] = None) -> torch.Tensor:
    """||q_i - k_j||^2 for [..., T, d] inputs, reusing q.k when given."""
    if qk is None:
        qk = q @ k.transpose(-1, -2)
    d = q.pow(2).sum(-1).unsqueeze(-1) + k.pow(2).sum(-1).unsqueeze(-2) - 2.0 * qk
    return d.clamp_min(0.0)


def _masked_std(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Std over the last dim counting only valid entries. x [..., T], valid broadcastable bool."""
    w = valid.to(x.dtype).expand_as(x)
    n = w.sum(-1).clamp_min(1.0)
    mean = (x * w).sum(-1) / n
    var = (((x - mean.unsqueeze(-1)) ** 2) * w).sum(-1) / n
    return var.clamp_min(0.0).sqrt()


class GravitationalSelfAttention(nn.Module):
    """Multi-head self-attention (bidirectional). x: [B, T, D]; key_padding_mask: [B, T] bool, True = real token."""

    def __init__(self, dim: int, num_heads: int, cfg: AttnConfig, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim, self.h, self.dk = dim, num_heads, dim // num_heads
        self.cfg = cfg
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        H = num_heads
        if cfg.gravitational:
            if cfg.mass == "learned":
                self.mass_head = nn.Linear(dim, 1)
                nn.init.normal_(self.mass_head.weight, std=0.02)
                nn.init.constant_(self.mass_head.bias, MASS_BIAS_INIT)
            self.log_g = nn.Parameter(torch.zeros(H), requires_grad=not cfg.freeze_g)
            if cfg.mode == "hybrid":
                if cfg.mixing in ("learned", "floor"):
                    self.lam_raw = nn.Parameter(torch.full((H,), cfg.lambda_init))
                else:
                    self.register_buffer("lam_const", torch.full((H,), cfg.lambda_fixed))
            if cfg.kernel == "bounded":
                # sigma = median distance at calibration (D3: frozen; D4: learned above a floor)
                self.register_buffer("sigma_ref", torch.ones(H))
                if cfg.learn_sigma:
                    self.nu = nn.Parameter(torch.zeros(H))
        self.register_buffer("calibrated", torch.tensor(False))
        self.record = False
        self.last: dict = {}
        self.mass_override = None   # callable [B,T] -> [B,T]; used only by the mass-shuffle tests

    # ------------------------------------------------------------------ parameters
    def lam(self) -> torch.Tensor:
        c = self.cfg
        if c.mixing == "learned":
            return self.lam_raw
        if c.mixing == "floor":
            return self.lam_raw.clamp_min(c.lambda_min)
        return self.lam_const

    @torch.no_grad()
    def project_(self):
        """Call after every optimiser step: keeps the floor variant's parameter inside its bound."""
        if self.cfg.mode == "hybrid" and self.cfg.mixing == "floor":
            self.lam_raw.clamp_(min=self.cfg.lambda_min)

    def sigma(self) -> torch.Tensor:
        if not self.cfg.learn_sigma:
            return self.sigma_ref
        floor = self.cfg.sigma_min_frac * self.sigma_ref
        return floor + F.softplus(self.nu)

    def G(self) -> torch.Tensor:
        return self.log_g.exp()

    def compute_mass(self, x: torch.Tensor) -> torch.Tensor:
        """Called inside raw_terms, where autocast is off; x is already float32 (or float64)."""
        c = self.cfg.mass
        if c == "learned":
            w, b = self.mass_head.weight.to(x.dtype), self.mass_head.bias.to(x.dtype)
            return F.softplus(F.linear(x, w, b).squeeze(-1))
        if c == "norm":
            return x.norm(dim=-1) / math.sqrt(self.dim)
        return torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)

    def kernel(self, d: torch.Tensor) -> torch.Tensor:
        k = self.cfg.kernel
        if k == "inverse_square":
            return 1.0 / (d + self.cfg.eps)
        if k == "l2":
            return -d
        if k == "bounded":
            return 1.0 / (1.0 + d / self.sigma().view(1, -1, 1, 1))
        return torch.ones_like(d)

    # ------------------------------------------------------------------ forward
    def _qkv(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        return q, k, v                                                     # each [B,H,T,dk]

    def raw_terms(self, x: torch.Tensor):
        """Unscaled pieces used by forward and by calibration.

        Scores, distances and masses are computed with autocast disabled, in float32 (float64 if the
        input is float64): ||q||^2 + ||k||^2 - 2 q.k cancels badly in bfloat16 when q is close to k,
        which is exactly where the inverse-square kernel is largest.
        """
        q, k, v = self._qkv(x)
        ft = torch.promote_types(q.dtype, torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            qf, kf = q.to(ft), k.to(ft)
            qk = qf @ kf.transpose(-1, -2)
            s_dot = qk / math.sqrt(self.dk)
            out = {"q": q, "k": k, "v": v, "qk": qk, "s_dot": s_dot}
            c = self.cfg
            if c.gravitational or c.mode == "l2dot":
                out["d"] = pairwise_sq_dist(qf, kf, qk)
            if c.gravitational:
                out.update(self._mass_terms(x.to(ft)))
        return out

    def _mass_terms(self, x):
        m = self.compute_mass(x)                                           # [B,T]
        if self.mass_override is not None:                                 # evaluation-only mass tests
            m = self.mass_override(m)
        return {"mass": m, "mm": m[:, None, :, None] * m[:, None, None, :]}   # mm [B,1,T,T]

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r = self.raw_terms(x)
        c = self.cfg
        g = None
        if c.mode == "standard":
            s = r["s_dot"]
        elif c.mode == "l2dot":
            s = -r["d"] / math.sqrt(self.dk)
        else:
            g = self.G().view(1, -1, 1, 1) * r["mm"] * self.kernel(r["d"])
            s = g if c.mode == "pure" else r["s_dot"] + self.lam().view(1, -1, 1, 1) * g
        if key_padding_mask is not None:
            s = s.masked_fill(~key_padding_mask[:, None, None, :], torch.finfo(s.dtype).min)
        p = torch.softmax(s, dim=-1)
        ctx = self.dropout(p).to(r["v"].dtype) @ r["v"]                     # [B,H,T,dk]
        b, _, t, _ = ctx.shape
        y = self.out(ctx.transpose(1, 2).reshape(b, t, self.dim))
        if self.record:
            self._record(r, g, s, p, key_padding_mask)
        return y

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def _record(self, r, g, s, p, key_padding_mask):
        """Per-head diagnostics (specification, section "Contribution diagnostic and logging")."""
        B, H, T, _ = r["s_dot"].shape
        if key_padding_mask is None:
            key_padding_mask = torch.ones(B, T, dtype=torch.bool, device=s.device)
        kv = key_padding_mask[:, None, None, :]                            # keys
        qv = key_padding_mask[:, None, :].expand(B, H, T)                  # queries
        pairs = (key_padding_mask[:, :, None] & key_padding_mask[:, None, :])[:, None].expand(B, H, T, T)

        def per_head_mean(x):                                              # x [B,H,T] over valid queries
            w = qv.float()
            return ((x * w).sum((0, 2)) / w.sum((0, 2)).clamp_min(1.0)).cpu().tolist()

        def quantiles(x):
            out = []
            for h in range(H):
                v = x[:, h][pairs[:, h]].float()
                if v.numel() > 200_000:
                    v = v[torch.randperm(v.numel(), device=v.device)[:200_000]]
                out.append(torch.quantile(v, torch.tensor([0.01, 0.5, 0.99], device=v.device, dtype=v.dtype)).cpu().tolist())
            return out

        logp = torch.log(p.clamp_min(1e-30))
        ent = -(p * logp).masked_fill(~kv, 0.0).sum(-1)                    # [B,H,T]
        rec = {"entropy": per_head_mean(ent), "s_q": quantiles(s.masked_fill(~kv, 0.0)),
               "max_score": [float(s[:, h][pairs[:, h]].max()) for h in range(H)]}
        if "d" in r:
            rec["d_q"] = quantiles(r["d"])
        if g is not None:
            m = r["mass"]
            mv = m[key_padding_mask]
            rec.update({"g_q": quantiles(g), "G": self.G().cpu().tolist(),
                        "mass_mean": float(mv.mean()), "mass_sd": float(mv.std()) if mv.numel() > 1 else 0.0,
                        "mass_p99": float(torch.quantile(mv, 0.99))})
            if self.cfg.kernel == "bounded":
                rec["sigma"] = self.sigma().cpu().tolist()
        if self.cfg.mode == "hybrid":
            lam = self.lam().view(1, -1, 1, 1)
            grav = lam * g
            C = _masked_std(grav, kv) / _masked_std(r["s_dot"], kv).clamp_min(1e-12)
            s_dot = r["s_dot"].masked_fill(~kv, torch.finfo(torch.float32).min)
            logq = torch.log_softmax(s_dot, dim=-1)
            D = (p * (logp - logq)).masked_fill(~kv, 0.0).sum(-1)
            lam_v = self.lam().cpu().tolist()
            rec.update({"lambda": lam_v, "lambda_G": [l * gg for l, gg in zip(lam_v, rec["G"])],
                        "C": per_head_mean(C), "D": per_head_mean(D)})
        self.last = rec

    # ------------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> dict:
        """Set sigma (bounded kernel) to the median distance, then G so that median |g| = std(s_dot).

        Uses valid query-key pairs of the batch x (the input this layer receives). Paper 1's rule, with
        |.| added because the l2 kernel is negative. Returns the values set.
        """
        if not self.cfg.gravitational:
            return {}
        r = self.raw_terms(x)
        B, H, T, _ = r["s_dot"].shape
        if key_padding_mask is None:
            key_padding_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        pairs = (key_padding_mask[:, :, None] & key_padding_mask[:, None, :])[:, None].expand(B, H, T, T)
        out = {}
        if self.cfg.kernel == "bounded":
            med_d = torch.stack([r["d"][:, h][pairs[:, h]].median() for h in range(H)]).clamp_min(1e-6)
            self.sigma_ref.copy_(med_d)
            if self.cfg.learn_sigma:  # start at the calibrated median: floor + softplus(nu) = median
                self.nu.copy_(torch.tensor([inv_softplus((1 - self.cfg.sigma_min_frac) * float(v)) for v in med_d],
                                           device=self.nu.device))
            out["sigma"] = med_d.cpu().tolist()
        raw = (r["mm"] * self.kernel(r["d"])).expand(B, H, T, T).abs()
        g = []
        for h in range(H):
            s = r["s_dot"][:, h][pairs[:, h]]
            med = raw[:, h][pairs[:, h]].median().clamp_min(1e-12)
            g.append(float(s.std() / med) if s.numel() > 1 else 1.0)
        self.log_g.copy_(torch.log(torch.tensor(g, device=self.log_g.device).clamp_min(1e-12)))
        self.calibrated.fill_(True)
        out["G"] = g
        return out

    # ------------------------------------------------------------------ bookkeeping
    def grav_parameters(self):
        for n in ("mass_head", "lam_raw", "log_g", "nu"):
            if hasattr(self, n):
                obj = getattr(self, n)
                if isinstance(obj, nn.Module):
                    yield from obj.parameters()
                elif isinstance(obj, nn.Parameter):
                    yield obj
