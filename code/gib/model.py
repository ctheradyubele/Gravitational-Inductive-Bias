"""Small pre-LayerNorm Transformer encoder for masked language modelling, trained from scratch.

Only the attention module changes between variants; embeddings, feed-forward blocks, the MLM head and
initialisation are identical for every configuration.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttnConfig, GravitationalSelfAttention


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 30522          # bert-base-uncased WordPiece
    max_len: int = 128
    dim: int = 512
    layers: int = 6
    heads: int = 8
    ffn: int = 2048
    dropout: float = 0.1
    attn_dropout: float = 0.1
    pad_id: int = 0

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, mc: ModelConfig, ac: AttnConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(mc.dim)
        self.attn = GravitationalSelfAttention(mc.dim, mc.heads, ac, mc.attn_dropout)
        self.ln2 = nn.LayerNorm(mc.dim)
        self.ff = nn.Sequential(nn.Linear(mc.dim, mc.ffn), nn.GELU(), nn.Linear(mc.ffn, mc.dim))
        self.drop = nn.Dropout(mc.dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.ff(self.ln2(x)))


class MLMEncoder(nn.Module):
    def __init__(self, mc: ModelConfig, ac: AttnConfig):
        super().__init__()
        self.mc, self.ac = mc, ac
        # Shared weights must be identical across variants for a given seed. Gravitational modules
        # consume random numbers at construction, so the RNG state is saved here and restored before
        # the shared initialisation below; afterwards the RNG is in the same state for every variant.
        rng_state = torch.random.get_rng_state()
        self.tok = nn.Embedding(mc.vocab_size, mc.dim, padding_idx=mc.pad_id)
        self.pos = nn.Embedding(mc.max_len, mc.dim)
        self.drop = nn.Dropout(mc.dropout)
        self.blocks = nn.ModuleList(Block(mc, ac) for _ in range(mc.layers))
        self.ln_f = nn.LayerNorm(mc.dim)
        self.head_transform = nn.Sequential(nn.Linear(mc.dim, mc.dim), nn.GELU(), nn.LayerNorm(mc.dim))
        self.head_bias = nn.Parameter(torch.zeros(mc.vocab_size))
        torch.random.set_rng_state(rng_state)
        self.apply(self._init)
        # GPT-2 style scaling of residual output projections for depth
        for b in self.blocks:
            nn.init.normal_(b.attn.out.weight, std=0.02 / math.sqrt(2 * mc.layers))
            nn.init.normal_(b.ff[2].weight, std=0.02 / math.sqrt(2 * mc.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            if m.out_features == 1:   # the learned mass head keeps its own initialisation (m = 1)
                return
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def attn_modules(self):
        return [b.attn for b in self.blocks]

    def embed(self, ids):
        t = ids.shape[1]
        return self.drop(self.tok(ids) + self.pos(torch.arange(t, device=ids.device))[None])

    def forward(self, ids: torch.Tensor, labels: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = ids != self.mc.pad_id
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x, mask)
        h = self.head_transform(self.ln_f(x))
        logits = h @ self.tok.weight.t() + self.head_bias                   # tied output embedding
        if labels is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), labels.view(-1), ignore_index=-100)
        return loss, logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> list:
        """Calibrate G (and sigma) layer by layer, bottom up, on one batch (eval mode, no dropout).

        Each layer is calibrated on the input it receives after the layers below were calibrated.
        """
        was = self.training
        self.eval()
        if mask is None:
            mask = ids != self.mc.pad_id
        x = self.embed(ids)
        out = []
        for b in self.blocks:
            out.append(b.attn.calibrate(b.ln1(x), mask))
            x = b(x, mask)
        self.train(was)
        return out

    def project_(self):
        for a in self.attn_modules():
            a.project_()

    def set_record(self, flag: bool):
        for a in self.attn_modules():
            a.record = flag
            if not flag:
                a.last = {}

    def diagnostics(self) -> list:
        return [a.last for a in self.attn_modules()]

    def grav_parameters(self):
        for a in self.attn_modules():
            yield from a.grav_parameters()

    def param_groups(self, lr: float, grav_lr: float, weight_decay: float = 0.01):
        grav = {id(p) for p in self.grav_parameters()}
        decay, no_decay, gp = [], [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in grav:
                gp.append(p)
            elif p.ndim < 2:   # biases, LayerNorm weights, head bias: no weight decay

                no_decay.append(p)
            else:
                decay.append(p)
        groups = [{"params": decay, "lr": lr, "weight_decay": weight_decay},
                  {"params": no_decay, "lr": lr, "weight_decay": 0.0}]
        if gp:
            groups.append({"params": gp, "lr": grav_lr, "weight_decay": 0.0})
        return [g for g in groups if g["params"]]

    def grav_state(self) -> dict:
        """Per-layer, per-head lambda, G and sigma for result files."""
        lam, G, sig = [], [], []
        for a in self.attn_modules():
            gr = a.cfg.gravitational
            lam.append(a.lam().detach().cpu().tolist() if a.cfg.mode == "hybrid" else None)
            G.append(a.G().detach().cpu().tolist() if gr else None)
            sig.append(a.sigma().detach().cpu().tolist() if gr and a.cfg.kernel == "bounded" else None)
        return {"lambda": lam, "G": G, "sigma": sig}
