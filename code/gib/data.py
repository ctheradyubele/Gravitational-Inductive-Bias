"""WikiText-103 data and masked-LM masking, matching Paper 1's Phase A.

Chunking is copied from Paper 1 (scripts/adapt_mlm.py: chunk_corpus): the corpus is tokenised without
special tokens, concatenated, cut into blocks of seq_len - 2 and wrapped as [CLS] ... [SEP].
Masking follows BERT / DataCollatorForLanguageModeling: 15% of non-special tokens, of which 80% become
[MASK], 10% a random token and 10% stay unchanged. It is reimplemented here so that a torch.Generator
controls it (evaluation masking is identical for every variant: seed 1234).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

EVAL_MASK_SEED = 1234


def load_wikitext(cache_dir: Optional[str] = None) -> dict:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)
    return {s: [t for t in ds[s]["text"] if t.strip()] for s in ("train", "validation", "test")}


def chunk_corpus(tokenizer, texts, seq_len: int) -> np.ndarray:
    parts = []
    for i in range(0, len(texts), 10000):
        enc = tokenizer(texts[i:i + 10000], add_special_tokens=False)["input_ids"]
        parts.append(np.fromiter((t for e in enc for t in e), dtype=np.int32))
    ids = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    body = seq_len - 2
    n = len(ids) // body
    arr = ids[: n * body].astype(np.int64).reshape(n, body)
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    return np.concatenate([np.full((n, 1), cls), arr, np.full((n, 1), sep)], axis=1)


def synthetic_corpus(n: int, seq_len: int, vocab_size: int, special: int = 5, seed: int = 0) -> np.ndarray:
    """Structured toy sequences (smoke tests only, never for reported results): each sequence repeats
    a short random motif, so a working model can learn to fill masked tokens from context."""
    rng = np.random.RandomState(seed)
    out = np.zeros((n, seq_len), dtype=np.int64)
    for i in range(n):
        motif = rng.randint(special, vocab_size, size=rng.randint(3, 8))
        body = np.resize(motif, seq_len - 2)
        out[i] = np.concatenate([[2], body, [3]])            # toy vocab: 0 PAD, 1 UNK, 2 CLS, 3 SEP, 4 MASK
    return out


class Masker:
    def __init__(self, vocab_size: int, mask_id: int, special_ids, prob: float = 0.15):
        self.vocab_size, self.mask_id, self.prob = vocab_size, mask_id, prob
        self.special = torch.tensor(sorted(set(int(s) for s in special_ids)), dtype=torch.long)

    def __call__(self, ids: torch.Tensor, gen: Optional[torch.Tensor] = None):
        """ids [B, T] (CPU) -> (inputs, labels); labels are -100 where not predicted."""
        ids = ids.clone()
        labels = ids.clone()
        special = torch.isin(ids, self.special)
        p = torch.full(ids.shape, self.prob)
        p[special] = 0.0
        chosen = torch.bernoulli(p, generator=gen).bool()
        labels[~chosen] = -100
        to_mask = torch.bernoulli(torch.full(ids.shape, 0.8), generator=gen).bool() & chosen
        ids[to_mask] = self.mask_id
        to_rand = torch.bernoulli(torch.full(ids.shape, 0.5), generator=gen).bool() & chosen & ~to_mask
        ids[to_rand] = torch.randint(self.vocab_size, ids.shape, generator=gen)[to_rand]
        return ids, labels

    @classmethod
    def from_tokenizer(cls, tok, prob: float = 0.15):
        return cls(len(tok), tok.mask_token_id, tok.all_special_ids, prob)
