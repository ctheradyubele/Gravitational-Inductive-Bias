"""Reproducibility helpers (adapted from Paper 1's gam/utils.py)."""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(enabled: bool):
    if enabled and torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


def git_commit() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "git_commit": git_commit(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "argv": sys.argv}
    for mod in ("transformers", "datasets"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return env


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def append_jsonl(path: str, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
