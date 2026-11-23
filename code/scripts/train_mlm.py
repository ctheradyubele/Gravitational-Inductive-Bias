"""Train one configuration from random initialisation with masked LM on WikiText-103.

Every variant gets the same data order (by seed), masking, steps, batch size, schedule and model
size; only the attention module differs. Outputs in --out:

    results.json      configuration, environment, curves, final metrics, lambda/G/sigma, counters
    diagnostics.jsonl per-layer, per-head diagnostics every --diag_every steps (fixed evaluation batch)
    train_log.jsonl   loss, gradient norms, clipping and spike flags every --log_every steps
    final.pt          final weights (best.pt too, with --save_best)

Interrupted runs are restarted from scratch, as in Paper 1 (no partial run contributes to results).

Example (Colab, A100):
    python scripts/train_mlm.py --variant A2_hybrid --seed 0 --out runs/A2_hybrid/s0 --amp
Smoke test (CPU, a minute):
    python scripts/train_mlm.py --variant A2_hybrid --seed 0 --out /tmp/smoke --synthetic --steps 40 \
        --layers 2 --dim 64 --heads 4 --ffn 128 --batch_size 16 --micro_batch 16 --eval_every 20 --diag_every 20
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gib import variants
from gib.data import EVAL_MASK_SEED, Masker, chunk_corpus, load_wikitext, synthetic_corpus
from gib.model import MLMEncoder, ModelConfig
from gib.utils import append_jsonl, autocast, device, environment, set_seed, write_json


LOCK_STALE_S = 20 * 60


def get_data(a):
    if a.synthetic:
        V = 64
        train = synthetic_corpus(4000, a.seq_len, V, seed=0)
        val = synthetic_corpus(400, a.seq_len, V, seed=1)
        test = synthetic_corpus(400, a.seq_len, V, seed=2)
        masker = Masker(V, mask_id=4, special_ids=[0, 1, 2, 3, 4])
        return train, val, test, masker, V, 0
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    cache = os.path.join(a.cache_dir, f"wikitext103_bert_{a.seq_len}")
    if not os.path.exists(cache + "_train.npy"):
        os.makedirs(a.cache_dir, exist_ok=True)
        corpus = load_wikitext()
        for split in ("train", "validation", "test"):
            np.save(f"{cache}_{split}.npy", chunk_corpus(tok, corpus[split], a.seq_len))
    train = np.load(cache + "_train.npy")
    val = np.load(cache + "_validation.npy")[: a.max_eval]
    test = np.load(cache + "_test.npy")[: a.max_eval]
    return train, val, test, Masker.from_tokenizer(tok), len(tok), tok.pad_token_id


def lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(set(variants.VARIANTS) | set(variants.ALIASES)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=256, help="sequences per optimiser step")
    ap.add_argument("--micro_batch", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grav_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--train_fraction", type=float, default=1.0, help="data-efficiency runs: use a fixed subset")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--diag_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--max_eval", type=int, default=2000, help="validation/test sequences")
    ap.add_argument("--cache_dir", default="data_cache")
    ap.add_argument("--recalibrate_at", default="",
                    help="comma-separated steps at which G (and sigma) are recalibrated, e.g. 1000 = end of warmup")
    ap.add_argument("--learn_g", action="store_true", help="learn G (Paper 1) instead of freezing it after calibration")
    ap.add_argument("--lambda_init", type=float, default=None, help="override lambda_init (pilot only)")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()

    if os.path.exists(os.path.join(a.out, "results.json")):
        print("exists, skipping", a.out)
        return
    os.makedirs(a.out, exist_ok=True)
    # A lock stops two sessions (two Colab tabs, say) from training the same run into the same folder.
    # The owner refreshes it while training; a lock untouched for LOCK_STALE_S is treated as abandoned
    # (a disconnected session) and the run restarts from scratch.
    # The heartbeat time is written inside the file: Google Drive does not sync timestamp-only changes.
    lock = os.path.join(a.out, "RUNNING.lock")
    owner = f"{socket.gethostname()} pid {os.getpid()} started {time.strftime('%Y-%m-%d %H:%M:%S')}"

    def write_lock():
        with open(lock, "w") as f:
            json.dump({"owner": owner, "heartbeat": time.time()}, f)

    if os.path.exists(lock):
        try:
            info = json.load(open(lock))
            age = time.time() - float(info["heartbeat"])
        except Exception:                                   # old-format or partly written lock
            info, age = {}, time.time() - os.path.getmtime(lock)
        if age < LOCK_STALE_S:
            print("another session is running this run, skipping:", a.out, info.get("owner", ""), flush=True)
            return
    write_lock()
    for f in ("diagnostics.jsonl", "train_log.jsonl"):        # a restarted run starts clean
        if os.path.exists(os.path.join(a.out, f)):
            os.remove(os.path.join(a.out, f))

    set_seed(a.seed)
    dev = device()
    train, val, test, masker, V, pad_id = get_data(a)
    if a.train_fraction < 1.0:                                 # same subset for every variant and seed
        keep = np.random.RandomState(4242).permutation(len(train))[: int(len(train) * a.train_fraction)]
        train = train[np.sort(keep)]

    acfg = variants.get(a.variant)
    if a.learn_g:
        acfg = acfg.but(freeze_g=False)
    if a.lambda_init is not None:
        acfg = acfg.but(lambda_init=a.lambda_init)
    recal_steps = {int(v) for v in a.recalibrate_at.split(",") if v.strip()}
    mcfg = ModelConfig(vocab_size=V, max_len=a.seq_len, dim=a.dim, layers=a.layers, heads=a.heads, ffn=a.ffn,
                       dropout=a.dropout, attn_dropout=a.dropout, pad_id=pad_id)
    model = MLMEncoder(mcfg, acfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    # fixed evaluation batch for calibration and diagnostics (same for all variants)
    diag_ids = torch.tensor(val[:32])
    g = torch.Generator().manual_seed(EVAL_MASK_SEED)
    diag_in, _ = masker(diag_ids, g)
    calibration = [{"step": 0, "layers": model.calibrate(diag_in.to(dev))}] if acfg.gravitational else None
    grav_init = model.grav_state()

    opt = torch.optim.AdamW(model.param_groups(a.lr, a.grav_lr, a.weight_decay), betas=(0.9, 0.98), eps=1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(a.warmup, a.steps))
    use_scaler = a.amp and torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    accum = max(1, a.batch_size // a.micro_batch)
    rng = np.random.RandomState(a.seed)
    order, pos = rng.permutation(len(train)), 0
    mask_gen = torch.Generator().manual_seed(10_000 + a.seed)

    def evaluate(data, mass_override=None):
        model.eval()
        for m in model.attn_modules():
            m.mass_override = mass_override
        gen = torch.Generator().manual_seed(EVAL_MASK_SEED)
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(data), a.micro_batch):
                x, y = masker(torch.tensor(data[i:i + a.micro_batch]), gen)
                with autocast(a.amp):
                    loss, _ = model(x.to(dev), y.to(dev))
                k = int((y != -100).sum())
                tot += float(loss) * k
                n += k
        for m in model.attn_modules():
            m.mass_override = None
        model.train()
        return tot / max(1, n)

    def diagnostics(step):
        model.eval()
        model.set_record(True)
        with torch.no_grad(), autocast(False):
            model(diag_in.to(dev))
        append_jsonl(os.path.join(a.out, "diagnostics.jsonl"), {"step": step, "layers": model.diagnostics()})
        model.set_record(False)
        model.train()

    layer_params = [(list(b.attn.qkv.parameters()), list(b.attn.grav_parameters())) for b in model.blocks]

    def grad_norm(ps):
        gs = [p.grad.detach().float().norm() for p in ps if p.grad is not None]
        return float(torch.stack(gs).norm()) if gs else 0.0

    counters = {"nonfinite_steps": 0, "clipped_steps": 0, "loss_spikes": 0}
    recent = []
    curve = [{"step": 0, "val_loss": evaluate(val)}]
    best = curve[0]["val_loss"]
    if acfg.gravitational:
        diagnostics(0)
    print(f"{a.variant} seed {a.seed}: {n_params/1e6:.1f}M params, step 0 val loss {best:.4f}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model.train()
    consecutive_bad = 0
    for step in range(1, a.steps + 1):
        step_loss, bad = 0.0, False
        for _ in range(accum):
            if pos + a.micro_batch > len(order):
                order, pos = rng.permutation(len(train)), 0
            rows = torch.tensor(train[order[pos:pos + a.micro_batch]])
            pos += a.micro_batch
            x, y = masker(rows, mask_gen)
            with autocast(a.amp):
                loss, _ = model(x.to(dev), y.to(dev))
            loss = loss / accum
            if not torch.isfinite(loss):
                bad = True
                break
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        if bad:
            counters["nonfinite_steps"] += 1
            consecutive_bad += 1
            opt.zero_grad(set_to_none=True)
            if consecutive_bad >= 10:
                raise FloatingPointError(f"10 consecutive non-finite steps at step {step}")
            sched.step()
            continue
        consecutive_bad = 0
        scaler.unscale_(opt)
        per_layer = [(grad_norm(d), grad_norm(gp)) for d, gp in layer_params] if step % a.log_every == 0 else None
        total = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
        counters["clipped_steps"] += int(total > a.clip)
        scaler.step(opt)
        scaler.update()
        model.project_()
        opt.zero_grad(set_to_none=True)
        sched.step()

        spike = len(recent) >= 20 and step_loss > 3 * statistics.median(recent)
        counters["loss_spikes"] += int(spike)
        recent = (recent + [step_loss])[-100:]
        if step % 500 == 0:
            write_lock()                                       # heartbeat for the lock
        if per_layer is not None or spike:
            append_jsonl(os.path.join(a.out, "train_log.jsonl"), {
                "step": step, "loss": step_loss, "grad_norm": total, "clipped": total > a.clip, "spike": spike,
                "lr": sched.get_last_lr()[0],
                "grad_norm_dot_by_layer": [d for d, _ in per_layer] if per_layer else None,
                "grad_norm_grav_by_layer": [gg for _, gg in per_layer] if per_layer else None})
        if acfg.gravitational and step in recal_steps:
            calibration.append({"step": step, "layers": model.calibrate(diag_in.to(dev))})
        if acfg.gravitational and step % a.diag_every == 0:
            diagnostics(step)
        if step % a.eval_every == 0 or step == a.steps:
            vl = evaluate(val)
            curve.append({"step": step, "val_loss": vl, "elapsed_s": time.time() - t0})
            print(curve[-1], flush=True)
            if a.save_best and vl < best:
                best = vl
                torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    train_time = time.time() - t0

    final = {"val_loss": curve[-1]["val_loss"], "val_ppl": math.exp(curve[-1]["val_loss"]),
             "test_loss": evaluate(test)}
    final["test_ppl"] = math.exp(final["test_loss"])
    if acfg.gravitational and acfg.mass != "none":            # Paper 1's mass tests, on validation
        gen = torch.Generator().manual_seed(EVAL_MASK_SEED)

        def shuffle(m):
            idx = torch.argsort(torch.rand(m.shape, generator=gen).to(m.device), dim=-1)
            return torch.gather(m, -1, idx)
        final["val_loss_mass_shuffled"] = evaluate(val, shuffle)
        final["val_loss_mass_uniform"] = evaluate(val, lambda m: m.mean(-1, keepdim=True).expand_as(m))
    torch.save(model.state_dict(), os.path.join(a.out, "final.pt"))
    write_json(os.path.join(a.out, "results.json"), {
        "phase": "pretrain_from_scratch", "variant": a.variant, "seed": a.seed,
        "attn_config": acfg.to_dict(), "model_config": mcfg.to_dict(), "n_params": n_params,
        "args": vars(a), "env": environment(), "curve": curve, "final": final,
        "grav_init": grav_init, "grav_final": model.grav_state(), "calibration": calibration,
        "counters": counters, "train_time_s": train_time,
        "tokens_per_s": a.steps * a.batch_size * a.seq_len / max(train_time, 1e-9),
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "train_sequences": int(len(train))})
    os.remove(lock)
    print("done", a.out, final, flush=True)


if __name__ == "__main__":
    main()
