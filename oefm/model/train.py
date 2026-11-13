"""Stage 1 training: python -m oefm.model.train --arm B [--benchmark 300] (torchrun for multi-GPU).

Design document "Implementation specification (v1)": AdamW (0.9, 0.95), peak lr 5e-4, weight
decay 0.05, gradient clip 1.0, linear warmup 5% then cosine to 1e-6; bf16 autocast; effective batch
128 sequences (per-GPU batch x accumulation x GPUs); the same number of steps for every arm.
Checkpoints every `checkpoint_minutes` and before the 48 h limit; a rerun resumes from the last
checkpoint (cluster/train.sbatch resubmits itself). Logs one JSON line per `log_every` steps to
runs/<name>/log.jsonl: losses, learning rate, sequences/s, data-loader wait share, peak GPU memory,
effective rank of E1 embeddings. `--benchmark N` runs N steps and writes benchmark.json
(throughput, memory, loader wait) without changing the training run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from ..config import REPO_ROOT, load_config
from .data import TrainData, collate, to_device
from .net import Model, effective_rank
from .sigreg import GroupSIGReg, SIGReg


def lr_at(step: int, o: dict) -> float:
    warm = max(1, int(o["warmup_frac"] * o["steps"]))
    if step < warm:
        return o["lr"] * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, o["steps"] - warm))
    return o["min_lr"] + 0.5 * (o["lr"] - o["min_lr"]) * (1 + math.cos(math.pi * p))


def infinite(loader):
    while True:
        yield from loader


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "train_stage1.yaml"))
    ap.add_argument("--arm", choices=["A", "B", "C"])
    ap.add_argument("--name", help="run name (default stage1_<arm>)")
    ap.add_argument("--benchmark", type=int, default=0, help="run N steps, report speed and memory, do not save")
    ap.add_argument("--override", nargs="*", default=[], help="dotted key=value, e.g. optim.per_gpu_batch=8")
    args = ap.parse_args(argv)

    tcfg = yaml.safe_load(Path(args.config).read_text())
    if args.arm:
        tcfg["arm"] = args.arm
    for kv in args.override:
        k, v = kv.split("=", 1)
        node = tcfg
        *path, last = k.split(".")
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(v)
    arm = tcfg["arm"]

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0
    torch.manual_seed(tcfg["run"]["seed"] + rank)

    cfg = load_config()
    run_dir = cfg.data_root / "runs" / (args.name or f"stage1_{arm}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if main_proc:
        (run_dir / "train_config.json").write_text(json.dumps(tcfg, indent=1))

    data = TrainData(cfg, tcfg, "train", seed=tcfg["run"]["seed"] + 1000 * rank)
    o = tcfg["optim"]
    per = int(o["per_gpu_batch"])
    accum = max(1, int(o["effective_batch"]) // (per * world))
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(data.weights, dtype=torch.double),
                                                     num_samples=10_000_000, replacement=True,
                                                     generator=torch.Generator().manual_seed(tcfg["run"]["seed"] + rank))
    nw = int(tcfg["data"]["num_workers"])
    loader = torch.utils.data.DataLoader(data, batch_size=per, sampler=sampler, collate_fn=collate, num_workers=nw,
                                         pin_memory=device.type == "cuda", persistent_workers=nw > 0,
                                         prefetch_factor=4 if nw > 0 else None)
    it = infinite(loader)

    model = Model(tcfg).to(device)
    L = tcfg["loss"]
    if L.get("sigreg_mode", "group") == "group":
        sig = GroupSIGReg(list(L["modality_weights"]), L["sigreg_slices"], L["sigreg_t_max"], L["sigreg_points"],
                          int(L.get("sigreg_min_group_tokens", 256))).to(device)
    else:
        sig = SIGReg(L["sigreg_slices"], L["sigreg_t_max"], L["sigreg_points"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    core = model.module if ddp else model
    decay = [p for n, p in core.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in core.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": o["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=o["lr"], betas=tuple(o["betas"]))
    step = 0
    ck = run_dir / "checkpoint.pt"
    if ck.exists() and not args.benchmark:
        st = torch.load(ck, map_location=device, weights_only=False)
        core.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sig.load_state_dict(st["sigreg"])
        step = st["step"]
        if main_proc:
            print(f"resumed from step {step}", flush=True)

    def save(tag: str = "") -> None:
        if not main_proc or args.benchmark:
            return
        st = {"model": core.state_dict(), "opt": opt.state_dict(), "sigreg": sig.state_dict(), "step": step,
              "config": tcfg, "saved": datetime.now(timezone.utc).isoformat()}
        tmp = ck.with_suffix(".tmp")
        torch.save(st, tmp)
        tmp.replace(ck)
        if tag:
            torch.save({"model": core.state_dict(), "step": step, "config": tcfg}, run_dir / f"weights_{tag}.pt")

    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=tcfg["run"]["precision"] == "bf16" and device.type == "cuda")
    lam, mu = tcfg["loss"]["lambda"], tcfg["loss"]["mu"]
    sig_tokens = max(1, int(tcfg["loss"]["sigreg_tokens"]) // world)
    t_start = time.time()
    last_ck = time.time()
    total_steps = args.benchmark or int(o["steps"])
    stop_s = float(tcfg["run"]["stop_after_hours"]) * 3600
    logf = open(run_dir / ("benchmark_log.jsonl" if args.benchmark else "log.jsonl"), "a") if main_proc else None
    if main_proc:
        print(f"arm {arm} | params {n_params / 1e6:.1f} M | world {world} | per-GPU batch {per} | accumulation {accum} "
              f"| effective batch {per * accum * world} | anchors {len(data):,} | device {device}", flush=True)
    model.train()
    wait = comp = 0.0
    seqs = 0
    bench_start = None
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, o)
        opt.zero_grad(set_to_none=True)
        agg: dict[str, float] = {}
        for micro in range(accum):
            t0 = time.time()
            batch = to_device(next(it), device)
            t1 = time.time()
            ctx = model.no_sync() if (ddp and micro < accum - 1) else contextlib.nullcontext()
            with ctx, torch.autocast(**amp):
                out = model(batch, sigreg=sig, lam=lam, mu=mu, sigreg_tokens=sig_tokens,
                            balance=tcfg["loss"]["balance"], weights=tcfg["loss"]["modality_weights"])
                loss = out["loss"] / accum
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t2 = time.time()
            wait += t1 - t0
            comp += t2 - t1
            seqs += batch["B"]
            for k in [k for k in out if k in ("loss", "pred", "sigreg", "recon", "pace_residual")
                      or k.startswith(("pred_", "recon_", "sigreg_"))]:
                if torch.is_tensor(out[k]) or isinstance(out[k], float):
                    agg[k] = agg.get(k, 0.0) + float(out[k]) / accum
        torch.nn.utils.clip_grad_norm_(core.parameters(), o["grad_clip"])
        opt.step()
        step += 1
        if args.benchmark and step == 10:
            bench_start = (time.time(), seqs, wait, comp)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        if main_proc and (step % tcfg["run"]["log_every"] == 0 or step == total_steps):
            rec = {"step": step, "lr": opt.param_groups[0]["lr"], **agg,
                   "seq_per_s": seqs * world / max(1e-9, wait + comp), "loader_wait_share": wait / max(1e-9, wait + comp),
                   "max_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
                   "hours": (time.time() - t_start) / 3600}
            if step % tcfg["run"]["rank_every"] == 0 or step == total_steps:
                with torch.no_grad():
                    e, m = core.embed_e1(batch, out["z"].detach(), out["gmask"])
                    sample = e[m]
                    if sample.shape[0] > 4096:
                        sample = sample[torch.randperm(sample.shape[0], device=sample.device)[:4096]]
                    rec["effective_rank_e1"] = effective_rank(sample) if sample.shape[0] > 8 else None
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if not args.benchmark and time.time() - last_ck > 60 * tcfg["run"]["checkpoint_minutes"]:
            save()
            last_ck = time.time()
        if not args.benchmark and time.time() - t_start > stop_s:
            save()
            if main_proc:
                print(f"time limit: saved at step {step}; resubmit to continue", flush=True)
            break
    if args.benchmark and main_proc and bench_start:
        t0, s0, w0, c0 = bench_start
        el = time.time() - t0
        res = {"arm": arm, "steps_timed": step - 10, "seq_per_s_global": (seqs - s0) * world / el,
               "loader_wait_share": (wait - w0) / max(1e-9, (wait - w0) + (comp - c0)),
               "peak_mem_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
               "per_gpu_batch": per, "world": world, "length": tcfg["data"]["length"], "params_m": n_params / 1e6,
               "projected_hours_for_run": (o["steps"] * o["effective_batch"]) / max(1e-9, (seqs - s0) * world / el) / 3600}
        (run_dir / f"benchmark_{arm}_L{tcfg['data']['length']}_b{per}.json").write_text(json.dumps(res, indent=1))
        print("BENCHMARK " + json.dumps(res), flush=True)
    elif step >= total_steps:
        save("final")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
