"""Two-process DDP (gloo, CPU) check: arms A/B/C with different view groups per rank, gradient
accumulation with no_sync. Regression test for job 116191 (B/C hung: SIGReg ran an autograd
all_reduce inside backward, interleaving with DDP's gradient all-reduces differently per rank)."""

import contextlib
import datetime
import os
import socket
import unittest

import numpy as np

try:
    import torch
    import torch.distributed as dist
except ImportError:
    torch = None


def _batch(rank, step):
    from oefm.model.data import collate, sample_mask
    from test_model import fake_obs

    rng = np.random.default_rng(100 * rank + step)
    items = []
    for _ in range(2):
        if rank == 0:
            obs = [fake_obs("viirsn_oc", [410, 443, 486, 551, 671], dt=float(t), rng=rng) for t in range(-1, 2)]
            obs.append(fake_obs("pace_oci_aop", [350, 400, 450, 500, 550, 600], dt=-1, rng=rng, pace=True))
        else:
            obs = [fake_obs("modisa_oc", [412, 443, 488, 547, 667], dt=float(t), rng=rng) for t in range(-1, 1)]
            obs.append(fake_obs("olcia_efr_oc", [400, 443, 560], olci=True, dt=0.5, rng=rng))
        sample_mask(obs, rng)
        items.append({"obs": obs, "mode": "x"})
    return {**collate(items), "device": torch.device("cpu"), "dtype": torch.float32}


def _worker(rank, world, arm, port, out_dir):
    from oefm.model.net import Model
    from oefm.model.sigreg import GroupSIGReg
    from test_model import TINY

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=60))
    torch.manual_seed(0)
    m = Model({**TINY, "arm": arm})
    ddp = torch.nn.parallel.DistributedDataParallel(m, find_unused_parameters=True)
    sig = GroupSIGReg(["MODIS", "VIIRS", "OLCI", "PACE", "SURFACE"], 64, min_tokens=16)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    for step in range(3):
        opt.zero_grad()
        for micro in range(2):
            with ddp.no_sync() if micro == 0 else contextlib.nullcontext():
                out = ddp(_batch(rank, 2 * step + micro), sigreg=sig, lam=0.05, mu=0.1, sigreg_tokens=512)
                (out["loss"] / 2).backward()
        opt.step()
    torch.save(torch.cat([p.detach().flatten() for p in m.parameters()]), os.path.join(out_dir, f"{arm}_{rank}.pt"))
    dist.destroy_process_group()


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipIf(torch is None or not torch.distributed.is_available(), "PyTorch distributed not available")
class TestDDP(unittest.TestCase):
    def test_arms_train_in_sync(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            for arm in ("A", "B", "C"):
                torch.multiprocessing.spawn(_worker, args=(2, arm, _port(), d), nprocs=2)
                w0, w1 = torch.load(f"{d}/{arm}_0.pt"), torch.load(f"{d}/{arm}_1.pt")
                self.assertTrue(torch.isfinite(w0).all(), arm)
                self.assertTrue(torch.equal(w0, w1), arm)

    def test_group_sigreg_matches_single_process(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            torch.multiprocessing.spawn(_sig_worker, args=(2, _port(), d), nprocs=2)
            data = _sig_data()
            groups = ["MODIS", "VIIRS", "PACE"]
            from oefm.model.sigreg import GroupSIGReg

            per = {k: torch.cat([x[k] for x in data if k in x]).clone().requires_grad_() for k in groups}
            tot, _ = GroupSIGReg(groups, 32, min_tokens=100)(per, 16, torch.device("cpu"))
            tot.backward()
            off = {k: 0 for k in groups}
            for r in range(2):
                t, g = torch.load(f"{d}/sig_{r}.pt")
                self.assertAlmostEqual(t, float(tot), places=5)
                ref = torch.cat([per[k].grad[off[k]:off[k] + data[r][k].shape[0]].flatten() for k in sorted(data[r])])
                for k in data[r]:
                    off[k] += data[r][k].shape[0]
                self.assertLess(float((g - ref).abs().max()), 1e-6 * max(1.0, float(ref.abs().max())) + 1e-7)


def _sig_data():
    g = torch.Generator().manual_seed(5)
    return [{"MODIS": torch.randn(300, 16, generator=g), "VIIRS": torch.randn(200, 16, generator=g),
             "PACE": torch.randn(40, 16, generator=g)},
            {"MODIS": torch.randn(100, 16, generator=g), "PACE": torch.randn(30, 16, generator=g)}]


def _sig_worker(rank, world, port, out_dir):
    from oefm.model.sigreg import GroupSIGReg

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=60))
    per = {k: v.clone().requires_grad_() for k, v in _sig_data()[rank].items()}
    tot, _ = GroupSIGReg(["MODIS", "VIIRS", "PACE"], 32, min_tokens=100)(per, 16, torch.device("cpu"))
    tot.backward()
    g = torch.cat([per[k].grad.flatten() for k in sorted(per)]) / world      # DDP averages over ranks
    torch.save((float(tot), g), os.path.join(out_dir, f"sig_{rank}.pt"))
    dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
