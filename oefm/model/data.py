"""Training data for the model: loader samples -> transformed tensors, masking, batching.

Per sequence (design document "Masking sampler"), one mode is drawn:
    whole view 40%   all observations of one sensor or product (PACE half the time when present)
    time step  30%   25-50% of the observations
    spatial    20%   4-6 rectangles per observation covering ~40% of its cells
    future     10%   every observation with dt > 0 (retrospective windows)
Whole-view masking draws the hidden view uniformly over the view groups present (MODIS Rrs,
VIIRS Rrs (all three platforms), OLCI Rrs, PACE Rrs, PACE rhos), not over observations.
At least one optical observation keeps visible cells. Each observation gets two cell masks:
ctx_cells (visible to the context encoder) and tgt_cells (to predict). Spectral dropout keeps a
random 20-100% of PACE bands in the context copy only.

Batches group observations by product (one band set per group) so the tokenizer runs once per
group. Values are transformed with the frozen training statistics (oefm/normstats.py).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Config
from ..inspect import QUALITY_FLAGS
from ..loader import TileSequences
from ..normstats import FAMILY, stats_dir
from ..spectral import GRID_NM, Normalizer, nominal_fwhm, srf_matrix

MODES = (("view", 0.4), ("time", 0.3), ("spatial", 0.2), ("future", 0.1))


def quality_channels(flags: np.ndarray, meanings: list[str], masks: list[int]) -> np.ndarray:
    lut = dict(zip(meanings, masks))
    return np.stack([((flags & np.uint32(lut.get(f, 0))) != 0) if lut.get(f) else np.zeros(flags.shape, bool)
                     for f in QUALITY_FLAGS]).astype(np.uint8)


MODIS_REF_WL = np.array([412.0, 443.0, 469.0, 488.0, 531.0, 547.0, 555.0, 645.0, 667.0, 678.0])


def reference_target(o: dict, raw: np.ndarray, src_wl: np.ndarray, norm: Normalizer) -> dict | None:
    """E2 training target (design document): a hidden PACE or OLCI observation SRF-convolved onto
    MODIS-Aqua's ocean bands, encoded alone at its own dt; a target-only pseudo-observation with the
    MODIS band signature, so the predictor learns the reference query used for E2."""
    from ..spectral import convolve

    flat = np.where(np.isfinite(raw), raw, np.nan).reshape(len(src_wl), -1).T              # [1024, C]
    h = convolve(src_wl, flat, MODIS_REF_WL, nominal_fwhm("MODIS", MODIS_REF_WL))            # [1024, 10]
    y, bm = norm.transform("rrs", MODIS_REF_WL, h.T.reshape(len(MODIS_REF_WL), 32, 32))
    cells = bm.any(0).ravel() & o["tgt_cells"]
    if not cells.any():
        return None
    return {"product": "modisa_oc", "family": "MODIS", "view": "rrs", "olci": False, "wl": MODIS_REF_WL,
            "fwhm": nominal_fwhm("MODIS", MODIS_REF_WL), "values": y.astype(np.float16), "band_mask": bm,
            "band_mask_ctx": bm, "quality": o["quality"], "cell_valid": bm.any(0).ravel(),
            "raw_mean": np.nanmean(np.where(bm, h.T.reshape(len(MODIS_REF_WL), 32, 32), np.nan), axis=0).ravel()
            if bm.any() else np.zeros(1024), "dt_days": o["dt_days"], "is_pace": False,
            "ctx_cells": np.zeros(1024, bool), "tgt_cells": cells, "synthetic": True}


def sample_mask(obs: list[dict], rng: np.random.Generator, modes=MODES) -> str:
    """Set o['ctx_cells'], o['tgt_cells'] ([1024] bool) on each observation; returns the mode."""
    n = len(obs)
    for o in obs:
        o["ctx_cells"] = o["cell_valid"].copy()
        o["tgt_cells"] = np.zeros(1024, bool)
    names, probs = zip(*modes)
    mode = rng.choice(names, p=np.asarray(probs) / sum(probs))
    groups = sorted({o["family"] for o in obs})
    if mode == "view" and len(groups) < 2:
        mode = "time"
    if mode == "future" and not any(o["dt_days"] > 0 for o in obs):
        mode = "time"
    if mode == "time" and n < 2:
        mode = "spatial"
    hide = np.zeros(n, bool)
    if mode == "view":
        pick = groups[rng.integers(len(groups))]
        hide = np.array([o["family"] == pick for o in obs])
    elif mode == "time":
        k = max(1, int(round(rng.uniform(0.25, 0.5) * n)))
        hide[rng.choice(n, min(k, n - 1), replace=False)] = True
    elif mode == "future":
        hide = np.array([o["dt_days"] > 0 for o in obs])
    if mode != "spatial" and hide.all():
        hide[rng.integers(n)] = False
    for o, h in zip(obs, hide):
        if h:
            o["tgt_cells"] = o["cell_valid"].copy()
            o["ctx_cells"] = np.zeros(1024, bool)
    if mode == "spatial":
        for o in obs:
            r = np.zeros((32, 32), bool)
            for _ in range(int(rng.integers(4, 7))):
                area = 0.4 * 1024 / 5
                h = int(np.clip(round(math.sqrt(area) * rng.uniform(0.6, 1.6)), 2, 24))
                w = int(np.clip(round(area / h), 2, 24))
                y, x = int(rng.integers(0, 33 - h)), int(rng.integers(0, 33 - w))
                r[y:y + h, x:x + w] = True
            rr = r.ravel()
            o["tgt_cells"] = o["cell_valid"] & rr
            o["ctx_cells"] = o["cell_valid"] & ~rr
    if not any(o["ctx_cells"].any() for o in obs):  # keep >= 1 optical observation visible
        o = obs[int(rng.integers(n))]
        o["ctx_cells"], o["tgt_cells"] = o["cell_valid"].copy(), np.zeros(1024, bool)
    return str(mode)


class TrainData(torch.utils.data.Dataset):
    """Wraps TileSequences: one item = one masked training sequence (numpy)."""

    def __init__(self, cfg: Config, tcfg: dict, split: str = "train", seed: int = 0) -> None:
        d = tcfg["data"]
        self.cfg, self.tcfg = cfg, tcfg
        self.seq = TileSequences(cfg, split=split, mode=d["mode"], length=d["length"], sensors=d.get("sensors"),
                                 selection="random" if split == "train" else "recent", include_300m=True,
                                 physics=False, seed=seed)
        self.norm = Normalizer.load(stats_dir(cfg) / "normstats.json")
        rp = stats_dir(cfg) / "pace_ridge.npz"
        self.ridge = dict(np.load(rp)) if (tcfg["arm"] == "C" and rp.exists()) else None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.weights = self._weights(d)

    def _weights(self, d: dict) -> np.ndarray:
        """Sampling weights over anchors: balanced by month, PACE era raised to pace_era_share."""
        import pandas as pd

        a = self.seq.obs.loc[self.seq.anchors]
        t = pd.to_datetime(a["t"], unit="s", utc=True)
        month = t.dt.month.to_numpy()
        w = 1.0 / np.bincount(month, minlength=13)[month]
        pace0 = pd.Timestamp(str(self.cfg.raw["training"]["partitions"]["pace_start"]), tz="UTC")
        era = (t >= pace0).to_numpy()
        share = float(d.get("pace_era_share", 0.3))
        if era.any() and (~era).any():
            w[era] *= share / w[era].sum()
            w[~era] *= (1 - share) / w[~era].sum()
        return w / w.sum()

    def __len__(self) -> int:
        return len(self.seq)

    def _obs(self, o: dict, anchor_t: float, rng) -> dict:
        view = "surface" if o["product"].endswith("sfrefl") else "rrs"
        wl = o["wavelength"].astype("f8")
        fam = FAMILY[o["product"]]
        olci = "reflectance_300m" in o
        raw = o["reflectance_300m"] if olci else o["reflectance"]
        valid_px = o["valid_300m"] if olci else o["valid"]
        y, bm = self.norm.transform(view, wl, raw)
        bm &= valid_px[None]
        cell_valid = (bm.reshape(len(wl), 32, 4, 32, 4).any(axis=(0, 2, 4)) if olci else bm.any(0)).ravel()
        q = quality_channels(o["flags"], o["flag_meanings"], o["flag_masks"])
        r12 = np.where(o["valid"][None], o["reflectance"], np.nan)
        with np.errstate(invalid="ignore"):
            raw_mean = np.nan_to_num(np.nanmean(np.where(np.isfinite(r12), r12, np.nan), axis=0)).ravel() \
                if np.isfinite(r12).any() else np.zeros(1024)
        out = {"product": o["product"], "family": "SURFACE" if view == "surface" else fam, "view": view, "olci": olci, "wl": wl, "fwhm": nominal_fwhm(fam, wl),
               "values": y.astype(np.float16), "band_mask": bm, "quality": q, "cell_valid": cell_valid,
               "raw_mean": raw_mean.astype(np.float32), "_raw12": r12,
               "dt_days": (o["time"] - anchor_t) / 86400.0, "is_pace": o["is_pace"] and view == "rrs"}
        if out["is_pace"] and self.ridge is not None:
            out.update(self._residual(o, wl, y, bm))
        return out

    def _residual(self, o, wl, y, bm) -> dict:
        r = self.ridge
        raw = np.where(o["valid"][None], o["reflectance"], np.nan).reshape(len(wl), -1).T      # [1024, P]
        if raw.shape[1] != r["conv"].shape[0]:
            return {}
        ok = np.isfinite(raw)
        mass = ok.astype("f4") @ r["conv"]
        full = r["conv"].sum(0)
        h = np.where(mass >= 0.8 * full, (np.where(ok, raw, 0) @ r["conv"]) / np.maximum(mass, 1e-12) * full, np.nan)
        ht, hv = self.norm.transform("rrs", r["heritage_wl"], h.T)
        recon = (ht.T - r["mu_h"]) @ r["W"] + r["mu_p"]                                          # [1024, P]
        p = y.reshape(len(wl), -1).T
        m = bm.reshape(len(wl), -1).T & hv.all(0)[:, None]
        return {"pace_resid": (p - recon).T.astype(np.float16), "pace_resid_mask": m.T}

    def __getitem__(self, k: int) -> dict[str, Any]:
        rng = np.random.default_rng((self.seed, k, int(self.rng.integers(1 << 31))))
        s = self.seq[int(k)]
        obs = [self._obs(o, s["anchor_time"], rng) for o in s["obs"]]
        obs = [o for o in obs if o["cell_valid"].any()]
        if not obs:
            return {"obs": [], "mode": "none"}
        mk = self.tcfg["masking"]
        mode = sample_mask(obs, rng, tuple(tuple(m) for m in mk["modes"]))
        # E2: in a share of sequences, a synthetic MODIS-band target from a hidden PACE or OLCI obs
        if rng.random() < float(mk.get("e2_reference_prob", 0.0)):
            src = [o for o in obs if o["family"] in ("PACE", "OLCI") and o["view"] == "rrs" and o["tgt_cells"].any()]
            if src:
                o = src[int(rng.integers(len(src)))]
                ref = reference_target(o, o["_raw12"], o["wl"], self.norm)
                if ref is not None:
                    obs.append(ref)
                    mode += "+e2ref"
        for o in obs:
            o.pop("_raw12", None)
        lo, hi = self.tcfg["masking"]["pace_band_keep"]
        for o in obs:
            o["band_mask_ctx"] = o["band_mask"]
            if o["is_pace"]:
                keep = rng.random(len(o["wl"])) < rng.uniform(lo, hi)
                keep[rng.integers(len(keep))] = True
                o["band_mask_ctx"] = o["band_mask"] & keep[:, None, None]
        return {"obs": obs, "mode": mode, "tile_id": s["tile_id"]}


def collate(items: list[dict]) -> dict:
    """Group observations by product across the batch; numpy -> torch (CPU)."""
    items = [it for it in items if it["obs"]]
    B = len(items)
    T = max(len(it["obs"]) for it in items) if items else 1
    groups: dict[str, dict] = {}
    for b, it in enumerate(items):
        for t, o in enumerate(it["obs"]):
            g = groups.setdefault(o["product"], {"product": o["product"], "family": o["family"], "view": o["view"], "olci": o["olci"],
                                                 "wl": o["wl"], "fwhm": o["fwhm"], "is_pace": o["is_pace"], "rows": []})
            g["rows"].append((b, t, o))
    out = []
    for g in groups.values():
        rows = g.pop("rows")
        os_ = [r[2] for r in rows]
        grid = GRID_NM  # one 340-2300 nm grid for every view; the view type conditions the generator
        gg = {**g, "view": 1 if g["view"] == "surface" else 0,
              "wl": torch.tensor(g["wl"], dtype=torch.float32),
              "srf": torch.tensor(srf_matrix(g["wl"], g["fwhm"], grid), dtype=torch.float32),
              "grid": torch.tensor(grid, dtype=torch.float32),
              "seq": torch.tensor([r[0] for r in rows]), "slot": torch.tensor([r[1] for r in rows]),
              "values": torch.from_numpy(np.stack([o["values"] for o in os_])),
              "band_mask": torch.from_numpy(np.stack([o["band_mask"] for o in os_])),
              "band_mask_ctx": torch.from_numpy(np.stack([o["band_mask_ctx"] for o in os_])),
              "quality": torch.from_numpy(np.stack([o["quality"] for o in os_])),
              "raw_mean": torch.from_numpy(np.stack([np.asarray(o["raw_mean"], np.float32) for o in os_])),
              "cell_valid": torch.from_numpy(np.stack([o["cell_valid"] for o in os_])),
              "ctx_cells": torch.from_numpy(np.stack([o["ctx_cells"] for o in os_])),
              "tgt_cells": torch.from_numpy(np.stack([o["tgt_cells"] for o in os_])),
              "dt": torch.tensor([o["dt_days"] for o in os_], dtype=torch.float32)}
        if g["is_pace"] and all("pace_resid" in o for o in os_):
            gg["pace_resid"] = torch.from_numpy(np.stack([o["pace_resid"] for o in os_]))
            gg["pace_resid_mask"] = torch.from_numpy(np.stack([o["pace_resid_mask"] for o in os_]))
        out.append(gg)
    return {"B": B, "T": T, "groups": out, "modes": [it["mode"] for it in items]}


def to_device(batch: dict, device) -> dict:
    gs = []
    for g in batch["groups"]:
        gs.append({k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in g.items()})
    return {**batch, "groups": gs, "device": device, "dtype": torch.float32}
