"""Frozen training statistics for the model (step 2.8a, numpy only): run once after the tiles.

1. normstats.json: the reflectance transform (oefm/spectral.py, design document "Normalization").
   From a sample of TRAINING-split observations (loader split 'train', all Rrs products):
   - s(lambda): the noise scale of the asinh transform. Per band, the robust spread of differences
     between neighbouring 1.2 km cells (1.4826 * MAD / sqrt 2): sensor noise plus unresolved
     small-scale variability, i.e. "near the typical noise level". log s is fitted as a cubic in
     wavelength pooled over all sensors.
   - m(lambda), sd(lambda): mean and standard deviation of asinh(x / s), quartic fits in
     wavelength pooled over all sensors.
   The surface-reflectance view (SFREFL rhos) gets its own entry ('surface'), never mixed with Rrs.
2. pace_ridge.npz: the heritage-band reconstruction for the PACE residual guard (arm C). Ridge
   regression from SRF-convolved heritage bands (MODIS-Aqua, VIIRS SNPP/NOAA-20/NOAA-21, OLCI-A,
   all in transformed units) to the full transformed PACE spectrum, fitted on >= 100k training PACE
   cells; the penalty is chosen on a 10% hold-out of those cells.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Config
from .spectral import Normalizer, convolution_matrix, nominal_fwhm
from .tiling import tiles_root

FAMILY = {"modisa_oc": "MODIS", "viirsn_oc": "VIIRS", "viirsj1_oc": "VIIRS", "viirsj2_oc": "VIIRS",
          "olcia_efr_oc": "OLCI", "olcib_efr_oc": "OLCI", "pace_oci_aop": "PACE", "pace_oci_sfrefl": "PACE"}
HERITAGE = ["modisa_oc", "viirsn_oc", "viirsj1_oc", "viirsj2_oc", "olcia_efr_oc"]


def stats_dir(cfg: Config) -> Path:
    return tiles_root(cfg) / "train_stats"


def _fit(wl, y, deg, log=False):
    wl, y = np.asarray(wl, "f8"), np.asarray(y, "f8")
    ok = np.isfinite(y) & (y > 0 if log else True)
    c, sc = float(wl[ok].mean()), float(max(wl[ok].std(), 1.0))
    coef = np.polyfit((wl[ok] - c) / sc, np.log(y[ok]) if log else y[ok], deg)
    return {"coef": coef.tolist(), "center": c, "scale": sc, "log": log,
            "range_nm": [float(wl[ok].min()), float(wl[ok].max())]}


def _sample(ds, n_obs: int, rng) -> list[dict]:
    if len(ds.obs) == 0:
        return []
    idx = rng.choice(len(ds.obs), min(n_obs, len(ds.obs)), replace=False)
    out = []
    for i in idx:
        o = ds._read_obs(int(i), float(ds._times[i]))
        out.append(o)
    return out


def build(cfg: Config, n_obs: int = 3000, n_pace_obs: int = 600, log=print) -> Path:
    import pandas as pd

    from .loader import TileSequences

    rng = np.random.default_rng(20261112)
    out_dir = stats_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"created": datetime.now(timezone.utc).isoformat(), "split": "train"}
    per_band_all = {}
    for view, products in (("rrs", [p for p in FAMILY if p != "pace_oci_sfrefl"]), ("surface", ["pace_oci_sfrefl"])):
        ds = TileSequences(cfg, split="train", sensors=products, physics=False)
        if len(ds.obs) == 0:
            log(f"normstats: no training observations for {view}; skipped")
            continue
        # balance products: equal number of observations per product present
        present = ds.obs["product"].unique()
        per_prod = max(1, n_obs // len(present))
        rows = []
        for p in present:
            sub = np.where(ds.obs["product"].to_numpy() == p)[0]
            take = rng.choice(sub, min(per_prod, sub.size), replace=False)
            for i in take:
                rows.append(ds._read_obs(int(i), float(ds._times[i])))
        # per band (wavelength rounded to 0.1 nm) noise scale from neighbour differences
        diffs: dict[float, list] = {}
        vals: dict[float, list] = {}
        for o in rows:
            r = np.where(o["valid"][None], o["reflectance"], np.nan)
            for k, w in enumerate(o["wavelength"]):
                key = round(float(w), 1)
                a = r[k]
                d = np.r_[(a[:, 1:] - a[:, :-1]).ravel(), (a[1:, :] - a[:-1, :]).ravel()]
                d = d[np.isfinite(d)]
                v = a[np.isfinite(a)]
                if d.size:
                    diffs.setdefault(key, []).append(d)
                if v.size:
                    vals.setdefault(key, []).append(v)
        bands = sorted(set(diffs) & set(vals))
        s_b = []
        for b in bands:
            d = np.concatenate(diffs[b])
            s_b.append(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
        s_b = np.clip(np.asarray(s_b), 2e-5, 5e-3)
        s_fit = _fit(bands, s_b, 3, log=True)
        tmp = Normalizer({view: {"s": s_fit, "mean": {"coef": [0.0], "center": 0, "scale": 1},
                                 "std": {"coef": [1.0], "center": 0, "scale": 1}}})
        s_curve = tmp.params(view, bands)[0]
        m_b, sd_b, n_b = [], [], []
        for b, sc in zip(bands, s_curve):
            v = np.concatenate(vals[b])
            v = v[v >= -5 * sc]
            t = np.arcsinh(v / sc)
            m_b.append(float(np.mean(t)))
            sd_b.append(float(np.std(t)))
            n_b.append(int(v.size))
        stats[view] = {"s": s_fit, "mean": _fit(bands, m_b, 4), "std": _fit(bands, sd_b, 4),
                       "per_band": {"wavelength": bands, "s": s_b.tolist(), "mean": m_b, "std": sd_b, "n": n_b},
                       "observations": len(rows), "products": list(map(str, present))}
        per_band_all[view] = pd.DataFrame({"wavelength": bands, "s": s_b, "mean": m_b, "std": sd_b, "n": n_b})
        log(f"normstats {view}: {len(rows)} training observations, {len(bands)} distinct bands, "
            f"s(443) ~ {tmp.params(view, [443.0])[0][0]:.2e}")
    path = out_dir / "normstats.json"
    path.write_text(json.dumps(stats, indent=1))
    for view, df in per_band_all.items():
        df.to_csv(out_dir / f"normstats_per_band_{view}.csv", index=False)
    log(f"normstats -> {path}")
    ridge(cfg, Normalizer(stats), n_pace_obs, rng, log)
    return path


def heritage_bands(cfg: Config, ds) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Centre wavelengths and FWHM of the heritage bands, read from one stored observation each."""
    wl, fw, names = [], [], []
    for p in HERITAGE:
        sub = np.where(ds.obs["product"].to_numpy() == p)[0]
        if sub.size == 0:
            continue
        o = ds._read_obs(int(sub[0]), float(ds._times[sub[0]]))
        w = [float(x) for x in o["wavelength"]]
        wl += w
        fw += list(nominal_fwhm(FAMILY[p], w))
        names += [f"{p}:{x:g}" for x in w]
    return np.asarray(wl), np.asarray(fw), names


def ridge(cfg: Config, norm: Normalizer, n_pace_obs: int, rng, log=print) -> Path | None:
    from .loader import TileSequences

    ds_all = TileSequences(cfg, split="train", physics=False)
    pace = np.where(ds_all.obs["product"].to_numpy() == "pace_oci_aop")[0]
    if pace.size == 0 or "rrs" not in norm.stats:
        log("pace ridge: no training PACE observations yet; skipped")
        return None
    h_wl, h_fw, names = heritage_bands(cfg, ds_all)
    take = rng.choice(pace, min(n_pace_obs, pace.size), replace=False)
    P, H = [], []
    pwl = None
    for i in take:
        o = ds_all._read_obs(int(i), float(ds_all._times[i]))
        pwl = o["wavelength"].astype("f8")
        x = np.where(o["valid"][None], o["reflectance"], np.nan).reshape(len(pwl), -1).T  # [cells, P]
        x = x[np.isfinite(x).all(axis=1)]
        if x.size == 0:
            continue
        P.append(x)
    if not P:
        log("pace ridge: no complete PACE spectra; skipped")
        return None
    X = np.concatenate(P)
    if X.shape[0] > 400_000:
        X = X[rng.choice(X.shape[0], 400_000, replace=False)]
    M = convolution_matrix(pwl, h_wl, h_fw)                       # [P, Ch]
    Hraw = X @ M
    Ht, hv = norm.transform("rrs", h_wl, Hraw.T)                  # [Ch, n]
    Pt, pv = norm.transform("rrs", pwl, X.T)                      # [P, n]
    ok = hv.all(axis=0) & pv.all(axis=0)
    Ht, Pt = Ht[:, ok].T.astype("f8"), Pt[:, ok].T.astype("f8")
    n = Ht.shape[0]
    hold = rng.random(n) < 0.1
    mu_h, mu_p = Ht[~hold].mean(0), Pt[~hold].mean(0)
    A, B = Ht[~hold] - mu_h, Pt[~hold] - mu_p
    best = None
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        W = np.linalg.solve(A.T @ A + alpha * n * np.eye(A.shape[1]), A.T @ B)
        err = float(np.mean(((Ht[hold] - mu_h) @ W + mu_p - Pt[hold]) ** 2))
        if best is None or err < best[0]:
            best = (err, alpha, W)
    err, alpha, W = best
    resid_var = float(np.mean((Pt[hold] - ((Ht[hold] - mu_h) @ W + mu_p)) ** 2))
    tot_var = float(np.mean((Pt[hold] - mu_p) ** 2))
    dest = stats_dir(cfg) / "pace_ridge.npz"
    np.savez(dest, W=W.astype("f4"), mu_h=mu_h.astype("f4"), mu_p=mu_p.astype("f4"), heritage_wl=h_wl,
             heritage_fwhm=h_fw, heritage_names=np.array(names), pace_wl=pwl, alpha=alpha,
             n_spectra=n, holdout_mse=err, holdout_explained=1 - resid_var / tot_var,
             conv=M.astype("f4"))
    log(f"pace ridge: {n:,} training PACE spectra, {len(h_wl)} heritage bands, alpha {alpha:g}, "
        f"hold-out variance explained {1 - resid_var / tot_var:.3f} -> {dest}")
    if n < 100_000:
        log(f"  NOTE: fewer than 100k spectra ({n:,}); the design document asks for >= 100k (use more observations)")
    return dest
