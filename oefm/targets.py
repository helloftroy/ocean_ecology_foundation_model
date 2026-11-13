"""Auxiliary training targets (step 2.6) and evaluation-set matching (step 2.7).

PISCES nutrients -> tiles/<version>/targets/pisces_daily.zarr
    values      (day, tile, 3) float32  PO4, NO3, Si (mmol m-3) at the surface level, mean over
                the tile's 1.2 km cell centres (bilinear from the 0.25 deg model grid)
    coverage    (day, tile)    float32  share of the tile's cells with a model value (land = none)
    normalized  (day, tile, 3) float32  (log(x + eps) - mean) / std, mean/std from the training
                partition only (training tiles, training period)
    source      (day,)         int8     0 = multi-year reanalysis, 1 = analysis/forecast, -1 = none
These are TARGETS ONLY. The loader returns them under sample['targets'] when asked; they are never
part of sample['obs'] or sample['physics']. At 0.25 deg a tile holds 1-4 model cells, so the
target is a tile-level value, not a per-pixel field.

Evaluation sets: `match_eval` joins the targets built on the laptop (scripts/make_eval_targets.py,
copied to processed/regional/eval/) with the usable-imagery counts of step 2.4b.
"""

from __future__ import annotations

import os
import warnings
from datetime import date, datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, load_config
from .grid import TileGrid
from .physics import bilinear
from .provenance import ProcessingManifest, code_version, regional_root
from .tiling import tiles_root

VARS = ["po4", "no3", "si"]
EPS = {"po4": 0.01, "no3": 0.01, "si": 0.1}  # mmol m-3, for the log transform


def targets_path(cfg: Config) -> Path:
    return tiles_root(cfg) / "targets" / "pisces_daily.zarr"


def _cell_points(cfg: Config, grid: TileGrid) -> tuple[np.ndarray, np.ndarray]:
    lon = np.zeros((grid.n_tiles, grid.tc * grid.tc))
    lat = np.zeros_like(lon)
    for t, (ix, iy) in enumerate(zip(grid.ix, grid.iy)):
        lo, la = grid.cell_centres(int(ix), int(iy))
        lon[t], lat[t] = lo.ravel(), la.ravel()
    return lon, lat


def pisces_file(args: tuple[str, str]) -> dict[str, Any]:
    cfg_path, path = args
    import xarray as xr

    cfg = load_config(cfg_path)
    grid = TileGrid(cfg)
    qlon, qlat = _cell_points(cfg, grid)
    out: dict[str, Any] = {"path": path, "days": {}}
    src = 1 if "anfc" in Path(path).name else 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = xr.open_dataset(path)
        with ds:
            la = next(n for n in ("latitude", "lat") if n in ds.variables)
            lo = next(n for n in ("longitude", "lon") if n in ds.variables)
            lat, lon = np.asarray(ds[la].values, "f8"), np.asarray(ds[lo].values, "f8")
            t = np.asarray(ds["time"].values).astype("datetime64[D]")
            for i, d in enumerate(t):
                vals, cov = [], None
                for v in VARS:
                    da = ds[v]
                    if "depth" in da.dims:
                        da = da.isel(depth=0)
                    f = np.asarray(da.isel(time=i).transpose(la, lo).values, "f8")
                    s = bilinear(f, lat, lon, qlat.ravel(), qlon.ravel()).reshape(qlat.shape)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        vals.append(np.nanmean(s, axis=1))
                    cov = np.isfinite(s).mean(axis=1)
                out["days"][str(d)] = (np.stack(vals, axis=1).astype("f4"), cov.astype("f4"), src)
        out["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        out["status"], out["error"] = "failed", f"{type(exc).__name__}: {exc}"
    return out


def build_pisces(cfg: Config, workers: int = 0, log=print) -> Path | None:
    import shutil

    import zarr

    from .loader import geographic_labels, temporal_labels
    from .manifest import Manifest

    man = Manifest(cfg.manifest_path)
    rows = [dict(r) for r in man.rows("status='verified' AND product='pisces_nutrients'")]
    man.close()
    if not rows:
        log("No verified pisces_nutrients files: run the inventory and download for it first (step 2.6).")
        return None
    grid = TileGrid(cfg)
    phys = cfg.physical_windows()
    day0, day1 = min(w.start for w in phys), max(w.end for w in phys)
    nd = (day1 - day0).days + 1
    values = np.full((nd, grid.n_tiles, len(VARS)), np.nan, "f4")
    coverage = np.zeros((nd, grid.n_tiles), "f4")
    source = np.full(nd, -1, "i1")
    jobs = [(str(cfg.path), str(cfg.data_root / r["local_path"])) for r in sorted(rows, key=lambda r: r["time_start"] or "")]
    n = max(1, workers or (os.cpu_count() or 2) // 2)
    fails = 0
    with get_context("spawn").Pool(n) as pool:
        for res in pool.imap_unordered(pisces_file, jobs, chunksize=2):
            if res["status"] != "done":
                fails += 1
                log(f"  FAILED {res['path']}: {res['error']}")
                continue
            for d, (v, c, s) in res["days"].items():
                i = (date.fromisoformat(d) - day0).days
                if 0 <= i < nd and (source[i] == -1 or s < source[i]):  # reanalysis wins over forecast
                    values[i], coverage[i], source[i] = v, c, s
    # normalisation from the training partition only
    t_unix = np.array([datetime(day0.year, day0.month, day0.day, tzinfo=timezone.utc).timestamp() + 86400.0 * k
                       for k in range(nd)])
    tr_days = temporal_labels(cfg, t_unix) == "train"
    geo = geographic_labels(cfg, grid)
    tr_tiles = np.array([geo[t] == "train" for t in grid.ids])
    logv = np.log(np.maximum(values, 0) + np.array([EPS[v] for v in VARS], "f4"))
    sub = logv[tr_days][:, tr_tiles]
    mean = np.nanmean(sub.reshape(-1, len(VARS)), axis=0)
    std = np.nanstd(sub.reshape(-1, len(VARS)), axis=0)
    norm = ((logv - mean) / std).astype("f4")
    dest = targets_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    root = zarr.open_group(str(tmp), mode="w")
    root.create_dataset("values", data=values, chunks=(366, grid.n_tiles, len(VARS)))
    root.create_dataset("normalized", data=norm, chunks=(366, grid.n_tiles, len(VARS)))
    root.create_dataset("coverage", data=coverage, chunks=(366, grid.n_tiles))
    root.create_dataset("source", data=source, chunks=(nd,))
    root.attrs.update({"day0": day0.isoformat(), "variables": VARS, "units": "mmol m-3", "eps": EPS,
                       "norm": {"transform": "log(x + eps)", "mean": mean.tolist(), "std": std.tolist(),
                                "fitted_on": "training tiles x training period (config training.partitions)"},
                       "source_codes": {"0": "cmems_mod_glo_bgc_my_0.25deg_P1D-m", "1": "cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m"},
                       "tile_ids": grid.ids, "role": "auxiliary training target only; never an encoder input",
                       "code_version": code_version(cfg), "created": datetime.now(timezone.utc).isoformat()})
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    have = source >= 0
    log(f"PISCES targets: {int(have.sum())} of {nd} days ({int((source == 0).sum())} reanalysis, "
        f"{int((source == 1).sum())} forecast); median tile coverage {np.median(coverage[have]):.2f}; -> {dest}")
    pm = ProcessingManifest(cfg)
    pm.record(dest, "targets", "pisces_nutrients", "done" if not fails else "failed",
              [{"file": Path(j[1]).name} for j in jobs[:50]], code_version(cfg),
              {"variables": VARS, "eps": EPS, "norm_mean": mean.tolist(), "norm_std": std.tolist()},
              ["surface level", "bilinear to 1.2 km cell centres, mean per tile", "log + z-score from training partition"],
              n_records=int(have.sum()), error=f"{fails} files failed" if fails else None)
    pm.close()
    return dest


# ----------------------------------------------------------------------------- evaluation sets
def eval_dir(cfg: Config) -> Path:
    return regional_root(cfg) / "eval"


def match_eval(cfg: Config, log=print) -> Path | None:
    """Join laptop-built targets with the usable-imagery counts: eval/matched/<dataset>.csv.gz and
    eval/matched/summary.csv (samples per dataset x label split with usable optical / PACE data)."""
    import pandas as pd

    d = eval_dir(cfg)
    cov_p = tiles_root(cfg) / "index" / "eval_coverage.csv"
    files = sorted(d.glob("*.csv.gz"))
    files = [f for f in files if "_components" not in f.name]
    if not files:
        log(f"No evaluation targets in {d}: copy them from the laptop (step 2.7).")
        return None
    if not cov_p.exists():
        log("No eval_coverage.csv yet: run `python -m oefm evalcover` after the tiles.")
        return None
    cov = pd.read_csv(cov_p, dtype={"sample_id": str})
    out_dir = d / "matched"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        t = pd.read_csv(f, dtype={"sample_id": str})
        ds = t["dataset"].iloc[0]
        m = t.merge(cov[cov["dataset"] == ds].drop(columns=["dataset", "time_utc"]), on="sample_id", how="left")
        fam = [c for c in m.columns if c.startswith("causal_")]
        m["usable_any"] = m[fam].fillna(0).sum(axis=1) > 0
        m.to_csv(out_dir / f.name, index=False)
        for split, g in m.groupby("label_split"):
            rows.append({"dataset": ds, "label_split": split, "samples": len(g),
                         "with_tile": int(g["tile_id"].notna().sum()),
                         "usable_optical_any": int(g["usable_any"].sum()),
                         "usable_PACE_ge1": int((g.get("causal_PACE", 0) >= 1).sum()),
                         "usable_PACE_ge2": int((g.get("causal_PACE", 0) >= 2).sum()),
                         "years": ",".join(sorted({str(x)[:4] for x in g["time_utc"]}))})
    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "summary.csv", index=False)
    log(summ.to_string(index=False))
    log(f"Matched evaluation sets -> {out_dir}")
    return out_dir
