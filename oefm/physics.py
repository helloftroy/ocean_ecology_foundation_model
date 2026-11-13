"""Daily physical fields on a coarse grid around every tile (step 2.5a).

Built once from the regional crops (processed/regional/physical/) into
tiles/<version>/physics/physics_daily.zarr:

    values   (day, tile, variable, K, K)  float32, physical units after CF decoding, NaN = missing
    static   (tile, static_variable, K, K) float32 (GEBCO depth, positive down)
    lon, lat (tile, K, K)                  coarse-cell centres

Attributes: day0 (first day, ISO), variables (name, product, source variable, units), grid spec.
The coarse grid of a tile is centred on the tile and spans `context_tiles` tiles in each
direction (config processing.physics), so it carries context beyond the 38.4 km optical tile.
Each coarse cell is the mean of subsamples x subsamples bilinear samples of the source field:
for fine sources (MUR, 0.01 deg) that is an area average, for coarse ones (CCMP, 0.25 deg) a smooth
interpolation. Sub-daily sources are averaged per UTC day first.
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from datetime import date, datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, load_config
from .grid import TileGrid
from .provenance import ProcessingManifest, code_version, regional_root
from .tiling import tiles_root


def physics_path(cfg: Config) -> Path:
    return tiles_root(cfg) / "physics" / "physics_daily.zarr"


def coarse_targets(cfg: Config, grid: TileGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample lon/lat (tile, K, K, S*S) and cell-centre lon/lat (tile, K, K)."""
    p = cfg.raw["processing"]["physics"]
    k, s, ctx = int(p["coarse_cells"]), int(p["subsamples"]), float(p["context_tiles"])
    span = ctx * grid.tile_m
    cell = span / k
    off_c = (np.arange(k) + 0.5) * cell - span / 2
    off_s = (np.arange(s) + 0.5) * cell / s - cell / 2
    sub_lon = np.zeros((grid.n_tiles, k, k, s * s))
    sub_lat = np.zeros_like(sub_lon)
    c_lon = np.zeros((grid.n_tiles, k, k))
    c_lat = np.zeros_like(c_lon)
    for t, (ix, iy) in enumerate(zip(grid.ix, grid.iy)):
        cx, cy = (ix + 0.5) * grid.tile_m, (iy + 0.5) * grid.tile_m
        yy, xx = np.meshgrid(cy + off_c, cx + off_c, indexing="ij")
        c_lon[t], c_lat[t] = grid.proj.inverse(xx, yy)
        sy, sx = np.meshgrid(off_s, off_s, indexing="ij")
        X = xx[:, :, None] + sx.ravel()[None, None, :]
        Y = yy[:, :, None] + sy.ravel()[None, None, :]
        sub_lon[t], sub_lat[t] = grid.proj.inverse(X, Y)
    return sub_lon, sub_lat, c_lon, c_lat


def bilinear(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, qlat: np.ndarray, qlon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of field[lat, lon] (1-D, monotonic coordinates) at query points;
    NaN outside the grid or next to missing values."""
    if lat[0] > lat[-1]:
        lat, field = lat[::-1], field[::-1]
    if lon[0] > lon[-1]:
        lon, field = lon[::-1], field[:, ::-1]
    fi = np.interp(qlat, lat, np.arange(lat.size), left=np.nan, right=np.nan)
    fj = np.interp(qlon, lon, np.arange(lon.size), left=np.nan, right=np.nan)
    ok = np.isfinite(fi) & np.isfinite(fj)
    out = np.full(qlat.shape, np.nan, dtype="f8")
    i0 = np.clip(np.floor(fi[ok]).astype(int), 0, lat.size - 2)
    j0 = np.clip(np.floor(fj[ok]).astype(int), 0, lon.size - 2)
    a, b = fi[ok] - i0, fj[ok] - j0
    f = field
    out[ok] = ((1 - a) * (1 - b) * f[i0, j0] + (1 - a) * b * f[i0, j0 + 1]
               + a * (1 - b) * f[i0 + 1, j0] + a * b * f[i0 + 1, j0 + 1])
    return out


def _coords(ds):
    la = next(n for n in ("lat", "latitude") if n in ds.variables)
    lo = next(n for n in ("lon", "longitude") if n in ds.variables)
    return la, lo


def _surface2d(da, la: str, lo: str):
    """Drop every dimension except time/lat/lon by taking its first index (surface level)."""
    for d in list(da.dims):
        if d not in (la, lo, "time"):
            da = da.isel({d: 0})
    order = [d for d in ("time", la, lo) if d in da.dims]
    return da.transpose(*order)


def physics_file(args: tuple[str, str, str, list[tuple[int, str, list[str], bool]], str]) -> dict[str, Any]:
    """Worker: one cropped file -> {day ISO: {variable index: (tile, K, K)}}."""
    cfg_path, product, path, wanted, day0 = args
    import xarray as xr

    cfg = load_config(cfg_path)
    grid = TileGrid(cfg)
    sub_lon, sub_lat, _, _ = coarse_targets(cfg, grid)
    out: dict[str, Any] = {"product": product, "path": path, "days": {}, "found": {}, "units": {}}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = xr.open_dataset(path)
        with ds:
            la, lo = _coords(ds)
            lat, lon = np.asarray(ds[la].values, "f8"), np.asarray(ds[lo].values, "f8")
            qlon = sub_lon % 360 if np.nanmax(lon) > 180 else sub_lon
            for vi, _name, cands, negate in wanted:
                src = next((c for c in cands if c in ds.data_vars), None)
                if src is None:
                    continue
                out["found"][vi] = src
                out["units"][vi] = str(ds[src].attrs.get("units", ""))
                da = _surface2d(ds[src], la, lo)
                if "time" in da.dims:
                    t = np.asarray(da["time"].values).astype("datetime64[D]")
                    for d in np.unique(t):
                        sel = np.where(t == d)[0]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            f = np.nanmean(np.asarray(da.isel(time=sel).values, "f8"), axis=0)
                        out["days"].setdefault(str(d), {})[vi] = _coarse(f, lat, lon, sub_lat, qlon, negate)
                else:  # static
                    f = np.asarray(da.values, "f8")
                    out["days"].setdefault("static", {})[vi] = _coarse(f, lat, lon, sub_lat, qlon, negate)
        out["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        out["status"], out["error"] = "failed", f"{type(exc).__name__}: {exc}"
    return out


def _coarse(f, lat, lon, qlat, qlon, negate: bool) -> np.ndarray:
    v = bilinear(f, lat, lon, qlat.ravel(), qlon.ravel()).reshape(qlat.shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(v, axis=-1)
    return (-m if negate else m).astype("f4")


def build(cfg: Config, workers: int = 0, log=print) -> Path:
    import zarr

    p = cfg.raw["processing"]["physics"]
    grid = TileGrid(cfg)
    k = int(p["coarse_cells"])
    dyn, stat = p["variables"], p.get("static", [])
    pm = ProcessingManifest(cfg)
    rows = pm.db.execute("SELECT output_path, product FROM outputs WHERE step='crop' AND status='done'").fetchall()
    files = defaultdict(list)
    for r in rows:
        files[r["product"]].append(str(cfg.data_root / r["output_path"]))
    phys_days = [w for w in cfg.physical_windows()]
    day0 = min(w.start for w in phys_days)
    day1 = max(w.end for w in phys_days)
    nd = (day1 - day0).days + 1
    log(f"physics: {grid.n_tiles} tiles x {nd} days ({day0}..{day1}) x {len(dyn)} variables, {k}x{k} cells")
    jobs = []
    for prod, paths in files.items():
        wanted = [(i, v["name"], v["candidates"], bool(v.get("negate"))) for i, v in enumerate(dyn) if prod in v["products"]]
        wanted += [(1000 + i, v["name"], v["candidates"], bool(v.get("negate"))) for i, v in enumerate(stat) if prod in v["products"]]
        for path in sorted(paths):
            if wanted:
                jobs.append((str(cfg.path), prod, path, wanted, day0.isoformat()))
    values = np.full((nd, grid.n_tiles, len(dyn), k, k), np.nan, dtype="f4")
    static = np.full((grid.n_tiles, max(1, len(stat)), k, k), np.nan, dtype="f4")
    found: dict[int, set] = defaultdict(set)
    units: dict[int, str] = {}
    sources: list[dict] = []
    n_fail = 0
    n = max(1, workers or (os.cpu_count() or 2) // 2)
    with get_context("spawn").Pool(n) as pool:
        for i, res in enumerate(pool.imap_unordered(physics_file, jobs, chunksize=4), 1):
            if res["status"] != "done":
                n_fail += 1
                log(f"  FAILED {res['path']}: {res['error']}")
                continue
            sources.append({"product": res["product"], "file": Path(res["path"]).name})
            for vi, src in res["found"].items():
                found[vi].add(f"{res['product']}:{src}")
                units[vi] = res["units"].get(vi, "")
            for day, per in res["days"].items():
                if day == "static":
                    for vi, a in per.items():
                        static[:, vi - 1000] = a
                    continue
                di = (date.fromisoformat(day) - day0).days
                if not 0 <= di < nd:
                    continue
                for vi, a in per.items():
                    cur = values[di, :, vi]
                    # imerg_final and imerg_late never overlap; any other duplicate keeps the first
                    values[di, :, vi] = np.where(np.isfinite(cur), cur, a)
            if i % 500 == 0 or i == len(jobs):
                log(f"  {i:,}/{len(jobs):,} files")
    _, _, c_lon, c_lat = coarse_targets(cfg, grid)
    dest = physics_path(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    import shutil

    if tmp.exists():
        shutil.rmtree(tmp)
    root = zarr.open_group(str(tmp), mode="w")
    root.create_dataset("values", data=values, chunks=(32, 16, len(dyn), k, k))
    root.create_dataset("static", data=static, chunks=static.shape)
    root.create_dataset("lon", data=c_lon.astype("f4"), chunks=c_lon.shape)
    root.create_dataset("lat", data=c_lat.astype("f4"), chunks=c_lat.shape)
    var_meta = [{"name": v["name"], "products": v["products"], "source": sorted(found.get(i, [])),
                 "units": units.get(i, "")} for i, v in enumerate(dyn)]
    root.attrs.update({"day0": day0.isoformat(), "n_days": nd, "variables": var_meta,
                       "static_variables": [{"name": v["name"], "source": sorted(found.get(1000 + i, []))}
                                            for i, v in enumerate(stat)],
                       "tile_ids": grid.ids, "coarse_cells": k, "context_tiles": p["context_tiles"],
                       "subsamples": p["subsamples"], "grid": grid.spec(), "code_version": code_version(cfg),
                       "created": datetime.now(timezone.utc).isoformat()})
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    cover = np.isfinite(values).any(axis=(1, 3, 4)).mean(axis=0)
    for v, c in zip(var_meta, cover):
        log(f"  {v['name']:11s} days with data {c:6.1%}  from {', '.join(v['source']) or 'NOT FOUND'}")
    pm.record(dest, "physics", None, "done" if n_fail == 0 else "failed", sources[:50] + ([{"more": len(sources) - 50}] if len(sources) > 50 else []),
              code_version(cfg), {k2: p[k2] for k2 in ("coarse_cells", "context_tiles", "subsamples")},
              ["daily mean of sub-daily steps", "bilinear subsamples averaged per coarse cell", "surface level of 3-D fields"],
              n_records=nd, error=f"{n_fail} files failed" if n_fail else None)
    pm.close()
    log(f"physics -> {dest}")
    return dest
