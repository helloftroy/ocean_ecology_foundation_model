"""Observation index (one row per satellite granule x region) and multisensor overlap checks.

The index links each observation to its region, instrument, time, wavelengths, valid-data
coverage, the physical files for the same day, and the original file path. It deliberately
commits to no grid, projection or chip format; the 0.05-degree cells used for coverage and
the 0.02-degree bins used for sensor matchups are diagnostics only.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, Region, load_config
from .inspect import MASKS, L2Swath, mask_bits
from .manifest import Manifest

COVER_CELL_DEG = 0.05
MATCH_BIN_DEG = 0.02


# ----------------------------------------------------------------------------- per granule
def granule_stats(args: tuple[str, str, dict, str]) -> list[dict[str, Any]]:
    cfg_path, path, row, today = args
    cfg = load_config(cfg_path, date.fromisoformat(today))
    out = []
    try:
        sw = L2Swath(Path(path))
    except Exception as exc:  # noqa: BLE001
        return [{"granule_key": row["granule_key"], "error": f"{type(exc).__name__}: {exc}"}]
    try:
        wls = sw.wavelengths
        for region in cfg.regions.values():
            win = sw.window(region)
            if win is None:
                continue
            rs, cs = win
            lon, lat = sw.lon[rs, cs], sw.lat[rs, cs]
            inreg = region.contains(lon, lat)
            ref, _ = sw.band(555.0 if sw.kind == "Rrs" else 865.0, rs, cs)
            bits, fmap = sw.flags(rs, cs)
            finite = np.isfinite(ref)
            land = (bits & fmap.get("LAND", 0)) != 0
            cloud = (bits & fmap.get("CLDICE", 0)) != 0
            water = inreg & ~land
            rec: dict[str, Any] = {
                "granule_key": row["granule_key"], "region": region.key,
                "n_pixels": int(inreg.sum()), "n_water_pixels": int(water.sum()),
                "cloud_fraction": round(float(cloud[water].mean()), 4) if water.any() else None,
                "geometric_coverage": _cell_fraction(region, lon[inreg], lat[inreg]),
                "n_bands": len(wls), "wl_min": min(wls) if wls else None, "wl_max": max(wls) if wls else None,
            }
            for mname, flist in MASKS.items():
                keep = water & finite & ((bits & mask_bits(fmap, flist)) == 0)
                rec[f"valid_{mname}"] = round(float(keep.sum() / max(water.sum(), 1)), 4)
                rec[f"valid_cover_{mname}"] = _cell_fraction(region, lon[keep], lat[keep])
            out.append(rec)
    finally:
        sw.close()
    return out or [{"granule_key": row["granule_key"], "region": None, "n_pixels": 0}]


def _cell_fraction(region: Region, lon: np.ndarray, lat: np.ndarray) -> float:
    """Share of the region's 0.05-degree cells that contain at least one of the given pixels."""
    nx = max(1, int(round((region.east - region.west) / COVER_CELL_DEG)))
    ny = max(1, int(round((region.north - region.south) / COVER_CELL_DEG)))
    if lon.size == 0:
        return 0.0
    ix = np.clip(((lon - region.west) / COVER_CELL_DEG).astype(int), 0, nx - 1)
    iy = np.clip(((lat - region.south) / COVER_CELL_DEG).astype(int), 0, ny - 1)
    return round(np.unique(iy * nx + ix).size / (nx * ny), 4)


# ----------------------------------------------------------------------------- build
def build(cfg: Config, workers: int = 0, log=print) -> dict[str, Path]:
    import pandas as pd

    man = Manifest(cfg.manifest_path)
    sat = list(cfg.satellite_products)
    rows = [dict(r) for r in man.rows(f"status='verified' AND product IN ({','.join('?' * len(sat))})", tuple(sat))]
    phys = [dict(r) for r in man.rows("status='verified' AND kind='physical'")]
    man.close()
    out_dir = cfg.data_root / "index"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "granule_stats.jsonl"
    cache: dict[str, list] = defaultdict(list)
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["_sha256"]].append(rec)
    todo = [r for r in rows if r["sha256"] not in cache]
    log(f"{len(rows):,} verified satellite files; {len(todo):,} need statistics ({len(rows) - len(todo):,} cached)")
    workers = workers or max(1, min(32, (os.cpu_count() or 2) - 1))
    jobs = [(str(cfg.path), str(cfg.data_root / r["local_path"]), r, cfg.today.isoformat()) for r in todo]
    with open(cache_path, "a") as fh, ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (r, recs) in enumerate(zip(todo, pool.map(granule_stats, jobs, chunksize=4)), 1):
            for rec in recs:
                rec["_sha256"] = r["sha256"]
                fh.write(json.dumps(rec) + "\n")
                cache[r["sha256"]].append(rec)
            if i % 100 == 0 or i == len(todo):
                log(f"  statistics {i:,}/{len(todo):,}")

    products = cfg.satellite_products
    phys_by_day = _physical_lookup(phys)
    records = []
    for r in rows:
        for st in cache.get(r["sha256"], []):
            if not st.get("region"):
                continue
            p = products[r["product"]]
            t0 = r["time_start"]
            day = t0[:10]
            rec = {
                "region": st["region"], "product": r["product"], "family": p.get("family"),
                "platform": p.get("platform"), "variable": p.get("reflectance"),
                "time_start": t0, "time_end": r["time_end"], "date": day,
                "year": int(day[:4]), "month": int(day[5:7]),
                **{k: v for k, v in st.items() if k not in ("granule_key", "region", "_sha256", "error")},
                "processing_version": r["version"], "path": r["local_path"], "granule_key": r["granule_key"],
            }
            for key, path in phys_by_day(day).items():
                rec[f"phys_{key}"] = path
            records.append(rec)
    df = pd.DataFrame.from_records(records)
    paths = {}
    obs = out_dir / "observations.csv"
    df.to_csv(obs, index=False)
    paths["observations"] = obs
    try:
        df.to_parquet(out_dir / "observations.parquet", index=False)
        paths["observations_parquet"] = out_dir / "observations.parquet"
    except Exception:  # noqa: BLE001 - parquet is a convenience
        pass
    if not df.empty:
        counts = (df.groupby(["family", "product", "region", "year", "month"])
                    .agg(observations=("granule_key", "size"),
                         with_valid_l3=("valid_l3_default", lambda s: int((s > 0).sum())),
                         mean_valid_cover_l3=("valid_cover_l3_default", "mean"))
                    .reset_index())
        counts.to_csv(out_dir / "counts_by_sensor_region_month.csv", index=False)
        paths["counts"] = out_dir / "counts_by_sensor_region_month.csv"
        co = coincidences(df)
        co.to_csv(out_dir / "multisensor_coincidences.csv", index=False)
        paths["coincidences"] = out_dir / "multisensor_coincidences.csv"
    log("Index written: " + ", ".join(str(p) for p in paths.values()))
    return paths


def _physical_lookup(phys: list[dict]):
    """date (YYYY-MM-DD) -> {product: local_path} for physical files covering that day."""
    daily: dict[tuple[str, str], str] = {}
    ranges: list[tuple[str, str, str, str]] = []
    static: dict[str, str] = {}
    for r in phys:
        if not r["time_start"]:
            static[r["product"]] = r["local_path"]
            continue
        t0 = datetime.fromisoformat(r["time_start"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat((r["time_end"] or r["time_start"]).replace("Z", "+00:00"))
        if (t1 - t0) <= timedelta(days=1, hours=1):
            daily[(r["product"], (t0 + (t1 - t0) / 2).date().isoformat())] = r["local_path"]
        else:
            ranges.append((r["product"], t0.date().isoformat(), t1.date().isoformat(), r["local_path"]))
    prods = {p for p, _ in daily} | {p for p, *_ in ranges}

    def lookup(day: str) -> dict[str, str]:
        out = dict(static)
        for p in prods:
            if (p, day) in daily:
                out[p] = daily[(p, day)]
        for p, a, b, path in ranges:
            if a <= day <= b:
                out[p] = path
        return out

    return lookup


def coincidences(df) -> "pd.DataFrame":  # noqa: F821
    """Region-days with observations from more than one instrument family (valid data > 0)."""
    import pandas as pd

    good = df[df["valid_minimal"] > 0].copy()
    good["t"] = pd.to_datetime(good["time_start"], utc=True)
    rows = []
    for (region, day), g in good.groupby(["region", "date"]):
        fams = sorted(g["family"].dropna().unique())
        if len(fams) < 2:
            continue
        rec = {"region": region, "date": day, "families": "+".join(fams), "n_families": len(fams),
               "products": "+".join(sorted(g["product"].unique()))}
        pace = g[g["family"] == "PACE"]
        for fam in ("OLCI", "MODIS", "VIIRS"):
            other = g[g["family"] == fam]
            if len(pace) and len(other):
                dt = min(abs((a - b).total_seconds()) for a in pace["t"] for b in other["t"]) / 3600
                rec[f"pace_{fam.lower()}_min_dt_h"] = round(dt, 2)
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- matchups
def matchup(cfg: Config, path_a: Path, path_b: Path, region: Region, out_dir: Path,
            wavelengths=(443, 490, 555, 665)) -> dict[str, Any]:
    """Bin two swaths onto a temporary 0.02-degree grid over a region and compare Rrs band by band."""
    from .inspect import _mpl

    def binned(path: Path):
        sw = L2Swath(path)
        try:
            win = sw.window(region)
            if win is None:
                return None, {}
            rs, cs = win
            lon, lat = sw.lon[rs, cs], sw.lat[rs, cs]
            bits, fmap = sw.flags(rs, cs)
            ok = region.contains(lon, lat) & ((bits & mask_bits(fmap, MASKS["minimal"])) == 0)
            nx = int(round((region.east - region.west) / MATCH_BIN_DEG))
            ny = int(round((region.north - region.south) / MATCH_BIN_DEG))
            ix = np.clip(((lon - region.west) / MATCH_BIN_DEG).astype(int), 0, nx - 1)
            iy = np.clip(((lat - region.south) / MATCH_BIN_DEG).astype(int), 0, ny - 1)
            grids, used = {}, {}
            for wl in wavelengths:
                band, wl_used = sw.band(wl, rs, cs)
                m = ok & np.isfinite(band)
                s = np.bincount((iy * nx + ix)[m], weights=band[m], minlength=nx * ny)
                n = np.bincount((iy * nx + ix)[m], minlength=nx * ny)
                with np.errstate(invalid="ignore", divide="ignore"):
                    grids[wl] = np.where(n > 0, s / n, np.nan).reshape(ny, nx)
                used[wl] = wl_used
            return grids, used
        finally:
            sw.close()

    ga, ua = binned(path_a)
    gb, ub = binned(path_b)
    res: dict[str, Any] = {"a": path_a.name, "b": path_b.name, "region": region.key, "bands": []}
    if ga is None or gb is None:
        res["note"] = "one of the swaths does not reach the region"
        return res
    plt = _mpl()
    fig, axes = plt.subplots(1, len(wavelengths), figsize=(2.4 * len(wavelengths), 2.5), dpi=150)
    for ax, wl in zip(np.atleast_1d(axes), wavelengths):
        a, b = ga[wl], gb[wl]
        m = np.isfinite(a) & np.isfinite(b)
        rec = {"target_nm": wl, "a_nm": ua[wl], "b_nm": ub[wl], "n_cells": int(m.sum())}
        if m.sum() > 10:
            rec["pearson_r"] = round(float(np.corrcoef(a[m], b[m])[0, 1]), 3)
            pos = m & (a > 0)
            rec["median_ratio_b_over_a"] = round(float(np.median(b[pos] / a[pos])), 3) if pos.any() else None
            rec["median_abs_diff"] = float(np.median(np.abs(b[m] - a[m])))
            lim = [np.nanpercentile(np.r_[a[m], b[m]], 1), np.nanpercentile(np.r_[a[m], b[m]], 99)]
            ax.scatter(a[m], b[m], s=2, alpha=0.4, color="#2a7fbf", rasterized=True)
            ax.plot(lim, lim, color="k", lw=0.6)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.set_xlabel(f"{ua[wl]:.0f} nm (a)")
        ax.set_ylabel(f"{ub[wl]:.0f} nm (b)")
        res["bands"].append(rec)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"matchup_{region.key}_{path_a.stem}__{path_b.stem}.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    res["figure"] = str(png)
    return res


def auto_matchups(cfg: Config, max_pairs: int = 6, max_dt_h: float = 3.0, log=print) -> Path:
    """Pick the closest-in-time PACE vs OLCI/MODIS/VIIRS pairs from the index and compare them."""
    import pandas as pd

    obs = pd.read_csv(cfg.data_root / "index" / "observations.csv")
    obs = obs[obs["variable"] == "Rrs"]
    obs["t"] = pd.to_datetime(obs["time_start"], utc=True)
    pace = obs[obs["family"] == "PACE"]
    pairs = []
    for fam in ("OLCI", "MODIS", "VIIRS"):
        other = obs[obs["family"] == fam]
        m = pace.merge(other, on=["region", "date"], suffixes=("_a", "_b"))
        if m.empty:
            continue
        m["dt_h"] = (m["t_a"] - m["t_b"]).abs().dt.total_seconds() / 3600
        m["score"] = m[["valid_cover_minimal_a", "valid_cover_minimal_b"]].min(axis=1)
        m = m[m["dt_h"] <= max_dt_h].sort_values("score", ascending=False)
        pairs += m.head(max(1, max_pairs // 3)).to_dict("records")
    out_dir = cfg.reports_dir / "overlap"
    results = []
    for p in pairs:
        r = matchup(cfg, cfg.data_root / p["path_a"], cfg.data_root / p["path_b"], cfg.regions[p["region"]], out_dir)
        r["dt_h"] = round(p["dt_h"], 2)
        results.append(r)
        log(f"  {p['region']} {p['date']}: {Path(p['path_a']).name} vs {Path(p['path_b']).name}, dt {p['dt_h']:.1f} h")
    dest = out_dir / "matchups.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=1, default=float))
    return dest
