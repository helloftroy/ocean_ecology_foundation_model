"""Regional crops of the physical fields (step 2.2).

Each verified raw file becomes one netCDF under processed/regional/physical/<product>/<year>/,
cut to the pilot domain plus `processing.physical_crop.extra_margin_deg`. Values are copied
without decoding (same dtype, scale_factor, add_offset, _FillValue, units and attributes, same
time coordinate); only the lat/lon extent changes and zlib compression is applied. Provenance goes
into `oefm_*` global attributes and the processing manifest. The raw files are never modified.

Copernicus Marine files are already domain subsets; they are copied at their full extent so every
physical product is read from one place.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, Region, load_config
from .provenance import ProcessingManifest, code_version, regional_root

LAT_NAMES = ("lat", "latitude")
LON_NAMES = ("lon", "longitude")


def crop_box(cfg: Config) -> Region:
    m = float(cfg.raw["processing"]["physical_crop"]["extra_margin_deg"])
    d = cfg.domain()
    return Region("crop", "pilot domain + physical context margin", d.west - m, d.south - m, d.east + m, d.north + m)


def out_path(cfg: Config, row: dict) -> Path:
    year = (row.get("time_start") or "static")[:4]
    stem = Path(row["filename"]).name
    for ext in (".nc4", ".nc", ".h5"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return regional_root(cfg) / "physical" / row["product"] / year / f"{stem}.nc"


def _coord(ds, names):
    for n in names:
        if n in ds.variables:
            return n
    raise KeyError(f"no coordinate among {names}")


def crop_file(args: tuple[str, dict, str, str, dict]) -> dict[str, Any]:
    """Worker: crop one raw file. Returns a result dict for the parent to record."""
    cfg_path, row, src, dest, code = args
    import xarray as xr

    cfg = load_config(cfg_path)
    box = crop_box(cfg)
    res: dict[str, Any] = {"row": row, "dest": dest}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = xr.open_dataset(src, decode_cf=False, mask_and_scale=False, decode_times=False)
        with ds:
            la, lo = _coord(ds, LAT_NAMES), _coord(ds, LON_NAMES)
            lat, lon = np.asarray(ds[la].values, dtype="f8"), np.asarray(ds[lo].values, dtype="f8")
            west, east = box.west, box.east
            if np.nanmax(lon) > 180:  # 0-360 longitudes (CCMP): keep them, shift the box instead
                west, east = west % 360, east % 360
            ilat = np.where((lat >= box.south) & (lat <= box.north))[0]
            ilon = np.where((lon >= west) & (lon <= east))[0]
            if ilat.size == 0 or ilon.size == 0:
                res["status"], res["error"] = "empty", "no grid points inside the crop box"
                return res
            sub = ds.isel({ds[la].dims[0]: slice(ilat[0], ilat[-1] + 1), ds[lo].dims[0]: slice(ilon[0], ilon[-1] + 1)})
            sub = sub.load()
            sub.attrs.update({
                "oefm_source_file": row["filename"], "oefm_source_sha256": row.get("sha256") or "",
                "oefm_source_granule_key": row["granule_key"],
                "oefm_crop_bbox_wsen": [float(box.west), float(box.south), float(box.east), float(box.north)],
                "oefm_crop_index_lat": [int(ilat[0]), int(ilat[-1]) + 1],
                "oefm_crop_index_lon": [int(ilon[0]), int(ilon[-1]) + 1],
                "oefm_code_version": str(code), "oefm_created": datetime.now(timezone.utc).isoformat(),
                "oefm_transformation": "lat/lon subset only; values, dtypes, scaling and attributes unchanged",
            })
            enc = {}
            for name, v in sub.variables.items():
                e: dict[str, Any] = {}
                if v.dtype.kind in "iuf" and v.ndim >= 1:
                    e.update(zlib=True, complevel=4)
                if v.ndim >= 2:
                    e["chunksizes"] = tuple(1 if d == 0 and v.ndim > 2 else s for d, s in enumerate(v.shape))
                if v.dtype.kind in "iuf":
                    enc[name] = e
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            tmp = dest + ".tmp"
            sub.to_netcdf(tmp, format="NETCDF4", encoding=enc)
            os.replace(tmp, dest)
            res["status"] = "done"
            res["shape"] = {k: int(v) for k, v in sub.sizes.items()}
    except Exception as exc:  # noqa: BLE001 - record and continue
        res["status"], res["error"] = "failed", f"{type(exc).__name__}: {exc}"
    return res


def run(cfg: Config, products: list[str] | None = None, workers: int = 0, limit: int = 0, log=print) -> dict:
    from .manifest import Manifest

    phys = [k for k, p in cfg.physical_products.items() if not products or k in products]
    man = Manifest(cfg.manifest_path)
    rows = [dict(r) for r in man.rows(f"status='verified' AND product IN ({','.join('?' * len(phys))})", tuple(phys))]
    man.close()
    pm = ProcessingManifest(cfg)
    done = pm.done("crop")
    code = code_version(cfg)
    box = crop_box(cfg)
    todo = []
    for r in sorted(rows, key=lambda r: (r["product"], r["time_start"] or "")):
        dest = out_path(cfg, r)
        if str(dest.relative_to(cfg.data_root)) in done and dest.exists():
            continue
        todo.append((str(cfg.path), r, str(cfg.data_root / r["local_path"]), str(dest), code))
    if limit:
        todo = todo[:limit]
    log(f"crop: {len(rows):,} verified physical files, {len(todo):,} to do; box {box.bbox}")
    stats: dict[str, int] = {}
    params = {"crop_bbox_wsen": list(box.bbox), "extra_margin_deg": cfg.raw["processing"]["physical_crop"]["extra_margin_deg"]}
    n = max(1, workers or (os.cpu_count() or 2) // 2)
    with get_context("spawn").Pool(n) as pool:
        for i, res in enumerate(pool.imap_unordered(crop_file, todo, chunksize=2), 1):
            r = res["row"]
            stats[res["status"]] = stats.get(res["status"], 0) + 1
            pm.record(Path(res["dest"]), "crop", r["product"], res["status"],
                      [{"granule_key": r["granule_key"], "filename": r["filename"], "sha256": r.get("sha256")}],
                      code, {**params, "shape": res.get("shape")},
                      ["subset to crop box (index slicing, no interpolation)", "zlib compression"],
                      n_records=1, error=res.get("error"))
            if res["status"] == "failed":
                log(f"  FAILED {r['filename']}: {res['error']}")
            if i % 200 == 0 or i == len(todo):
                log(f"  {i:,}/{len(todo):,} {stats}")
    pm.close()
    return stats
