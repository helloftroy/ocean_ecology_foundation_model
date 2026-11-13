"""Satellite granules -> tiles on the fixed grid (step 2.3), one overpass at a time.

Regridding is area-weighted. Each L2 pixel is treated as a footprint spanning half the distance
to its neighbours along scan and along track; the footprint is sub-sampled k_scan x k_track times
(k chosen so sub-samples are at most half a target cell apart) and each sub-sample carries
1 / (k_scan * k_track) of the pixel. A cell's value is the weight-averaged reflectance of the valid
pixels overlapping it. This handles the 1-6 km swath-edge pixels of PACE and MODIS without holes,
and 300 m OLCI pixels without dropping any.

Per granule, one zipped Zarr store `tiles/<version>/<product>/<year>/<granule>.zarr.zip` holds
every pilot tile that has at least one valid 1.2 km cell:

    reflectance      (tile, band, 32, 32)  float32  Rrs (sr^-1) or rhos; NaN where no valid pixel
    w_valid          (tile, 32, 32)        float32  valid pixel-equivalents (training mask)
    w_strict         (tile, 32, 32)        float32  pixel-equivalents passing the strict mask
    w_all            (tile, 32, 32)        float32  all pixels incl. masked (swath coverage)
    n_pixels         (tile, 32, 32)        uint16   distinct valid pixels contributing
    flags_or         (tile, 32, 32)        uint32   OR of l2_flags over every overlapping pixel
    senz             (tile, 32, 32)        float32  sensor zenith (deg), area-weighted, all pixels
    tile_time_unix   (tile,)               float64  area-weighted scan-line time (UTC seconds)
    tile_solz        (tile,)               float32  solar zenith at the scan centre (deg)
    wavelength       (band,)               float32  nm
  OLCI only, native 300 m:
    reflectance_300m (tile, band, 128, 128), w_valid_300m, w_all_300m, n_pixels_300m, flags_or_300m

Chunks are one tile each, so the loader reads only the tiles it needs. Attributes carry the
product, sensor, version, granule, source sha256, flag meanings and bit masks, mask policy, grid
spec and code version. Arrays are stored south-up (row 0 = southern edge), as in oefm/grid.py.
One zip per granule keeps the file count on GPFS at about one per granule.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, load_config
from .grid import TileGrid
from .inspect import L2Swath, mask_bits
from .provenance import ProcessingManifest, code_version, processing_version, regional_root
from .qcpolicy import policy
from .zenith import nadir_column, sensor_zenith

ROW_BLOCK = 256
MAX_K = 12


def tiles_root(cfg: Config) -> Path:
    return regional_root(cfg) / "tiles" / processing_version(cfg)


def out_path(cfg: Config, row: dict) -> Path:
    stem = Path(row["filename"]).name
    for ext in (".nc", ".nc4"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    return tiles_root(cfg) / row["product"] / (row.get("time_start") or "0000")[:4] / f"{stem}.zarr.zip"


# ----------------------------------------------------------------------------- footprint geometry
def _track_vectors(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column median step between consecutive rows (robust to VIIRS scan/bow-tie jumps)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dx, dy = np.diff(x, axis=0), np.diff(y, axis=0)
        sx, sy = np.nanmedian(dx, axis=0), np.nanmedian(dy, axis=0)
    return np.nan_to_num(sx), np.nan_to_num(sy)


def dequantise(a: np.ndarray, min_repeat_fraction: float = 0.05) -> np.ndarray:
    """Undo stair-stepped geolocation along scan.

    OB.DAAC OLCI EFR files repeat each longitude (and, through 0.001 deg rounding, latitude) over
    2-3 neighbouring columns, so about half of the along-scan steps are zero. Each run of equal
    values is assigned to the centre of the run and columns are linearly interpolated between run
    centres. Rows with few repeats (all other sensors) are returned unchanged.
    """
    a = np.asarray(a, dtype="f8")
    d = np.diff(a, axis=1)
    if not ((d == 0) & np.isfinite(d)).mean() > min_repeat_fraction:
        return a
    out = a.copy()
    cols = np.arange(a.shape[1], dtype="f8")
    for r in range(a.shape[0]):
        v = a[r]
        ok = np.isfinite(v)
        if ok.sum() < 2:
            continue
        c, vv = cols[ok], v[ok]
        change = np.r_[True, vv[1:] != vv[:-1]]
        start = np.where(change)[0]
        end = np.r_[start[1:], vv.size] - 1
        centre = (c[start] + c[end]) / 2.0
        if centre.size < 2:
            continue
        val = vv[start]
        res = np.interp(c, centre, val)
        lo, hi = c < centre[0], c > centre[-1]  # half-runs at the ends: extend the end slopes
        res[lo] = val[0] + (val[1] - val[0]) / (centre[1] - centre[0]) * (c[lo] - centre[0])
        res[hi] = val[-1] + (val[-1] - val[-2]) / (centre[-1] - centre[-2]) * (c[hi] - centre[-1])
        out[r, ok] = res
    return out


def _k(spacing_m: np.ndarray | float, cell_m: float) -> np.ndarray:
    return np.clip(np.ceil(np.nan_to_num(np.asarray(spacing_m, dtype="f8")) / (cell_m / 2.0)), 1, MAX_K).astype(int)


def footprint_pairs(grid: TileGrid, x: np.ndarray, y: np.ndarray, trx: np.ndarray, try_: np.ndarray,
                    cell_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pixel flat index within the block, slot, weight) for every pixel-cell overlap of a row block."""
    nr, nc = x.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sdx, sdy = np.gradient(x, axis=1), np.gradient(y, axis=1)
        scan_sp = np.nanmedian(np.hypot(sdx, sdy), axis=0)
    track_sp = float(np.nanmedian(np.hypot(trx, try_))) if np.isfinite(trx).any() else cell_m
    kt = int(_k(track_sp, cell_m))
    ks_col = _k(scan_sp, cell_m)
    ot = (np.arange(kt) + 0.5) / kt - 0.5
    pix_l, slot_l, w_l = [], [], []
    for ks in np.unique(ks_col):
        cols = np.where(ks_col == ks)[0]
        os_ = (np.arange(ks) + 0.5) / ks - 0.5
        xs = (x[:, cols, None, None] + sdx[:, cols, None, None] * os_[None, None, None, :]
              + trx[None, cols, None, None] * ot[None, None, :, None])
        ys = (y[:, cols, None, None] + sdy[:, cols, None, None] * os_[None, None, None, :]
              + try_[None, cols, None, None] * ot[None, None, :, None])
        sl = grid.slots(xs, ys, cell_m)
        pix = (np.arange(nr)[:, None] * nc + cols[None, :])[:, :, None, None]
        pix = np.broadcast_to(pix, sl.shape)
        keep = sl >= 0
        if not keep.any():
            continue
        pix_l.append(pix[keep])
        slot_l.append(sl[keep])
        w_l.append(np.full(int(keep.sum()), 1.0 / (ks * kt), dtype="f4"))
    if not pix_l:
        e = np.zeros(0, "i8")
        return e, e, np.zeros(0, "f4")
    pix, slot, w = np.concatenate(pix_l), np.concatenate(slot_l), np.concatenate(w_l)
    # merge sub-samples of the same pixel in the same cell
    nslot = int(slot.max()) + 1
    key = pix * nslot + slot
    uk, inv = np.unique(key, return_inverse=True)
    ws = np.bincount(inv, weights=w).astype("f4")
    return uk // nslot, uk % nslot, ws


class _Acc:
    """Dense accumulators over every cell of every pilot tile for one grid resolution."""

    def __init__(self, n_tiles: int, n: int, n_bands: int, aux: bool) -> None:
        self.n = n
        size = n_tiles * n * n
        self.size = size
        self.sum = np.zeros((n_bands, size), "f4")
        self.wb = np.zeros((n_bands, size), "f4")
        self.w_valid = np.zeros(size, "f8")
        self.w_all = np.zeros(size, "f8")
        self.n_pix = np.zeros(size, "i8")
        self.flags = np.zeros(size, "u4")
        self.aux = aux
        if aux:
            self.w_strict = np.zeros(size, "f8")
            self.senz = np.zeros(size, "f8")
            self.t = np.zeros(size, "f8")

    def add(self, pix, slot, w, valid, strict, bits, vals, zen, trel) -> None:
        if slot.size == 0:
            return
        us, inv = np.unique(slot, return_inverse=True)
        m = us.size

        def bc(weights=None):
            return np.bincount(inv, weights=weights, minlength=m)

        v = valid[pix]
        self.w_all[us] += bc(w)
        self.w_valid[us] += bc(w * v)
        self.n_pix[us] += bc(v.astype("f8")).astype("i8")
        b = bits[pix]
        acc = np.zeros(m, "u4")
        for bit in range(32):
            on = ((b >> bit) & 1).astype(bool)
            if on.any():
                acc |= (np.bincount(inv[on], minlength=m) > 0).astype("u4") << np.uint32(bit)
        self.flags[us] |= acc
        if self.aux:
            self.w_strict[us] += bc(w * strict[pix])
            self.senz[us] += bc(w * zen[pix])
            self.t[us] += bc(w * trel[pix])
        for k in range(vals.shape[1]):
            vk = vals[pix, k]
            ok = v & np.isfinite(vk)
            if not ok.any():
                continue
            ww = np.where(ok, w, 0.0)
            self.sum[k, us] += bc(ww * np.where(ok, vk, 0.0)).astype("f4")
            self.wb[k, us] += bc(ww).astype("f4")


def tile_granule(args: tuple[str, dict, str, str, dict]) -> dict[str, Any]:
    """Worker: regrid one granule onto the pilot tiles and write its zipped Zarr store."""
    cfg_path, row, src, dest, code = args
    res: dict[str, Any] = {"row": row, "dest": dest, "tile_rows": []}
    try:
        cfg = load_config(cfg_path)
        prod = cfg.satellite_products[row["product"]]
        fam = prod["family"]
        pol = policy(cfg, row["product"])
        grid = TileGrid(cfg)
        sw = L2Swath(Path(src))
    except Exception as exc:  # noqa: BLE001
        res["status"], res["error"] = "failed", f"open: {type(exc).__name__}: {exc}"
        return res
    try:
        win = sw.window(grid.lonlat_box())
        if win is None:
            res["status"], res["error"] = "empty", "swath does not reach the tiles"
            return res
        rs, cs = win
        lat, lon = dequantise(sw.lat[rs, cs]), dequantise(sw.lon[rs, cs])
        x, y = grid.proj.forward(lon, lat)
        trx, try_ = _track_vectors(x, y)
        nadir = nadir_column(sw, fam)
        zen = sensor_zenith(sw, fam, rs, cs, nadir)
        bits, fmap = sw.flags(rs, cs)
        bits = bits.astype("u4")
        hard_all = mask_bits(fmap, pol.training_flags)
        strict_all = mask_bits(fmap, pol.strict_flags)
        t_row, res["time_source"] = scan_times(sw, rs)
        t0 = float(np.nanmin(t_row)) if np.isfinite(t_row).any() else np.nan
        t_row = np.where(np.isfinite(t_row), t_row, t0)
        sla = sw.ds.groups.get("scan_line_attributes")
        solz_row = (np.asarray(np.ma.filled(sla["csol_z"][rs], np.nan), dtype="f8")
                    if sla is not None and "csol_z" in sla.variables else None)
        nb = len(sw.bands)
        native = fam == "OLCI"
        acc = _Acc(grid.n_tiles, grid.tc, nb, aux=True)
        acc3 = _Acc(grid.n_tiles, grid.olci_tc, nb, aux=False) if native else None
        solz_sum = np.zeros(grid.n_tiles * grid.tc * grid.tc, "f8")
        nrow = x.shape[0]
        for r0 in range(0, nrow, ROW_BLOCK):
            r1 = min(nrow, r0 + ROW_BLOCK)
            xb, yb = x[r0:r1], y[r0:r1]
            if not np.isfinite(xb).any():
                continue
            pix, slot, w = footprint_pairs(grid, xb, yb, trx, try_, grid.cell_m)
            pix3 = slot3 = w3 = None
            if native:
                pix3, slot3, w3 = footprint_pairs(grid, xb, yb, trx, try_, grid.olci_cell_m)
            if slot.size == 0 and (slot3 is None or slot3.size == 0):
                continue
            vals = _read_bands(sw, slice(rs.start + r0, rs.start + r1), cs)  # (pixels, bands)
            bb = bits[r0:r1].ravel()
            finite_any = np.isfinite(vals).any(axis=1)
            valid = ((bb & hard_all) == 0) & finite_any & np.isfinite(xb.ravel())
            strict = (((bb & strict_all) == 0) & finite_any).astype("f8")
            zb = np.nan_to_num(zen[r0:r1].ravel().astype("f8"))
            trel = np.broadcast_to((t_row[r0:r1] - t0)[:, None], xb.shape).ravel()
            trel = np.nan_to_num(trel)
            acc.add(pix, slot, w, valid, strict, bb, vals, zb, trel)
            if solz_row is not None and slot.size:
                sz = np.broadcast_to(solz_row[r0:r1, None], xb.shape).ravel()
                solz_sum += np.bincount(slot, weights=w * np.nan_to_num(sz[pix]), minlength=solz_sum.size)
            if native and slot3.size:
                acc3.add(pix3, slot3, w3, valid, strict, bb, vals, zb, trel)
        res.update(_write(cfg, grid, sw, row, prod, pol, code, acc, acc3, solz_sum, t0, fmap, dest,
                          res.get("time_source", "")))
    except Exception as exc:  # noqa: BLE001
        res["status"], res["error"] = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        sw.close()
    return res


def scan_times(sw: L2Swath, rs: slice) -> tuple[np.ndarray, str]:
    """UTC seconds since 1970 for each row in rs, and where they came from.

    Newer OB.DAAC files have scan_line_attributes/time; older reprocessings (e.g. 2018 R2022
    files) only have year/day/msec. Last resort: linear between time_coverage_start and _end.
    """
    from datetime import datetime, timezone

    n = rs.stop - rs.start
    sla = sw.ds.groups.get("scan_line_attributes")
    names = set(sla.variables) if sla is not None else set()
    if "time" in names:
        t = np.asarray(np.ma.filled(sla["time"][rs], np.nan), dtype="f8")
        if np.isfinite(t).any():
            return t, "scan_line_attributes/time"
    if {"year", "day", "msec"} <= names:
        yr = np.asarray(np.ma.filled(sla["year"][rs], -1), dtype="i8")
        dy = np.asarray(np.ma.filled(sla["day"][rs], -1), dtype="i8")
        ms = np.asarray(np.ma.filled(sla["msec"][rs], -1), dtype="f8")
        ok = (yr > 1900) & (dy > 0) & (ms >= 0)
        if ok.any():
            t = np.full(n, np.nan)
            jan1 = {y: datetime(int(y), 1, 1, tzinfo=timezone.utc).timestamp() for y in np.unique(yr[ok])}
            t[ok] = np.array([jan1[y] for y in yr[ok]]) + (dy[ok] - 1) * 86400.0 + ms[ok] / 1000.0
            return t, "scan_line_attributes/year,day,msec"
    attrs = sw.ds.ncattrs()
    if "time_coverage_start" in attrs and "time_coverage_end" in attrs:
        def parse(v):
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
        a, b = parse(sw.ds.getncattr("time_coverage_start")), parse(sw.ds.getncattr("time_coverage_end"))
        total = sw.lat.shape[0]
        rows = np.arange(rs.start, rs.stop, dtype="f8")
        return a + (b - a) * rows / max(total - 1, 1), "time_coverage_start/end (linear)"
    return np.full(n, np.nan), "none"


def _read_bands(sw: L2Swath, rs: slice, cs: slice) -> np.ndarray:
    """All bands of a row block as (pixels, bands) float32, NaN where fill."""
    first = sw.bands[0]
    if first[2] is not None:  # one 3-D variable (PACE)
        v = sw.geo[first[1]][rs, cs, :]
        a = np.ma.filled(np.ma.asarray(v, dtype="f4"), np.nan)
        return a.reshape(-1, a.shape[-1])
    cols = [np.ma.filled(np.ma.asarray(sw.geo[name][rs, cs], dtype="f4"), np.nan).ravel() for _, name, _ in sw.bands]
    return np.stack(cols, axis=1)


def _finish(acc: _Acc, n_tiles: int):
    n = acc.n
    with np.errstate(invalid="ignore", divide="ignore"):
        refl = np.where(acc.wb > 0, acc.sum / acc.wb, np.nan).astype("f4")
    refl = refl.reshape(refl.shape[0], n_tiles, n, n).transpose(1, 0, 2, 3)

    def r(a, dt):
        return a.reshape(n_tiles, n, n).astype(dt)

    return refl, r


def _write(cfg, grid, sw, row, prod, pol, code, acc, acc3, solz_sum, t0, fmap, dest,
           time_source: str = "") -> dict[str, Any]:
    import zarr
    from numcodecs import Blosc

    nt, n = grid.n_tiles, grid.tc
    w_valid = acc.w_valid.reshape(nt, n * n)
    keep = np.where((w_valid > 0).any(axis=1))[0]
    if keep.size == 0:
        return {"status": "empty", "error": "no valid cell in any tile"}
    refl, r = _finish(acc, nt)
    w_all = r(acc.w_all, "f8")
    with np.errstate(invalid="ignore", divide="ignore"):
        senz = np.where(w_all > 0, r(acc.senz, "f8") / w_all, np.nan).astype("f4")
        tw = acc.w_all.reshape(nt, -1).sum(axis=1)
        t_tile = t0 + acc.t.reshape(nt, -1).sum(axis=1) / tw
        solz = (solz_sum.reshape(nt, -1).sum(axis=1) / tw).astype("f4")
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    nb = refl.shape[1]
    arrays: dict[str, np.ndarray] = {
        "reflectance": refl[keep],
        "w_valid": r(acc.w_valid, "f4")[keep], "w_strict": r(acc.w_strict, "f4")[keep],
        "w_all": w_all.astype("f4")[keep], "n_pixels": np.minimum(r(acc.n_pix, "i8"), 65535).astype("u2")[keep],
        "flags_or": r(acc.flags, "u4")[keep], "senz": senz[keep],
        "tile_time_unix": t_tile[keep], "tile_solz": solz[keep],
        "wavelength": np.asarray(sw.wavelengths, dtype="f4"),
    }
    if acc3 is not None:
        n3 = grid.olci_tc
        refl3, r3 = _finish(acc3, nt)
        arrays.update({"reflectance_300m": refl3[keep], "w_valid_300m": r3(acc3.w_valid, "f4")[keep],
                       "w_all_300m": r3(acc3.w_all, "f4")[keep],
                       "n_pixels_300m": np.minimum(r3(acc3.n_pix, "i8"), 65535).astype("u2")[keep],
                       "flags_or_300m": r3(acc3.flags, "u4")[keep]})
    else:
        n3 = None
    tmp = dest + ".tmp"
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    store = zarr.ZipStore(tmp, mode="w")
    root = zarr.group(store=store)
    for name, a in arrays.items():
        chunks = (1,) + a.shape[1:] if a.ndim > 1 else a.shape
        root.create_dataset(name, data=a, chunks=chunks, compressor=comp)
    tile_ids = [grid.ids[k] for k in keep]
    hz = fmap.get("HISATZEN", 0)
    root.attrs.update({
        "tile_ids": tile_ids, "time_source": time_source, "product": row["product"], "family": prod["family"], "platform": prod.get("platform"),
        "archive_short_name": prod.get("short_name"), "archive_version": prod.get("version"),
        "variable": sw.kind, "units": "sr^-1" if sw.kind == "Rrs" else "1",
        "granule_key": row["granule_key"], "source_file": row["filename"], "source_sha256": row.get("sha256"),
        "time_coverage_start": row.get("time_start"), "flag_meanings": list(fmap), "flag_masks": list(fmap.values()),
        "training_mask": pol.training_mask, "training_flags": pol.training_flags,
        "strict_mask": pol.strict_mask, "strict_flags": pol.strict_flags,
        "grid": grid.spec(), "code_version": code, "created": datetime.now(timezone.utc).isoformat(),
        "regridding": "area-weighted pixel footprints (sub-sampled), mean of valid pixels per cell",
        "orientation": "row 0 = southern edge (south-up)",
    })
    store.close()
    os.replace(tmp, dest)

    rows = []
    cells = n * n
    for j, k in enumerate(keep):
        wv = arrays["w_valid"][j]
        fl = arrays["flags_or"][j]
        cov = arrays["w_all"][j] > 0
        rows.append({
            "product": row["product"], "family": prod["family"], "granule_key": row["granule_key"],
            "filename": row["filename"], "row": j, "tile_id": tile_ids[j],
            "time_utc": datetime.fromtimestamp(float(arrays["tile_time_unix"][j]), timezone.utc).isoformat(timespec="seconds")
            if np.isfinite(arrays["tile_time_unix"][j]) else row.get("time_start"),
            "valid_fraction": round(float((wv > 0).sum() / cells), 4),
            "strict_fraction": round(float((arrays["w_strict"][j] > 0).sum() / cells), 4),
            "cover_fraction": round(float(cov.sum() / cells), 4),
            "hisatzen_fraction": round(float(((fl & hz) != 0)[cov].mean()), 4) if hz and cov.any() else 0.0,
            "senz_mean": round(float(np.nanmean(arrays["senz"][j])), 2),
            "senz_max": round(float(np.nanmax(arrays["senz"][j])), 2),
            "solz": round(float(arrays["tile_solz"][j]), 2) if np.isfinite(arrays["tile_solz"][j]) else None,
            "n_pixels": int(arrays["n_pixels"][j].sum()),
            "valid_fraction_300m": round(float((arrays["w_valid_300m"][j] > 0).mean()), 4) if n3 else None,
        })
    return {"status": "done", "tile_rows": rows, "n_tiles": int(keep.size), "n_bands": nb}


# ----------------------------------------------------------------------------- driver
def run(cfg: Config, products: list[str] | None = None, workers: int = 0, limit: int = 0,
        start: str | None = None, end: str | None = None, log=print) -> dict:
    from .grid import write_static
    from .manifest import Manifest

    root = tiles_root(cfg)
    grid = TileGrid(cfg)
    if not (root / "grid" / "tiles.parquet").exists():
        write_static(cfg, grid, root / "grid", log)
    sat = [k for k in cfg.satellite_products if not products or k in products]
    man = Manifest(cfg.manifest_path)
    rows = [dict(r) for r in man.rows(f"status='verified' AND product IN ({','.join('?' * len(sat))})", tuple(sat))]
    man.close()
    if start:
        rows = [r for r in rows if (r["time_start"] or "") >= start]
    if end:
        rows = [r for r in rows if (r["time_start"] or "")[:10] <= end]
    pm = ProcessingManifest(cfg)
    done = pm.done("tiles")
    code = code_version(cfg)
    todo = []
    for r in sorted(rows, key=lambda r: (r["time_start"] or "", r["product"])):
        dest = out_path(cfg, r)
        if str(dest.relative_to(cfg.data_root)) in done:
            continue
        todo.append((str(cfg.path), r, str(cfg.data_root / r["local_path"]), str(dest), code))
    if limit:
        todo = todo[:limit]
    log(f"tiles: {grid.n_tiles} pilot tiles; {len(rows):,} verified granules, {len(todo):,} to do -> {root}")
    stats: dict[str, int] = {}
    n = max(1, workers or (os.cpu_count() or 2) // 2)
    t_start = datetime.now(timezone.utc)
    with get_context("spawn").Pool(n, maxtasksperchild=20) as pool:
        for i, res in enumerate(pool.imap_unordered(tile_granule, todo, chunksize=1), 1):
            r = res["row"]
            stats[res["status"]] = stats.get(res["status"], 0) + 1
            pol = policy(cfg, r["product"])
            pm.record(Path(res["dest"]), "tiles", r["product"],
                      res["status"], [{"granule_key": r["granule_key"], "filename": r["filename"], "sha256": r.get("sha256")}],
                      code, {"grid": grid.spec(), "training_mask": pol.training_mask, "strict_mask": pol.strict_mask},
                      ["project to LAEA", "area-weighted footprint regridding to 1.2 km"
                       + (" and native 300 m" if r["product"].startswith("olci") else ""),
                       "training mask excludes pixels from the mean; flags OR-ed per cell"],
                      n_records=res.get("n_tiles"), error=res.get("error"), tile_rows=res["tile_rows"])
            if res["status"] == "failed":
                log(f"  FAILED {r['filename']}: {res['error']}")
            if i % 100 == 0 or i == len(todo):
                el = (datetime.now(timezone.utc) - t_start).total_seconds()
                log(f"  {i:,}/{len(todo):,} {stats} ({el / i:.1f} s per granule wall)")
    pm.close()
    export_index(cfg, log)
    return stats


def export_index(cfg: Config, log=print) -> Path | None:
    """tile_obs -> tiles/<version>/index/tile_observations.parquet (the loader's entry point)."""
    import pandas as pd

    pm = ProcessingManifest(cfg)
    df = pd.read_sql_query("SELECT * FROM tile_obs", pm.db)
    pm.close()
    if df.empty:
        return None
    dest = tiles_root(cfg) / "index" / "tile_observations.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["tile_id", "time_utc", "product"]).to_parquet(dest, index=False)
    summ = (df.groupby("product").agg(granules=("granule_key", "nunique"), tile_obs=("tile_id", "size"),
                                      tiles=("tile_id", "nunique"), median_valid=("valid_fraction", "median"))
            .reset_index())
    summ.to_csv(dest.parent / "summary_by_product.csv", index=False)
    log(f"Index: {len(df):,} tile observations -> {dest}")
    return dest
