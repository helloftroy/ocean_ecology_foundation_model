"""Tiling v3: L2 granules -> catalog_v1 tiles, keeping everything a redownload would be needed for.

Design (decision log "Tiling v3"): tiles store what cannot be recomputed without the raw granule;
design choices (flag policy, transform, normalisation, windows, sampling) are applied by the loader.
Raw granules can then be deleted (download-tile-delete, oefm/stream.py).

Coverage: every catalog_v1 tile the swath touches (full swath, every zone; border tiles in both
zones), at native resolution: 1.2 km cells (32 x 32) for MODIS, VIIRS, PACE; 300 m cells (128 x 128)
for OLCI only (the loader aggregates 1.2 km). A tile observation with under `min_valid` (1 %) of its
cells valid stores no arrays, but every overpass (tile x granule with any pixel) is recorded in the
overpass table.

Regridding as v2: area-weighted pixel footprints (sub-sampled), no flag mask in the mean.

Per cell, per stored tile (one zipped Zarr per granule and zone, `<granule>__<zone>.zarr.zip`):
  refl          (tile, band, n, n) int16  Rrs (or PACE SFREFL rhos): mean over pixels with a finite
                value (no flag mask), quantised with the source's scale/offset (|error| <= half a
                source step). SFREFL (float32 source): scale 2e-5, offset 0.6, and pixels flagged
                CLDICE or LAND excluded (cloud / land cells empty).
  refl_clean    (tile, band, n, n) int16  mean over pixels passing the product's training mask, ONLY
                in mixed cells (some contributing pixels hard-flagged, some not); fill elsewhere
  refl_unc      (tile, band, n, n) int16  PACE Rrs_unc (same rule as refl)
  x_<name>      (tile, n, n)  every single-layer geophysical field the file has (chlor_a, Kd_490,
                nflh, aot_*, angstrom, pic, poc, par, ipar, avw, PACE BGC fields ...): int16 with the
                source scale/offset, or float32 for float sources
  w_all         (tile, n, n) float32  pixel-equivalents overlapping the cell (swath coverage)
  w_valid       (tile, n, n) float32  pixel-equivalents with a finite reflectance (after the SFREFL rule)
  w_clean       (tile, n, n) float32  ... of which pass the training mask
  n_pixels      (tile, n, n) uint16   distinct pixels overlapping the cell
  flags_or      (tile, n, n) uint32   OR of l2_flags over the overlapping pixels
  flag_share    (tile, 32, n, n) uint8  share of the cell's pixel weight carrying each bit, x 255
  dt            (tile, n, n) int16    seconds from tile_time (area-weighted scan time)
  senz, sena    (tile, n, n) int16    sensor zenith / azimuth, 0.01 deg (reconstructed from geolocation:
                sub-satellite point = nadir column; azimuth = bearing from the cell to it)
  solz, sola    (tile, n, n) int16    solar zenith / azimuth, 0.01 deg (computed from time and position)
  tile_time     (tile,) float64       UTC seconds
Attributes: catalog spec, product, wavelengths, flag names/masks, training mask, every variable's
scale/offset/fill, source sha256, code version.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import catalog as C
from .config import Config, load_config
from .inspect import L2Swath, mask_bits
from .tiling import ROW_BLOCK, _track_vectors, dequantise, footprint_pairs
from .zenith import nadir_column, sensor_zenith

FILL = -32767
SFREFL_SCALE, SFREFL_OFFSET = 2e-5, 0.6
SFREFL_EMPTY_FLAGS = ("CLDICE", "LAND")
MIN_VALID = 0.01
ANGLE_SCALE = 0.01
TILE_STEP = "tiles_v3"


def v3_root(cfg: Config) -> Path:
    """Tile store shared by every config (catalog_v1 IDs are global): `processing.v3_root` relative to
    the repo root, default <data_root>/processed/global/tiles/v3."""
    from .config import REPO_ROOT

    r = (cfg.raw.get("processing") or {}).get("v3_root")
    if r:
        return Path(r) if Path(r).is_absolute() else REPO_ROOT / r
    return cfg.data_root / "processed" / "global" / "tiles" / "v3"


def out_path(root: Path, row: dict, zone: str) -> Path:
    stem = Path(row["filename"]).name
    for ext in (".nc", ".nc4"):
        stem = stem.removesuffix(ext)
    return root / row["product"] / (row.get("time_start") or "0000")[:4] / f"{stem}__{zone}.zarr.zip"


# ----------------------------------------------------------------------------- geometry helpers
class ZoneSlots:
    """slots(x, y, cell_m) for footprint_pairs over the member tiles of one zone inside a box of tiles."""

    def __init__(self, zone: C.Zone, tx: np.ndarray, ty: np.ndarray, n: int) -> None:
        self.zone, self.n = zone, n
        self.tx0, self.ty0 = int(tx.min()), int(ty.min())
        w, h = int(tx.max()) - self.tx0 + 1, int(ty.max()) - self.ty0 + 1
        gx, gy = np.meshgrid(np.arange(w) + self.tx0, np.arange(h) + self.ty0)
        member = C.is_member(zone, gx.ravel(), gy.ravel()).reshape(h, w)
        self.lut = np.full((h, w), -1, "i8")
        self.ix, self.iy = gx[member], gy[member]
        self.lut[member] = np.arange(int(member.sum()))

    @property
    def n_tiles(self) -> int:
        return len(self.ix)

    def slots(self, x, y, cell_m=None):
        n = self.n
        cm = C.TILE_M / n
        with np.errstate(invalid="ignore"):
            cx, cy = np.floor(x / cm), np.floor(y / cm)
        ok = np.isfinite(cx) & np.isfinite(cy)
        cx = np.where(ok, cx, 0).astype("i8")
        cy = np.where(ok, cy, 0).astype("i8")
        tx, ty = np.floor_divide(cx, n), np.floor_divide(cy, n)
        li, lj = tx - self.tx0, ty - self.ty0
        inb = ok & (li >= 0) & (lj >= 0) & (lj < self.lut.shape[0]) & (li < self.lut.shape[1])
        t = np.full(np.shape(x), -1, "i8")
        t[inb] = self.lut[lj[inb], li[inb]]
        out = t * n * n + (cy - ty * n) * n + (cx - tx * n)
        out[t < 0] = -1
        return out


def solar_angles(t_unix: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solar zenith and azimuth (deg, azimuth clockwise from north), NOAA solar-position equations
    (accuracy ~0.01 deg for 1950-2050)."""
    jd = np.asarray(t_unix, "f8") / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    l0 = np.radians((280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360)
    m = np.radians(357.52911 + jc * (35999.05029 - 0.0001537 * jc))
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    c = (np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + np.sin(2 * m) * (0.019993 - 0.000101 * jc)
         + np.sin(3 * m) * 0.000289)
    omega = np.radians(125.04 - 1934.136 * jc)
    lam = l0 + np.radians(c) - np.radians(0.00569 + 0.00478 * np.sin(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = np.radians(eps0 + 0.00256 * np.cos(omega))
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    y = np.tan(eps / 2) ** 2
    eot = 4 * np.degrees(y * np.sin(2 * l0) - 2 * e * np.sin(m) + 4 * e * y * np.sin(m) * np.cos(2 * l0)
                         - 0.5 * y * y * np.sin(4 * l0) - 1.25 * e * e * np.sin(2 * m))
    minutes = (np.asarray(t_unix, "f8") % 86400.0) / 60.0
    tst = (minutes + eot + 4 * np.asarray(lon, "f8")) % 1440
    ha = np.radians(np.where(tst / 4 < 0, tst / 4 + 180, tst / 4 - 180))
    la = np.radians(np.asarray(lat, "f8"))
    cosz = np.clip(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha), -1, 1)
    zen = np.arccos(cosz)
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(decl) * np.cos(la))) + 180.0
    return np.degrees(zen), az % 360.0


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing from point 1 to point 2 (deg clockwise from north)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, "f8") - np.asarray(lon1, "f8"))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arctan2(y, x)) % 360.0


# ----------------------------------------------------------------------------- variables
def _attrs(var) -> dict:
    a = {k: var.getncattr(k) for k in var.ncattrs()}
    return {"dtype": str(var.dtype), "scale": float(a["scale_factor"]) if "scale_factor" in a else None,
            "offset": float(a["add_offset"]) if "add_offset" in a else 0.0,
            "fill": float(a["_FillValue"]) if "_FillValue" in a else None, "units": str(a.get("units", ""))}


def variables(sw: L2Swath, product: str) -> dict[str, Any]:
    """What to store from this file: reflectance family (+ PACE Rrs_unc) and single-layer fields."""
    geo = sw.geo
    refl_names = {name for _, name, _ in sw.bands}
    out: dict[str, Any] = {"refl": None, "refl_unc": None, "extras": {}}
    if sw.bands:
        first = sw.bands[0]
        out["refl"] = {"three_d": first[2] is not None, "names": [b[1] for b in sw.bands],
                       "wavelength": [b[0] for b in sw.bands], **_attrs(geo[first[1]])}
        if out["refl"]["dtype"].startswith("float"):   # PACE SFREFL rhos
            out["refl"].update(scale=SFREFL_SCALE, offset=SFREFL_OFFSET, requantised=True)
        if "Rrs_unc" in geo.variables and geo["Rrs_unc"].ndim == 3:
            out["refl_unc"] = {"three_d": True, "names": ["Rrs_unc"] * len(sw.bands), **_attrs(geo["Rrs_unc"])}
    for name, var in geo.variables.items():
        if name in refl_names or name in ("l2_flags", "Rrs_unc") or var.ndim != 2:
            continue
        if var.shape != sw.lat.shape:
            continue
        out["extras"][name] = _attrs(var)
    return out


def _decode(raw: np.ndarray, info: dict) -> np.ndarray:
    a = raw.astype("f8")
    if info.get("fill") is not None:
        a[raw == info["fill"]] = np.nan
    if not info["dtype"].startswith("float") and info.get("scale"):
        a = a * info["scale"] + info["offset"]
    return a


def _quantise(mean: np.ndarray, info: dict) -> tuple[np.ndarray, int]:
    """int16 at the stored scale/offset; returns (array, number clipped)."""
    q = np.round((mean - info["offset"]) / info["scale"])
    fin = np.isfinite(q)
    clipped = int(((q < -32766) | (q > 32767))[fin].sum())
    q = np.clip(np.where(fin, q, FILL), -32767, 32767).astype("i2")
    q[~fin] = FILL
    return q, clipped


def _read_raw(var, rs, cs, idx=None):
    var.set_auto_maskandscale(False)
    a = var[rs, cs] if idx is None else var[rs, cs, idx]
    return np.asarray(a)


# ----------------------------------------------------------------------------- one granule
def tile_granule(args: tuple) -> dict[str, Any]:
    """Worker: one granule -> one store per zone with stored tiles. Returns per-zone results with
    stored-tile rows and overpass rows."""
    cfg_path, row, src, root, code = args
    res: dict[str, Any] = {"row": row, "zones": [], "error": None}
    try:
        cfg = load_config(cfg_path)
        prod = cfg.satellite_products[row["product"]]
        from .qcpolicy import policy

        pol = policy(cfg, row["product"])
        sw = L2Swath(Path(src))
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"open: {type(exc).__name__}: {exc}"
        return res
    try:
        fam = prod["family"]
        olci = fam == "OLCI"
        n = C.OLCI_TILE_CELLS if olci else C.TILE_CELLS
        lat_all, lon_all = dequantise(sw.lat), dequantise(sw.lon)
        zkeys = C.zones_near(lat_all[::8, ::8], lon_all[::8, ::8])
        nadir = nadir_column(sw, fam) if fam in ("MODIS", "VIIRS", "OLCI", "PACE") else lat_all.shape[1] // 2
        info = variables(sw, row["product"])
        from .tiling import scan_times

        t_row_all, tsrc = scan_times(sw, slice(0, lat_all.shape[0]))
        t0 = float(np.nanmin(t_row_all)) if np.isfinite(t_row_all).any() else np.nan
        for zk in zkeys:
            res["zones"].append(_tile_zone(cfg, sw, row, prod, pol, info, C.zone_map()[zk], lat_all, lon_all,
                                           n, nadir, t_row_all, t0, tsrc, Path(root), code))
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        sw.close()
    return res


def _tile_zone(cfg, sw, row, prod, pol, info, zone, lat_all, lon_all, n, nadir, t_row_all, t0, tsrc, root, code):
    fam = prod["family"]
    cm = C.TILE_M / n
    out = {"zone": zone.key, "dest": str(out_path(root, row, zone.key)), "status": "empty", "tiles": [],
           "overpasses": [], "clipped": 0}
    x_all, y_all = zone.proj.forward(lon_all, lat_all)
    fin = np.isfinite(x_all) & np.isfinite(y_all)
    if not fin.any():
        return out
    tx = np.floor(np.where(fin, x_all, 0) / C.TILE_M).astype("i8")
    ty = np.floor(np.where(fin, y_all, 0) / C.TILE_M).astype("i8")
    # candidate tiles: those containing pixel centres; the slot table keeps only zone members
    ut = np.unique(np.stack([tx[fin], ty[fin]], 1), axis=0)
    mem = C.is_member(zone, ut[:, 0], ut[:, 1])
    if not mem.any():
        return out
    mt = ut[mem]
    near = np.zeros_like(fin)
    keyset = set(map(tuple, mt.tolist()))
    # pixels in (or within one tile of) a member tile
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            k = (tx + dx) * 1_000_003 + (ty + dy)
            want = np.array([a * 1_000_003 + b for a, b in keyset], dtype="i8")
            near |= fin & np.isin(k, want)
    rr, cc = np.where(near)
    rs, cs = slice(int(rr.min()), int(rr.max()) + 1), slice(int(cc.min()), int(cc.max()) + 1)
    slots_obj = ZoneSlots(zone, np.r_[mt[:, 0] - 1, mt[:, 0] + 1], np.r_[mt[:, 1] - 1, mt[:, 1] + 1], n)
    x, y = x_all[rs, cs], y_all[rs, cs]
    trx, try_ = _track_vectors(x, y)
    pix_l, slot_l, w_l = [], [], []
    nc = x.shape[1]
    for r0 in range(0, x.shape[0], ROW_BLOCK):
        r1 = min(x.shape[0], r0 + ROW_BLOCK)
        if not np.isfinite(x[r0:r1]).any():
            continue
        p, s, w = footprint_pairs(slots_obj, x[r0:r1], y[r0:r1], trx, try_, cm)
        if s.size:
            pix_l.append(p + r0 * nc)
            slot_l.append(s)
            w_l.append(w)
    if not slot_l:
        return out
    pix, slot, w = np.concatenate(pix_l), np.concatenate(slot_l), np.concatenate(w_l).astype("f8")
    cells, inv = np.unique(slot, return_inverse=True)
    m = cells.size

    def bc(weights=None):
        return np.bincount(inv, weights=weights, minlength=m)

    # per-pixel inputs over the window
    bits, fmap = sw.flags(rs, cs)
    bits = (bits.astype("i8") & 0xFFFFFFFF).ravel()
    fmap = {k: int(v) & 0xFFFFFFFF for k, v in fmap.items()}
    hard = mask_bits(fmap, pol.training_flags) & 0xFFFFFFFF
    pb = bits[pix]
    w_all = bc(w)
    n_pix = bc()
    flags_or = np.zeros(m, "u4")
    share = np.zeros((32, m), "u1")
    for b in range(32):
        on = ((pb >> b) & 1).astype(bool)
        if on.any():
            wb = bc(np.where(on, w, 0.0))
            flags_or |= (wb > 0).astype("u4") << np.uint32(b)
            share[b] = np.round(255 * wb / np.maximum(w_all, 1e-12)).astype("u1")
    t_row = t_row_all[rs]
    t_row = np.where(np.isfinite(t_row), t_row, t0)
    trel = np.broadcast_to((t_row - t0)[:, None], x.shape).ravel()[pix]
    t_cell = bc(w * trel) / np.maximum(w_all, 1e-12)
    zen = sensor_zenith(sw, fam, rs, cs, nadir).ravel()[pix].astype("f8")
    senz = bc(w * np.nan_to_num(zen)) / np.maximum(w_all, 1e-12)
    la, lo = lat_all[rs, cs].ravel()[pix], lon_all[rs, cs].ravel()[pix]
    nla = np.broadcast_to(lat_all[rs, nadir][:, None], x.shape).ravel()[pix]
    nlo = np.broadcast_to(lon_all[rs, nadir][:, None], x.shape).ravel()[pix]
    az = np.radians(bearing_deg(la, lo, nla, nlo))
    sena = np.degrees(np.arctan2(bc(w * np.sin(az)), bc(w * np.cos(az)))) % 360.0

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    empty_bits = mask_bits(fmap, list(SFREFL_EMPTY_FLAGS)) & 0xFFFFFFFF
    w_valid = np.zeros(m)
    w_clean = np.zeros(m)
    if info["refl"] is not None:
        ri = info["refl"]
        sfrefl = bool(ri.get("requantised"))
        keep_pix = ((pb & empty_bits) == 0) if sfrefl else np.ones(pix.size, bool)
        clean_pix = keep_pix & ((pb & hard) == 0)
        var = sw.geo[ri["names"][0]] if ri["three_d"] else None
        raw3 = _read_raw(var, rs, cs) if ri["three_d"] else None
        nb = len(ri["names"])
        refl = np.full((nb, m), np.nan)
        refl_c = np.full((nb, m), np.nan)
        anyfin = np.zeros(pix.size, bool)
        anyclean = np.zeros(pix.size, bool)
        src = dict(ri, scale=None, offset=0.0) if sfrefl else ri   # decode float rhos as is
        for k in range(nb):
            raw = raw3[:, :, k] if ri["three_d"] else _read_raw(sw.geo[ri["names"][k]], rs, cs)
            v = _decode(raw.ravel()[pix], src)
            ok = np.isfinite(v) & keep_pix
            okc = ok & clean_pix
            anyfin |= ok
            anyclean |= okc
            s1, w1 = bc(np.where(ok, w * np.nan_to_num(v), 0.0)), bc(np.where(ok, w, 0.0))
            s2, w2 = bc(np.where(okc, w * np.nan_to_num(v), 0.0)), bc(np.where(okc, w, 0.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                refl[k] = np.where(w1 > 0, s1 / w1, np.nan)
                mixed = (w2 > 0) & (w2 < w1 - 1e-9)
                refl_c[k] = np.where(mixed, s2 / np.maximum(w2, 1e-12), np.nan)
        del raw3
        w_valid = bc(np.where(anyfin, w, 0.0))
        w_clean = bc(np.where(anyclean, w, 0.0))
        arrays["refl"], meta["refl"] = refl, ri
        arrays["refl_clean"] = refl_c
        if info["refl_unc"] is not None:
            ui = info["refl_unc"]
            raw3 = _read_raw(sw.geo["Rrs_unc"], rs, cs)
            unc = np.full((nb, m), np.nan)
            for k in range(nb):
                v = _decode(raw3[:, :, k].ravel()[pix], ui)
                ok = np.isfinite(v)
                s1, w1 = bc(np.where(ok, w * np.nan_to_num(v), 0.0)), bc(np.where(ok, w, 0.0))
                with np.errstate(invalid="ignore", divide="ignore"):
                    unc[k] = np.where(w1 > 0, s1 / w1, np.nan)
            del raw3
            arrays["refl_unc"], meta["refl_unc"] = unc, ui
    else:   # no reflectance (e.g. PACE BGC): validity from the extras
        anyfin = np.zeros(pix.size, bool)
    extras = {}
    for name, ei in info["extras"].items():
        v = _decode(_read_raw(sw.geo[name], rs, cs).ravel()[pix], ei)
        ok = np.isfinite(v)
        if info["refl"] is None:
            anyfin |= ok
        s1, w1 = bc(np.where(ok, w * np.nan_to_num(v), 0.0)), bc(np.where(ok, w, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            extras[name] = np.where(w1 > 0, s1 / w1, np.nan)
    if info["refl"] is None:
        w_valid = bc(np.where(anyfin, w, 0.0))

    # ------------------------------------------------------------- per tile
    tile_no, local = np.divmod(cells, n * n)
    nn = n * n
    covered = np.unique(tile_no)
    when = datetime.fromtimestamp(t0, timezone.utc) if np.isfinite(t0) else None
    stored = []
    for t in covered:
        sel = tile_no == t
        tid = C.tile_id(zone.key, slots_obj.ix[t], slots_obj.iy[t])
        nvalid = int((w_valid[sel] > 0).sum())
        vf = nvalid / nn
        tt = t0 + float(np.sum(w_all[sel] * t_cell[sel]) / max(np.sum(w_all[sel]), 1e-12))
        out["overpasses"].append({"tile_id": tid, "zone": zone.key, "product": row["product"], "family": fam,
                                  "granule_key": row["granule_key"],
                                  "time_utc": datetime.fromtimestamp(tt, timezone.utc).isoformat(timespec="seconds")
                                  if np.isfinite(tt) else row.get("time_start"),
                                  "cover_fraction": round(int(sel.sum()) / nn, 4), "valid_fraction": round(vf, 4),
                                  "stored": vf >= MIN_VALID})
        if vf >= MIN_VALID:
            stored.append((t, tid, tt))
    if not stored:
        return out

    T = len(stored)
    pos = {t: j for j, (t, _, _) in enumerate(stored)}
    rowsel = np.array([pos.get(int(t), -1) for t in tile_no])
    keep = rowsel >= 0
    tj, lc = rowsel[keep], local[keep]
    rr_, cc_ = np.divmod(lc, n)

    def dense(vals, dtype, fill):
        a = np.full((T, n, n), fill, dtype)
        a[tj, rr_, cc_] = vals[keep]
        return a

    A: dict[str, np.ndarray] = {}
    clipped = 0
    for key in ("refl", "refl_clean", "refl_unc"):
        if key not in arrays:
            continue
        vi = meta.get(key, meta.get("refl"))
        nb = arrays[key].shape[0]
        a = np.full((T, nb, n, n), FILL, "i2")
        for k in range(nb):
            q, c = _quantise(arrays[key][k], vi)
            clipped += c if key == "refl" else 0
            a[tj, k, rr_, cc_] = q[keep]
        A[key] = a
    for name, v in extras.items():
        ei = info["extras"][name]
        if ei["dtype"].startswith("float") or not ei.get("scale"):
            A[f"x_{name}"] = dense(v.astype("f4"), "f4", np.nan)
        else:
            A[f"x_{name}"] = dense(_quantise(v, ei)[0], "i2", FILL)
    A["w_all"] = dense(w_all.astype("f4"), "f4", 0.0)
    A["w_valid"] = dense(w_valid.astype("f4"), "f4", 0.0)
    A["w_clean"] = dense(w_clean.astype("f4"), "f4", 0.0)
    A["n_pixels"] = dense(np.minimum(n_pix, 65535).astype("u2"), "u2", 0)
    A["flags_or"] = dense(flags_or, "u4", 0)
    fs = np.zeros((T, 32, n, n), "u1")
    for b in range(32):
        if share[b].any():
            fs[tj, b, rr_, cc_] = share[b][keep]
    A["flag_share"] = fs
    tile_time = np.array([s[2] for s in stored], "f8")
    dtc = t0 + t_cell - tile_time[np.where(keep, rowsel, 0)]
    A["dt"] = dense(np.clip(np.round(np.nan_to_num(dtc)), -32767, 32767).astype("i2"), "i2", FILL)
    A["senz"] = dense(np.round(senz / ANGLE_SCALE).astype("i2"), "i2", FILL)
    A["sena"] = dense(np.round(sena / ANGLE_SCALE).astype("i2"), "i2", FILL)
    # solar angles at cell centres and cell times
    sz = np.full((T, n, n), FILL, "i2")
    sa = np.full((T, n, n), FILL, "i2")
    for j, (t, tid, tt) in enumerate(stored):
        lo_c, la_c = C.tile_lonlat(zone, slots_obj.ix[t], slots_obj.iy[t], n)
        m_t = tj == j
        tcell = np.full((n, n), tt)
        tcell[rr_[m_t], cc_[m_t]] = (t0 + t_cell[keep][m_t])
        z_, a_ = solar_angles(tcell, la_c, lo_c)
        cov = A["w_all"][j] > 0
        sz[j][cov] = np.round(z_[cov] / ANGLE_SCALE).astype("i2")
        sa[j][cov] = np.round(a_[cov] / ANGLE_SCALE).astype("i2")
    A["solz"], A["sola"] = sz, sa
    A["tile_time"] = tile_time
    if "refl" in arrays:
        A["wavelength"] = np.asarray(info["refl"]["wavelength"], "f4")

    attrs = {"catalog": C.CATALOG_VERSION, "zone": zone.key, "tile_ids": [s[1] for s in stored],
             "cells_per_side": n, "cell_m": C.TILE_M / n, "product": row["product"], "family": fam,
             "platform": prod.get("platform"), "archive_short_name": prod.get("short_name"),
             "archive_version": prod.get("version"), "variable": sw.kind, "granule_key": row["granule_key"],
             "source_file": row["filename"], "source_sha256": row.get("sha256"), "time_source": tsrc,
             "flag_meanings": list(fmap), "flag_masks": list(fmap.values()),
             "training_mask": pol.training_mask, "training_flags": pol.training_flags,
             "sfrefl_empty_flags": list(SFREFL_EMPTY_FLAGS) if info["refl"] and info["refl"].get("requantised") else [],
             "variables": {"refl": info["refl"], "refl_unc": info["refl_unc"], "extras": info["extras"]},
             "angle_scale_deg": ANGLE_SCALE, "fill_int16": FILL, "min_valid_fraction": MIN_VALID,
             "code_version": code, "created": datetime.now(timezone.utc).isoformat(),
             "regridding": "area-weighted pixel footprints (sub-sampled); mean of pixels with a finite value, no flag mask",
             "orientation": "row 0 = southern edge (south-up)"}
    _write_store(Path(out["dest"]), A, attrs)
    out.update(status="done", clipped=clipped)
    for j, (t, tid, tt) in enumerate(stored):
        wv = A["w_valid"][j]
        out["tiles"].append({"tile_id": tid, "zone": zone.key, "product": row["product"], "family": fam,
                             "granule_key": row["granule_key"], "row": j,
                             "time_utc": datetime.fromtimestamp(tt, timezone.utc).isoformat(timespec="seconds"),
                             "valid_fraction": round(float((wv > 0).mean()), 4),
                             "clean_fraction": round(float((A["w_clean"][j] > 0).mean()), 4),
                             "cover_fraction": round(float((A["w_all"][j] > 0).mean()), 4),
                             "senz_mean": round(float(np.nanmean(np.where(A["senz"][j] == FILL, np.nan, A["senz"][j]))) * ANGLE_SCALE, 2)
                             if (A["senz"][j] != FILL).any() else None})
    return out


def _write_store(dest: Path, arrays: dict[str, np.ndarray], attrs: dict) -> None:
    import zarr
    from numcodecs import Blosc

    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dest) + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    store = zarr.ZipStore(tmp, mode="w")
    root = zarr.group(store=store)
    for name, a in arrays.items():
        chunks = (1,) + a.shape[1:] if a.ndim > 1 else a.shape
        root.create_dataset(name, data=a, chunks=chunks, compressor=comp)
    root.attrs.update(json.loads(json.dumps(attrs, default=lambda o: o.item() if hasattr(o, "item") else str(o))))
    store.close()
    os.replace(tmp, dest)


# ----------------------------------------------------------------------------- driver
SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS v3_granules (
    granule_key TEXT PRIMARY KEY, product TEXT, filename TEXT, sha256 TEXT, status TEXT, error TEXT,
    zones TEXT, stored_tiles INTEGER, overpasses INTEGER, clipped INTEGER, code TEXT, finished TEXT);
CREATE TABLE IF NOT EXISTS v3_tiles (
    tile_id TEXT, zone TEXT, product TEXT, family TEXT, granule_key TEXT, output_path TEXT, row INTEGER,
    time_utc TEXT, valid_fraction REAL, clean_fraction REAL, cover_fraction REAL, senz_mean REAL,
    PRIMARY KEY (granule_key, tile_id));
CREATE TABLE IF NOT EXISTS v3_overpasses (
    tile_id TEXT, zone TEXT, product TEXT, family TEXT, granule_key TEXT, time_utc TEXT,
    cover_fraction REAL, valid_fraction REAL, stored INTEGER, PRIMARY KEY (granule_key, tile_id));
"""


def v3_db(cfg: Config):
    import sqlite3

    from .provenance import regional_root

    p = regional_root(cfg) / "processing.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(p, timeout=300)
    db.execute("PRAGMA journal_mode=DELETE")
    db.executescript(SCHEMA_V3)
    return db


def record(cfg: Config, db, res: dict, code: dict) -> str:
    """Write one granule's result; returns 'done' | 'failed'."""
    row = res["row"]
    status = "failed" if res["error"] else "done"
    rel = lambda p: str(Path(p).relative_to(v3_root(cfg))) if p else None  # noqa: E731
    with db:
        for z in res["zones"]:
            db.executemany("INSERT OR REPLACE INTO v3_tiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           [(t["tile_id"], t["zone"], t["product"], t["family"], t["granule_key"], rel(z["dest"]),
                             t["row"], t["time_utc"], t["valid_fraction"], t["clean_fraction"], t["cover_fraction"],
                             t["senz_mean"]) for t in z["tiles"]])
            db.executemany("INSERT OR REPLACE INTO v3_overpasses VALUES (?,?,?,?,?,?,?,?,?)",
                           [(o["tile_id"], o["zone"], o["product"], o["family"], o["granule_key"], o["time_utc"],
                             o["cover_fraction"], o["valid_fraction"], int(o["stored"])) for o in z["overpasses"]])
        db.execute("INSERT OR REPLACE INTO v3_granules VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (row["granule_key"], row["product"], row["filename"], row.get("sha256"), status, res["error"],
                    ",".join(z["zone"] for z in res["zones"]),
                    sum(len(z["tiles"]) for z in res["zones"]), sum(len(z["overpasses"]) for z in res["zones"]),
                    sum(z["clipped"] for z in res["zones"]), json.dumps(code), datetime.now(timezone.utc).isoformat()))
    return status


def done_keys(db) -> set[str]:
    return {r[0] for r in db.execute("SELECT granule_key FROM v3_granules WHERE status='done'")}


def run(cfg: Config, rows: list[dict] | None = None, products: list[str] | None = None, workers: int = 0,
        limit: int = 0, log=print) -> dict:
    """Tile verified granules not yet tiled (v3). rows: a given batch (stream), else all verified."""
    from multiprocessing import get_context

    from .manifest import Manifest
    from .provenance import code_version

    root = v3_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    (root / "catalog.json").write_text(json.dumps(C.spec(), indent=1))
    db = v3_db(cfg)
    done = done_keys(db)
    if rows is None:
        sat = [k for k in cfg.satellite_products if not products or k in products]
        man = Manifest(cfg.manifest_path)
        rows = [dict(r) for r in man.rows(f"status='verified' AND product IN ({','.join('?' * len(sat))})", tuple(sat))]
        man.close()
    todo = [r for r in rows if r["granule_key"] not in done]
    if limit:
        todo = todo[:limit]
    code = code_version(cfg)
    log(f"tiles v3: {len(rows):,} granules, {len(todo):,} to tile -> {root}")
    stats: dict[str, int] = {}
    t_start = datetime.now(timezone.utc)
    nw = max(1, workers or (os.cpu_count() or 2) // 2)
    jobs = [(str(cfg.path), r, str(cfg.data_root / r["local_path"]), str(root), code) for r in todo]
    with get_context("spawn").Pool(nw, maxtasksperchild=10) as pool:
        for i, res in enumerate(pool.imap_unordered(tile_granule, jobs, chunksize=1), 1):
            st = record(cfg, db, res, code)
            stats[st] = stats.get(st, 0) + 1
            if res["error"]:
                log(f"  FAILED {res['row']['filename']}: {res['error']}")
            if i % 50 == 0 or i == len(jobs):
                el = (datetime.now(timezone.utc) - t_start).total_seconds()
                log(f"  {i:,}/{len(jobs):,} {stats} ({el / i:.1f} s per granule wall)")
    db.close()
    return stats


def export_index(cfg: Config, log=print) -> dict[str, Path]:
    """v3 tables of this data root -> <v3_root>/index/{tile_observations,overpasses}_<phase>.parquet."""
    import pandas as pd

    db = v3_db(cfg)
    phase = str(cfg.raw.get("phase", "default"))
    out = {}
    for table, name in (("v3_tiles", "tile_observations"), ("v3_overpasses", "overpasses")):
        df = pd.read_sql_query(f"SELECT * FROM {table}", db)
        dest = v3_root(cfg) / "index" / f"{name}_{phase}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.sort_values(["tile_id", "time_utc", "product"]).to_parquet(dest, index=False)
        out[name] = dest
        log(f"index v3: {len(df):,} rows -> {dest}")
    g = pd.read_sql_query("SELECT product, status, COUNT(*) n, SUM(stored_tiles) tiles, SUM(overpasses) overpasses, "
                          "SUM(clipped) clipped FROM v3_granules GROUP BY product, status", db)
    log(g.to_string(index=False))
    db.close()
    return out
