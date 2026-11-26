"""Reflectance values outside NASA's valid range, read as tiling v3 reads them (read-only).

K2 (job 116523) reported 291,515,357 clipped SFREFL cell-band values over 1,776 granules (~164,000
per granule); the K1 trial, with clouds emptied, had 1,142 over 3 granules. In tiling v3 a cell mean
below the stored range (rhos < -0.1106) is clipped to -32767, which is FILL, so it is stored as
MISSING; a mean above 2.5107 is stored as 2.5107.

The first version of this check read with netCDF4's default masking, which also hides values outside
the variable's valid_min / valid_max, and found none. Tiling reads raw values and removes only
_FillValue, so pixels NASA marks as outside the valid range enter the cell means. This version reads
like tiling (raw, only _FillValue removed, then scale/offset) and reports, per granule:
  the reflectance variable's _FillValue, valid_min, valid_max (decoded);
  per band (8 worst): values below valid_min / above valid_max, below / above the stored int16 range
  (SFREFL only), min, max, and the 0.1 / 99.9 % quantiles of the out-of-valid values;
  for pixels with any band out of the valid range: the share carrying each l2_flags bit (vs the
  whole swath) and the median solar zenith.
Part 1 (SFREFL only, from processing.sqlite): clipped values per granule.

    python scripts/sfrefl_range_check.py [--product pace_oci_sfrefl] [--granules 6] [--stride 4] \
        [--files NAME ...] 2>&1 | tail -150

--files adds named granules (e.g. the top clipped ones from part 1) to the evenly spaced sample.
Works for any L2 product (e.g. --product pace_oci_aop or viirsn_oc, to see whether Rrs has the same
issue).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def db_summary(cfg, product: str) -> None:
    from oefm.provenance import regional_root

    p = regional_root(cfg) / "processing.sqlite"
    db = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=300)
    rows = db.execute("SELECT granule_key, filename, clipped FROM v3_granules WHERE product=? AND status='done'",
                      (product,)).fetchall()
    db.close()
    if not rows:
        print(f"no done {product} granules in v3_granules")
        return
    c = np.array([r[2] or 0 for r in rows], dtype=float)
    q = np.percentile(c, [0, 10, 25, 50, 75, 90, 99, 100])
    print(f"Part 1: {len(rows):,} {product} granules, clipped total {c.sum():,.0f}; granules with 0 clipped "
          f"{(c == 0).sum():,}; percentiles 0/10/25/50/75/90/99/100: " + " ".join(f"{v:,.0f}" for v in q))


def _attr(var, name):
    return var.getncattr(name) if name in var.ncattrs() else None


def _decode_attr(x, scale, offset):
    if x is None:
        return None
    return float(np.asarray(x, dtype="f8").ravel()[0]) * (scale or 1.0) + offset


def read_like_tiling(var, rows: slice, idx=None) -> tuple[np.ndarray, dict]:
    var.set_auto_maskandscale(False)
    raw = np.asarray(var[rows, :] if idx is None else var[rows, :, idx])
    fill = _attr(var, "_FillValue")
    scale = float(_attr(var, "scale_factor")) if _attr(var, "scale_factor") is not None else None
    offset = float(_attr(var, "add_offset")) if _attr(var, "add_offset") is not None else 0.0
    a = raw.astype("f8")
    if fill is not None:
        a[raw == fill] = np.nan
    if scale is not None and not str(var.dtype).startswith("float"):
        a = a * scale + offset
    vr = _attr(var, "valid_range")
    vmin = _decode_attr(_attr(var, "valid_min") if vr is None else np.asarray(vr).ravel()[0],
                        scale if not str(var.dtype).startswith("float") else None, offset if scale else 0.0)
    vmax = _decode_attr(_attr(var, "valid_max") if vr is None else np.asarray(vr).ravel()[-1],
                        scale if not str(var.dtype).startswith("float") else None, offset if scale else 0.0)
    return a, {"fill": fill, "vmin": vmin, "vmax": vmax, "dtype": str(var.dtype)}


def check(path: Path, stride: int, store_lo: float | None, store_hi: float | None) -> None:
    from oefm.inspect import L2Swath

    sw = L2Swath(path)
    try:
        nr, _ = sw.lat.shape
        rows = slice(0, nr, stride)
        bits, fmap = sw.flags(slice(0, nr), slice(None))
        bits = bits[::stride].astype("i8") & 0xFFFFFFFF
        fmap = {k: int(v) & 0xFFFFFFFF for k, v in fmap.items()}
        wl = np.array(sw.wavelengths)
        first = sw.bands[0]
        if first[2] is not None:
            v, meta = read_like_tiling(sw.geo[first[1]], rows)
        else:
            parts = [read_like_tiling(sw.geo[name], rows) for _, name, _ in sw.bands]
            v, meta = np.stack([p[0] for p in parts], -1), parts[0][1]
        try:
            g = sw.ds["geolocation_data"]
            g["solar_zenith"].set_auto_maskandscale(True)
            solz = np.ma.filled(np.ma.asarray(g["solar_zenith"][rows, :], dtype="f4"), np.nan)
        except Exception:  # noqa: BLE001
            solz = None
    finally:
        sw.close()
    fin = np.isfinite(v)
    vmin = meta["vmin"] if meta["vmin"] is not None else -np.inf
    vmax = meta["vmax"] if meta["vmax"] is not None else np.inf
    lo_v, hi_v = fin & (v < vmin), fin & (v > vmax)
    print(f"\n{path.name}: {v.shape[0]} of {nr} lines (stride {stride}), {v.shape[2]} bands, {meta['dtype']}, "
          f"_FillValue {meta['fill']}, valid range {meta['vmin']} .. {meta['vmax']}; finite values {fin.sum():,}; "
          f"below valid_min {lo_v.sum():,} ({lo_v.sum() / max(fin.sum(), 1):.2e}), above valid_max {hi_v.sum():,} "
          f"({hi_v.sum() / max(fin.sum(), 1):.2e})")
    if store_lo is not None:
        lo_s, hi_s = fin & (v < store_lo), fin & (v > store_hi)
        print(f"   below the stored range {store_lo:.4f}: {lo_s.sum():,}; above {store_hi:.4f}: {hi_s.sum():,}")
    bad = lo_v | hi_v
    if not bad.any():
        return
    nb, na = lo_v.sum((0, 1)), hi_v.sum((0, 1))
    worst = np.argsort(-(nb + na))[:8]
    print(f"   {'wl_nm':>7s} {'<vmin':>9s} {'>vmax':>9s} {'min':>11s} {'max':>11s} {'bad_q0.001':>11s} {'bad_q0.999':>11s}")
    for k in sorted(worst):
        x = v[..., k][fin[..., k]]
        b = v[..., k][bad[..., k]]
        bq = (f"{np.quantile(b, 0.001):11.4g} {np.quantile(b, 0.999):11.4g}") if b.size else f"{'-':>11s} {'-':>11s}"
        print(f"   {wl[k]:7.1f} {nb[k]:9,d} {na[k]:9,d} {x.min():11.4g} {x.max():11.4g} {bq}")
    print(f"   bands with any value below valid_min: {(nb > 0).sum()} of {v.shape[2]}; above valid_max: {(na > 0).sum()}")
    m = bad.any(-1)
    allbad = (bad | ~fin).all(-1) & m
    share = {name: float(((bits[m] & bit) != 0).mean()) for name, bit in fmap.items() if bit}
    top = sorted(share.items(), key=lambda kv: -kv[1])[:10]
    base = {name: float(((bits & fmap[name]) != 0).mean()) for name, _ in top}
    sz = (f"; solar zenith median {np.nanmedian(solz[m]):.1f} deg (swath {np.nanmedian(solz):.1f})"
          if solz is not None else "")
    print(f"   pixels with any band out of the valid range: {int(m.sum()):,} of {m.size:,} "
          f"({int(allbad.sum()):,} with every band out or fill); no flag bit set: "
          f"{float((bits[m] == 0).mean()):.3f}; flag shares there (whole swath): "
          + ", ".join(f"{n} {s:.2f} ({base[n]:.2f})" for n, s in top) + sz)


def main(argv=None) -> None:
    from oefm.config import load_config
    from oefm.manifest import Manifest
    from oefm.tiling3 import SFREFL_OFFSET, SFREFL_SCALE

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", default="pace_oci_sfrefl")
    ap.add_argument("--granules", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--files", nargs="*", default=[], help="granule file names to add")
    args = ap.parse_args(argv)
    cfg = load_config()
    sfrefl = args.product == "pace_oci_sfrefl"
    lo, hi = (SFREFL_OFFSET - 32766 * SFREFL_SCALE, SFREFL_OFFSET + 32767 * SFREFL_SCALE) if sfrefl else (None, None)
    db_summary(cfg, args.product)
    man = Manifest(cfg.manifest_path)
    cand = [dict(r) for r in man.rows("status IN ('verified','tiled') AND product=?", (args.product,))]
    man.close()
    if not cand:
        raise SystemExit(f"no {args.product} granule in the manifest")
    pick = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, args.granules + 2)[1:-1]] if args.granules else []
    by_name = {Path(r["local_path"]).name: r for r in cand}
    for f in args.files:
        if f in by_name:
            pick.append(by_name[f])
        else:
            print(f"{f}: not in the manifest as verified/tiled")
    print(f"\nPart 2: {len(pick)} raw {args.product} granules, read raw (only _FillValue removed)")
    for r in pick:
        p = cfg.data_root / r["local_path"]
        if not p.exists():
            print(f"{p.name}: missing", flush=True)
            continue
        try:
            check(p, args.stride, lo, hi)
        except Exception as exc:  # noqa: BLE001
            print(f"{p.name}: failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
