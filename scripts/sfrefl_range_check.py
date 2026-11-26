"""Which SFREFL rhos values fall outside the stored int16 range, and what are they? (read-only)

K2 (job 116523) reported 291,515,357 clipped SFREFL cell-band values over 1,776 granules (~164,000
per granule); the K1 trial, with clouds emptied, had 1,142 over 3 granules. In tiling v3 a cell mean
below the range (rhos < OFFSET - 32766 * SCALE = -0.1106) is clipped to -32767, which is FILL, so it
is stored as MISSING; a mean above 2.5107 is stored as 2.5107. Before raw is deleted, find out which
side, which bands and which pixels.

Part 1 (from processing.sqlite): clipped values per granule (distribution, top granules).
Part 2 (from raw files): for --granules verified SFREFL granules spread over time, every --stride-th
scan line, per band: pixels below / above the range, min, 0.1 % and 99.9 % quantiles, max; and, for
out-of-range pixels, the share carrying each l2_flags bit (CLDICE, HILT, LAND, HISOLZEN, ...) and the
solar zenith there. Tiling averages pixels into cells, so pixel counts here are an upper bound on
cell counts.

    python scripts/sfrefl_range_check.py [--granules 6] [--stride 4] [--files NAME ...] 2>&1 | tail -120

--files adds named granules (e.g. the top clipped ones from part 1) to the evenly spaced sample.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PRODUCT = "pace_oci_sfrefl"


def db_summary(cfg) -> None:
    from oefm.provenance import regional_root

    p = regional_root(cfg) / "processing.sqlite"
    db = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=300)
    rows = db.execute("SELECT granule_key, filename, clipped FROM v3_granules WHERE product=? AND status='done'",
                      (PRODUCT,)).fetchall()
    db.close()
    if not rows:
        print("no done SFREFL granules in v3_granules")
        return
    c = np.array([r[2] or 0 for r in rows], dtype=float)
    q = np.percentile(c, [0, 10, 25, 50, 75, 90, 99, 100])
    print(f"Part 1: {len(rows):,} SFREFL granules, clipped total {c.sum():,.0f}; granules with 0 clipped "
          f"{(c == 0).sum():,}; percentiles 0/10/25/50/75/90/99/100: " + " ".join(f"{v:,.0f}" for v in q))
    print("  top 10 granules: " + "; ".join(f"{r[1]} {r[2]:,}" for r in sorted(rows, key=lambda r: -(r[2] or 0))[:10]))


def check(path: Path, stride: int, lo: float, hi: float) -> None:
    from oefm.inspect import L2Swath

    sw = L2Swath(path)
    try:
        nr, _ = sw.lat.shape
        bits, fmap = sw.flags(slice(0, nr), slice(None))
        bits = bits[::stride].astype("i8") & 0xFFFFFFFF
        fmap = {k: int(v) & 0xFFFFFFFF for k, v in fmap.items()}
        first = sw.bands[0]
        v = np.ma.filled(np.ma.asarray(sw.geo[first[1]][::stride, :, :], dtype="f4"), np.nan)
        wl = np.array(sw.wavelengths)
        try:
            solz = np.ma.filled(np.ma.asarray(sw.ds["geolocation_data"]["solar_zenith"][::stride, :], dtype="f4"), np.nan)
        except Exception:  # noqa: BLE001
            solz = None
    finally:
        sw.close()
    fin = np.isfinite(v)
    below, above = fin & (v < lo), fin & (v > hi)
    n_pix = fin.any(-1).sum()
    print(f"\n{path.name}: {v.shape[0]} of {nr} lines (stride {stride}), {v.shape[2]} bands, "
          f"{n_pix:,} pixels with any finite band; cell-band values below {lo:.4f}: {below.sum():,} "
          f"({below.sum() / max(fin.sum(), 1):.2e} of finite), above {hi:.4f}: {above.sum():,}")
    if not (below.any() or above.any()):
        return
    nb = below.sum((0, 1))
    na = above.sum((0, 1))
    worst = np.argsort(-(nb + na))[:8]
    print(f"   {'wl_nm':>7s} {'below':>9s} {'above':>9s} {'min':>9s} {'q0.001':>9s} {'q0.999':>9s} {'max':>9s}")
    for k in sorted(worst):
        x = v[..., k][fin[..., k]]
        print(f"   {wl[k]:7.1f} {nb[k]:9,d} {na[k]:9,d} {x.min():9.4f} {np.quantile(x, 0.001):9.4f} "
              f"{np.quantile(x, 0.999):9.4f} {x.max():9.4f}")
    bands_hit = (nb > 0).sum(), (na > 0).sum()
    print(f"   bands with any value below: {bands_hit[0]} of {v.shape[2]}; above: {bands_hit[1]}")
    for side, m in (("below", below.any(-1)), ("above", above.any(-1))):
        k = int(m.sum())
        if not k:
            continue
        share = {name: float(((bits[m] & bit) != 0).mean()) for name, bit in fmap.items() if bit}
        top = sorted(share.items(), key=lambda kv: -kv[1])[:8]
        base = {name: float(((bits & bit) != 0).mean()) for name, _ in top for bit in [fmap[name]]}
        sz = f"; solar zenith median {np.nanmedian(solz[m]):.1f} deg (swath {np.nanmedian(solz):.1f})" if solz is not None else ""
        print(f"   pixels with a band {side}: {k:,}; flag shares there (whole swath): "
              + ", ".join(f"{n} {s:.2f} ({base[n]:.2f})" for n, s in top) + sz)


def main(argv=None) -> None:
    from oefm.config import load_config
    from oefm.manifest import Manifest
    from oefm.tiling3 import SFREFL_OFFSET, SFREFL_SCALE

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--granules", type=int, default=6)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--files", nargs="*", default=[], help="granule file names to add")
    args = ap.parse_args(argv)
    cfg = load_config()
    lo, hi = SFREFL_OFFSET - 32766 * SFREFL_SCALE, SFREFL_OFFSET + 32767 * SFREFL_SCALE
    db_summary(cfg)
    man = Manifest(cfg.manifest_path)
    cand = [dict(r) for r in man.rows("status IN ('verified','tiled') AND product=?", (PRODUCT,))]
    man.close()
    if not cand:
        raise SystemExit("no SFREFL granule in the manifest")
    pick = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, args.granules + 2)[1:-1]] if args.granules else []
    by_name = {Path(r["local_path"]).name: r for r in cand}
    for f in args.files:
        if f in by_name:
            pick.append(by_name[f])
        else:
            print(f"{f}: not in the manifest as verified/tiled")
    print(f"\nPart 2: {len(pick)} raw granules, range kept {lo:.4f} .. {hi:.4f}")
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
