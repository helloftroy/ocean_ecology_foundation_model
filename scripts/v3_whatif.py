"""What would each candidate tile rule save? Measured on existing v3 stores (read-only, ~1-5 min).

Run on the K1b trial stores (written without these rules). Per product:
  cloud/land rule: share of stored tiles, and of bytes, that a tile rule would drop, for
    basis "tile" (cloud-or-land weight over covered weight > X) and basis "water" (cloud weight over
    non-land weight > X), X = 0.5, 0.75, 0.9. The stores keep only per-bit shares, so cloud-or-land
    is approximated as min(1, cloud + land) (an upper bound where pixels carry both).
  swath edge: share of valid cells, and of bytes, with sensor zenith above 55 / 60 / 65 degrees, and
    the share of tiles left under 1 % valid.
  compact Rrs_unc (PACE): bytes of uint8 log10 codes (and of one band-averaged code per cell) relative
    to the stored int16, on a sample of tiles.
Bytes per tile are the store's bytes split over its tiles in proportion to valid cells.
Florida totals: MB per granule x the product's granules in the manifest (verified or tiled).

    python scripts/v3_whatif.py [--root data/processed/global/tiles/v3] [--unc-tiles 64]
"""

from __future__ import annotations

import argparse
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
THRESH = (0.5, 0.75, 0.9)
SENZ = (55, 60, 65)


def main(argv=None) -> None:
    import zarr
    from numcodecs import Blosc, blosc

    from oefm.config import load_config
    from oefm.manifest import Manifest
    from oefm.tiling3 import FILL, encode_unc, flag_share_of, v3_root

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--unc-tiles", type=int, default=64)
    args = ap.parse_args(argv)
    blosc.use_threads = False
    warnings.filterwarnings("ignore", "Mean of empty slice")
    cfg = load_config()
    root = Path(args.root) if args.root else v3_root(cfg)
    try:
        man = Manifest(cfg.manifest_path)
        ngran = {k: len(man.rows("status IN ('verified','tiled') AND product=?", (k,))) for k in cfg.satellite_products}
        man.close()
    except Exception:  # noqa: BLE001
        ngran = {}
    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    totals = {"now": 0.0, "cloud_tile_0.75": 0.0, "unc": 0.0, "edge_60": 0.0}
    for pdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "index"):
        stores = sorted(pdir.rglob("*.zarr.zip"))
        if not stores:
            continue
        tb, clf, cwf, nval, edge = [], [], [], [], {s: [] for s in SENZ}
        unc_bytes = 0
        granules = set()
        unc_ratio = unc_avg_ratio = None
        unc_sample = None
        for s in stores:
            granules.add(s.name.split("__")[0])
            with zipfile.ZipFile(s) as zf:
                per_field = {}
                for info in zf.infolist():
                    per_field[info.filename.split("/")[0]] = per_field.get(info.filename.split("/")[0], 0) + info.compress_size
                total = sum(per_field.values())
            unc_bytes += per_field.get("refl_unc", 0)
            st = zarr.ZipStore(str(s), mode="r")
            g = zarr.open_group(st, mode="r")
            wa, wv = g["w_all"][:], g["w_valid"][:]
            cld = flag_share_of(g, "CLDICE")
            lnd = flag_share_of(g, "LAND")
            senz = g["senz"][:]
            if "refl_unc" in g and unc_sample is None and g["refl_unc"].dtype == np.int16:
                k = min(args.unc_tiles, g["refl_unc"].shape[0])
                u = g["refl_unc"][:k]
                vi = g.attrs["variables"]["refl_unc"]
                unc_sample = (u, float(vi.get("scale") or 1.0), float(vi.get("offset") or 0.0))
            st.close()
            T = wa.shape[0]
            cldf = (cld.astype("f8") / 255) if cld is not None else np.zeros_like(wa)
            lndf = (lnd.astype("f8") / 255) if lnd is not None else np.zeros_like(wa)
            sw = np.maximum(wa.reshape(T, -1).sum(1), 1e-12)
            c_w = (cldf * wa).reshape(T, -1).sum(1)
            l_w = (lndf * wa).reshape(T, -1).sum(1)
            clf.append(np.minimum(1.0, (c_w + l_w) / sw))
            cwf.append(np.where(sw - l_w > 1e-9, c_w / np.maximum(sw - l_w, 1e-12), 1.0))
            nv = (wv > 0).reshape(T, -1).sum(1)
            nval.append(nv)
            tb.append(total * (nv + 1) / (nv + 1).sum())
            ang = np.where(senz == FILL, 0, senz).astype("f8") * 0.01
            for a in SENZ:
                edge[a].append(((wv > 0) & (ang > a)).reshape(T, -1).sum(1))
        tb, clf, cwf, nval = map(np.concatenate, (tb, clf, cwf, nval))
        now = tb.sum()
        ng = max(len(granules), 1)
        fl = ngran.get(pdir.name, 0)
        print(f"\n== {pdir.name}: {ng} granules, {tb.size:,} stored tiles, {now / ng / 1e6:,.1f} MB per granule"
              f"{f', {fl:,} granules in the manifest' if fl else ''}", flush=True)
        for basis, f in (("tile", clf), ("water", cwf)):
            parts = []
            for x in THRESH:
                d = f > x
                parts.append(f">{x}: {d.mean():.3f} of tiles, {tb[d].sum() / now:.3f} of bytes")
            print(f"   cloud/land rule, basis {basis:5s}: " + "; ".join(parts), flush=True)
        for a in SENZ:
            e = np.concatenate(edge[a])
            frac_cells = e.sum() / max(nval.sum(), 1)
            byt = float((tb * e / np.maximum(nval, 1)).sum())
            left = (nval - e) / 1024.0 if pdir.name[:4] != "olci" else (nval - e) / 16384.0
            print(f"   swath edge, senz > {a}: {frac_cells:.3f} of valid cells, ~{byt / now:.3f} of bytes, "
                  f"{(left < 0.01).mean():.3f} of tiles left under 1 % valid", flush=True)
        f_cloud = tb[clf > 0.75].sum()
        save_unc = 0.0
        if unc_sample is not None:
            u, sc, off = unc_sample
            v = np.where(u == FILL, np.nan, u.astype("f8") * sc + off)
            code = encode_unc(v)
            avg = encode_unc(np.nanmean(np.where(np.isfinite(v), v, np.nan), axis=1)) if np.isfinite(v).any() else code[:, 0]
            b16 = sum(len(comp.encode(np.ascontiguousarray(u[i:i + 1]))) for i in range(u.shape[0]))
            b8 = sum(len(comp.encode(np.ascontiguousarray(code[i:i + 1]))) for i in range(u.shape[0]))
            bav = sum(len(comp.encode(np.ascontiguousarray(avg[i:i + 1]))) for i in range(u.shape[0]))
            unc_ratio, unc_avg_ratio = b8 / max(b16, 1), bav / max(b16, 1)
            save_unc = unc_bytes * (1 - unc_ratio)
            ok = np.isfinite(v) & (v > 0)
            from oefm.tiling3 import decode_unc
            rel = np.abs(decode_unc(code)[ok] / v[ok] - 1)
            print(f"   Rrs_unc ({unc_bytes / now:.3f} of bytes): uint8 log10 per band {unc_ratio:.2f} x its int16 size "
                  f"(max relative error {rel.max():.3f}, median {np.median(rel):.4f}); one band-averaged code per "
                  f"cell {unc_avg_ratio:.3f} x", flush=True)
        e60 = np.concatenate(edge[60])
        save_edge = float((tb * e60 / np.maximum(nval, 1)).sum())
        per = lambda b: b / ng / 1e6  # noqa: E731
        print(f"   MB per granule: now {per(now):.1f}; cloud/land tile>0.75 saves {per(f_cloud):.1f}; compact Rrs_unc "
              f"saves {per(save_unc):.1f}; both {per(now - f_cloud - save_unc * (1 - f_cloud / now)):.1f}; edge > 60 deg "
              f"would save ~{per(save_edge):.1f} more", flush=True)
        if fl:
            totals["now"] += per(now) * fl / 1e3
            totals["cloud_tile_0.75"] += per(f_cloud) * fl / 1e3
            totals["unc"] += per(save_unc * (1 - f_cloud / now)) * fl / 1e3
            totals["edge_60"] += per(save_edge) * fl / 1e3
    if totals["now"]:
        t = totals
        print(f"\nAll manifest granules (GB): now {t['now']:,.0f}; cloud/land rule saves {t['cloud_tile_0.75']:,.0f}; "
              f"compact Rrs_unc saves {t['unc']:,.0f} -> {t['now'] - t['cloud_tile_0.75'] - t['unc']:,.0f}; "
              f"edge > 60 deg would save ~{t['edge_60']:,.0f} more (overlaps the cloud rule)")


if __name__ == "__main__":
    main()
