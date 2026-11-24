"""Which l2_flags bits co-occur with fill (missing) reflectance in standard OB.DAAC L2 files?

Before tiles store "the mean over pixels with valid Rrs, no flag mask" (tile format v3), check
whether NASA's processing already writes fill values into Rrs (or rhos) for flagged pixels (the
"L2 mask default"). If HISATZEN, STRAYLIGHT or COCCOLITH pixels are filled, those parts of the flag
policy cannot be applied from standard files.

For a few verified granules per product (from the manifest; reads only, writes one small CSV):
  per flag bit: pixels with the bit set (over all swath pixels), the share of those with every band
  fill, with any band fill; the same for pixels with no bit set; and the geophysical variables with
  dtype, scale_factor, add_offset and _FillValue (to see what can be stored as int16 and which
  extra products, e.g. chlor_a, Kd_490, nflh, each file carries).

    python scripts/flag_fill_check.py [--per-product 2] [--stride 4] [--out reports/flag_fill_check.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(path: Path, product: str, stride: int) -> tuple[list[dict], list[dict]]:
    from oefm.inspect import L2Swath

    sw = L2Swath(path)
    try:
        nr, nc = sw.lat.shape
        rs, cs = slice(0, nr, stride), slice(0, nc)
        bits, fmap = sw.flags(slice(0, nr), cs)
        bits = bits[::stride].astype("i8") & 0xFFFFFFFF          # flag_masks hold bit 31 as a negative int32
        fmap = {k: int(v) & 0xFFFFFFFF for k, v in fmap.items()}
        first = sw.bands[0]
        if first[2] is not None:   # one 3-D variable (PACE): bands on the last axis
            v = np.ma.filled(np.ma.asarray(sw.geo[first[1]][rs, cs, :], dtype="f4"), np.nan)
            fin = np.isfinite(v)
        else:
            fin = np.stack([np.isfinite(np.ma.filled(np.ma.asarray(sw.geo[name][rs, cs], dtype="f4"), np.nan))
                            for _, name, _ in sw.bands], axis=-1)
        all_fill, any_fill = ~fin.any(-1), ~fin.all(-1)
        rows = []
        n = bits.size
        for name, bit in list(fmap.items()) + [("(no bit set)", 0)]:
            sel = (bits & bit) != 0 if bit else bits == 0
            k = int(sel.sum())
            rows.append({"product": product, "granule": path.name, "flag": name, "bit": int(bit),
                         "pixels": k, "share_of_swath": round(k / n, 5),
                         "all_bands_fill": round(float(all_fill[sel].mean()), 4) if k else None,
                         "any_band_fill": round(float(any_fill[sel].mean()), 4) if k else None})
        var_rows = []
        for vname, var in sw.geo.variables.items():
            a = {k: var.getncattr(k) for k in var.ncattrs()}
            var_rows.append({"product": product, "granule": path.name, "variable": vname, "dtype": str(var.dtype),
                             "shape": "x".join(map(str, var.shape)), "scale_factor": a.get("scale_factor"),
                             "add_offset": a.get("add_offset"), "fill": a.get("_FillValue"),
                             "units": a.get("units")})
        return rows, var_rows
    finally:
        sw.close()


def main(argv=None) -> None:
    import pandas as pd

    from oefm.config import load_config
    from oefm.manifest import Manifest

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-product", type=int, default=2)
    ap.add_argument("--stride", type=int, default=4, help="read every Nth scan line (memory)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    man = Manifest(cfg.manifest_path)
    rows, var_rows = [], []
    for prod in cfg.satellite_products:
        cand = [dict(r) for r in man.rows("status='verified' AND product=?", (prod,))]
        if not cand:
            print(f"{prod}: no verified granule", flush=True)
            continue
        # spread over time, prefer daytime-centre granules: take evenly spaced files
        pick = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, args.per_product + 2)[1:-1]]
        for r in pick:
            p = cfg.data_root / r["local_path"]
            print(f"{prod}: {p.name}", flush=True)
            try:
                a, b = check(p, prod, args.stride)
            except Exception as exc:  # noqa: BLE001
                print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            rows += a
            var_rows += b
    man.close()
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no granule could be read")
    out = Path(args.out) if args.out else cfg.reports_dir / "flag_fill_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pd.DataFrame(var_rows).to_csv(out.with_name(out.stem + "_variables.csv"), index=False)
    pd.set_option("display.width", 250)
    g = (df[df["pixels"] > 0].groupby(["product", "flag"])
         .apply(lambda d: pd.Series({"pixels": int(d["pixels"].sum()),
                                     "all_fill": round(float((d["all_bands_fill"] * d["pixels"]).sum() / d["pixels"].sum()), 3),
                                     "any_fill": round(float((d["any_band_fill"] * d["pixels"]).sum() / d["pixels"].sum()), 3)}))
         .reset_index())
    print("\nShare of pixels with fill reflectance, by flag bit (all bands fill / any band fill):")
    print(g.pivot_table(index="flag", columns="product", values="all_fill").round(3).to_string())
    print("\nany band fill:")
    print(g.pivot_table(index="flag", columns="product", values="any_fill").round(3).to_string())
    v = pd.DataFrame(var_rows).drop_duplicates(["product", "variable"])
    print("\nGeophysical variables per product (dtype, scale, offset, fill):")
    print(v[["product", "variable", "dtype", "scale_factor", "add_offset", "fill"]].to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
