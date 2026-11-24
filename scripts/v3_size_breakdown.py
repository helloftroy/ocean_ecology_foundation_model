"""Where do the bytes of the v3 tile stores go? (read-only; run on a login node after a tiles3 trial)

Reads every store (`<v3_root>/<product>/**/*.zarr.zip`; --per-product N to sample) and prints:
  per field: stored (compressed) MB, share of the product's bytes, uncompressed MB, ratio, and the MB
    it takes under the size rules of 2026-11-24 (0 when the field is dropped), with the shuffle chosen;
  per product: MB per granule now, and after each rule in turn:
    1 extras    only processing.v3_store.extras (nflh, chlor_a, Kd_490; PACE BGC fields)
    2 flagbits  flag_share only for processing.v3_store.flag_share_bits (flags_or keeps all 32 bits)
    4 sun       no solz / sola (the loader computes them)
    3 codec     zstd 9, better of byte / bit shuffle, chunks of whole tiles up to ~1 MB
  Steps 1, 2 and 4 are measured with the old codec, step 3 on what is left, so the columns add up.
  Stores written with the new rules (attrs flag_share_bits) are reported as they are.

    python scripts/v3_size_breakdown.py [--per-product 0] [--root data/processed/global/tiles/v3]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def old_size(a: np.ndarray) -> int:
    """Bytes under the K1 settings: zstd 3, bit shuffle, one tile per chunk."""
    from numcodecs import Blosc

    comp = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    if a.ndim <= 1:
        return len(comp.encode(np.ascontiguousarray(a)))
    return sum(len(comp.encode(np.ascontiguousarray(a[i:i + 1]))) for i in range(a.shape[0]))


def new_size(a: np.ndarray) -> tuple[int, str]:
    from oefm.tiling3 import store_chunks, store_codec

    ch = store_chunks(a)
    codec = store_codec(a, ch)
    if a.ndim <= 1:
        n = len(codec.encode(np.ascontiguousarray(a)))
    else:
        n = sum(len(codec.encode(np.ascontiguousarray(a[i:i + ch[0]]))) for i in range(0, a.shape[0], ch[0]))
    return n, {1: "byte", 2: "bit"}.get(codec.shuffle, str(codec.shuffle))


def main(argv=None) -> None:
    import pandas as pd
    import zarr

    from oefm.config import load_config
    from oefm.tiling3 import store_rules, v3_root

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-product", type=int, default=0, help="stores per product (0 = all)")
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    root = Path(args.root) if args.root else v3_root(cfg)
    field_rows, prod_rows = [], []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "index"):
        stores = sorted(pdir.rglob("*.zarr.zip"))
        if not stores:
            continue
        if args.per_product and len(stores) > args.per_product:
            stores = [stores[int(i)] for i in np.linspace(0, len(stores) - 1, args.per_product)]
        keep_extras, share_bits = store_rules(cfg, pdir.name)
        per = defaultdict(lambda: {"stored": 0, "raw": 0, "new": 0, "shuffle": set()})
        steps = defaultdict(int)
        granules, tiles, new_rules = set(), 0, 0
        for s in stores:
            granules.add(s.name.split("__")[0])
            with zipfile.ZipFile(s) as zf:
                for info in zf.infolist():
                    name = info.filename.split("/")[0]
                    per["(metadata)" if name.startswith(".") else name]["stored"] += info.compress_size
                    steps["now"] += info.compress_size
            st = zarr.ZipStore(str(s), mode="r")
            g = zarr.open_group(st, mode="r")
            attrs = g.attrs.asdict()
            arrays = {k: g[k][...] for k in g.array_keys()}
            st.close()
            tiles += int(arrays["tile_time"].shape[0])
            for k, a in arrays.items():
                per[k]["raw"] += a.nbytes
            if "flag_share_bits" in attrs:       # already written with the new rules
                new_rules += 1
                for k, a in arrays.items():
                    n, sh = new_size(a)
                    per[k]["new"] += n
                    per[k]["shuffle"].add(sh)
                    steps["codec"] += n
                continue
            # 1 extras
            drop = [k for k in arrays if k.startswith("x_") and k[2:] not in keep_extras]
            steps["extras"] += sum(old_size(arrays[k]) for k in drop)
            # 2 flag bits: rows of the bits kept (old stores hold all 32 bits, row = bit number)
            names = dict(zip(attrs["flag_meanings"], attrs["flag_masks"]))
            rows = [int(names[b] & 0xFFFFFFFF).bit_length() - 1 for b in share_bits if names.get(b)]
            fs = arrays["flag_share"]
            fs_new = fs[:, rows]
            steps["flagbits"] += old_size(fs) - old_size(fs_new)
            # 4 sun angles
            steps["sun"] += sum(old_size(arrays[k]) for k in ("solz", "sola") if k in arrays)
            kept = {k: a for k, a in arrays.items() if k not in drop and k not in ("solz", "sola")}
            kept["flag_share"] = fs_new
            before3 = sum(old_size(a) for a in kept.values())
            after3 = 0
            for k, a in kept.items():
                n, sh = new_size(a)
                per[k]["new"] += n
                per[k]["shuffle"].add(sh)
                after3 += n
            steps["codec"] += before3 - after3
        total = sum(v["stored"] for v in per.values())
        for k, v in sorted(per.items(), key=lambda kv: -kv[1]["stored"]):
            field_rows.append({"product": pdir.name, "field": k, "MB": round(v["stored"] / 1e6, 1),
                               "share": round(v["stored"] / total, 3), "raw_MB": round(v["raw"] / 1e6, 1),
                               "ratio": round(v["raw"] / v["stored"], 1) if v["stored"] else None,
                               "new_MB": round(v["new"] / 1e6, 1), "shuffle": "/".join(sorted(v["shuffle"])) or "-"})
        ng = max(len(granules), 1)
        if new_rules == len(stores):
            after = steps["now"]
            row = {"now": steps["now"]}
        else:
            after = steps["now"] - steps["extras"] - steps["flagbits"] - steps["sun"] - steps["codec"]
            row = {"now": steps["now"], "1_extras": -steps["extras"], "2_flagbits": -steps["flagbits"],
                   "4_sun": -steps["sun"], "3_codec": -steps["codec"], "after": after}
        prod_rows.append({"product": pdir.name, "granules": ng, "stores": len(stores), "tiles": tiles,
                          **{f"{k}_MB_per_granule": round(v / ng / 1e6, 1) for k, v in row.items()},
                          "after_vs_now": round(after / max(steps["now"], 1), 3),
                          "KB_per_tile_after": round(after / max(tiles, 1) / 1e3, 1),
                          "new_rules_stores": new_rules})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 30)
    print("Per field (all sampled stores of each product):")
    print(pd.DataFrame(field_rows).to_string(index=False))
    print("\nPer product, MB per granule (negative = saved by that rule):")
    print(pd.DataFrame(prod_rows).to_string(index=False))


if __name__ == "__main__":
    main()
