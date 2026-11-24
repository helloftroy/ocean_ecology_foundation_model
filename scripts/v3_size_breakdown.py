"""Where do the bytes of the v3 tile stores go? (read-only; seconds, run on a login node)

Per product, from the zip directories and attributes only (nothing is decompressed):
  which storage rules the stores were written with (new size rules: flag_share_bits in attrs; SFREFL
  empty flags), granules, stores, tiles, MB per granule, KB per tile; and per field: stored MB, share of
  the product's bytes, zip members. Printed product by product.

--predict K: also estimate, from K sampled tiles of up to 2 stores per product written with the OLD
rules, the size under the size rules of 2026-11-24 (extras list, flag-share bits, no sun angles,
zstd 9 with the better shuffle). Slower (decompresses and recompresses the sample).

    python scripts/v3_size_breakdown.py [--root data/processed/global/tiles/v3] [--predict 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def listing(store: Path) -> tuple[dict[str, list[int]], dict]:
    """{field: [bytes, members]}, attrs, read from the zip directory and .zattrs only."""
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with zipfile.ZipFile(store) as zf:
        for info in zf.infolist():
            name = info.filename.split("/")[0]
            key = "(metadata)" if name.startswith(".") or info.filename.endswith((".zarray", ".zattrs", ".zgroup")) else name
            per[key][0] += info.compress_size
            per[key][1] += 1
        attrs = json.loads(zf.read(".zattrs")) if ".zattrs" in zf.namelist() else {}
    return per, attrs


def predict(store: Path, k: int, keep_extras, share_bits) -> tuple[int, int]:
    """(old-rule bytes, new-rule bytes) of the first k tiles of an old-rule store."""
    import zarr
    from numcodecs import Blosc

    from oefm.tiling3 import store_chunks, store_codec

    old = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    st = zarr.ZipStore(str(store), mode="r")
    g = zarr.open_group(st, mode="r")
    attrs = g.attrs.asdict()
    a = {name: g[name][:k] if g[name].ndim > 1 else g[name][...] for name in g.array_keys()}
    st.close()
    nb_old = sum(len(old.encode(np.ascontiguousarray(v[i:i + 1]))) for v in a.values() if v.ndim > 1
                 for i in range(v.shape[0]))
    masks = dict(zip(attrs["flag_meanings"], attrs["flag_masks"]))
    rows = [int(masks[b] & 0xFFFFFFFF).bit_length() - 1 for b in share_bits if masks.get(b)]
    new = {n: v for n, v in a.items() if v.ndim > 1 and n not in ("solz", "sola")
           and not (n.startswith("x_") and n[2:] not in keep_extras)}
    new["flag_share"] = a["flag_share"][:, rows]
    nb_new = 0
    for v in new.values():
        ch = store_chunks(v)
        c = store_codec(v, ch)
        nb_new += sum(len(c.encode(np.ascontiguousarray(v[i:i + ch[0]]))) for i in range(0, v.shape[0], ch[0]))
    return nb_old, nb_new


def main(argv=None) -> None:
    from oefm.config import load_config
    from oefm.tiling3 import store_rules, v3_root

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--predict", type=int, default=0, help="tiles per sampled store for the new-rule estimate (0 = off)")
    args = ap.parse_args(argv)
    cfg = load_config()
    root = Path(args.root) if args.root else v3_root(cfg)
    summary = []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "index"):
        stores = sorted(pdir.rglob("*.zarr.zip"))
        if not stores:
            continue
        per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        granules, tiles, rules = set(), 0, defaultdict(int)
        old_stores = []
        for s in stores:
            p, attrs = listing(s)
            for k, (b, m) in p.items():
                per[k][0] += b
                per[k][1] += m
            granules.add(s.name.split("__")[0])
            tiles += len(attrs.get("tile_ids", []))
            new = "flag_share_bits" in attrs
            rules[("new size rules" if new else "old rules")
                  + (f", SFREFL empty {'+'.join(attrs['sfrefl_empty_flags'])}" if attrs.get("sfrefl_empty_flags") else "")] += 1
            if not new:
                old_stores.append(s)
        total = sum(v[0] for v in per.values())
        ng = max(len(granules), 1)
        print(f"\n== {pdir.name}: {ng} granules, {len(stores)} stores, {tiles:,} tiles, {total / 1e6:,.1f} MB "
              f"({total / ng / 1e6:,.1f} MB per granule, {total / max(tiles, 1) / 1e3:,.1f} KB per tile) | "
              + "; ".join(f"{k}: {v} stores" for k, v in rules.items()), flush=True)
        print(f"   {'field':14s} {'MB':>9s} {'share':>6s} {'members':>8s}")
        for k, (b, m) in sorted(per.items(), key=lambda kv: -kv[1][0]):
            print(f"   {k:14s} {b / 1e6:9.1f} {b / total:6.3f} {m:8,d}", flush=True)
        row = {"product": pdir.name, "MB_per_granule": round(total / ng / 1e6, 1),
               "KB_per_tile": round(total / max(tiles, 1) / 1e3, 1)}
        if args.predict and old_stores:
            ex, bits = store_rules(cfg, pdir.name)
            o = n = 0
            for s in old_stores[:: max(1, len(old_stores) // 2)][:2]:
                a, b = predict(s, args.predict, ex, bits)
                o, n = o + a, n + b
            row["predicted_new_vs_old"] = round(n / max(o, 1), 3)
            print(f"   predicted size under the new rules: {row['predicted_new_vs_old']:.3f} x (sample)", flush=True)
        summary.append(row)
    print("\nSummary:")
    for r in summary:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
