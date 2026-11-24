"""Where do the bytes of the v3 tile stores go? (read-only; run on a login node after a tiles3 trial)

For a few stores per product (`<v3_root>/<product>/**/*.zarr.zip`):
  per array: stored (compressed) bytes, share of the store, raw bytes, compression ratio, zip members;
  per product: tiles, MB per tile, mean pixel-equivalents per covered cell (w_all < 1 means the
  sensor's pixels are larger than the 1.2 km / 300 m cell, so cells oversample the swath);
  and the size the same arrays would take with other storage settings, all lossless:
    current   zstd level 3, bitshuffle, one chunk per tile
    zstd9     zstd level 9
    chunk64   2-D / per-tile arrays chunked 64 tiles at a time (fewer, larger zip members)
    both      zstd9 + chunk64
    sparse    both, and flag_share kept only for bits set somewhere in the store (bit list in attrs)

    python scripts/v3_size_breakdown.py [--per-product 3] [--root data/processed/global/tiles/v3]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compressed_size(arrays: dict[str, np.ndarray], level: int, tiles_per_chunk: int) -> int:
    from numcodecs import Blosc

    comp = Blosc(cname="zstd", clevel=level, shuffle=Blosc.BITSHUFFLE)
    total = 0
    for a in arrays.values():
        if a.ndim <= 1:
            total += len(comp.encode(np.ascontiguousarray(a)))
            continue
        step = tiles_per_chunk if a.ndim == 3 else 1   # (tile, n, n) arrays get multi-tile chunks
        for i in range(0, a.shape[0], step):
            total += len(comp.encode(np.ascontiguousarray(a[i:i + step])))
    return total


def main(argv=None) -> None:
    import pandas as pd
    import zarr

    from oefm.config import load_config
    from oefm.tiling3 import v3_root

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-product", type=int, default=3)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else v3_root(load_config())
    arr_rows, prod_rows = [], []
    for pdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "index"):
        stores = sorted(pdir.rglob("*.zarr.zip"), key=lambda p: -p.stat().st_size)
        if not stores:
            continue
        pick = [stores[int(i)] for i in np.linspace(0, len(stores) - 1, min(args.per_product, len(stores)))]
        per_arr = defaultdict(lambda: [0, 0, 0])   # stored, raw, members
        alt = defaultdict(int)
        tiles, wsum, wn, file_bytes = 0, 0.0, 0, 0
        for s in pick:
            file_bytes += s.stat().st_size
            with zipfile.ZipFile(s) as zf:
                for info in zf.infolist():
                    name = info.filename.split("/")[0]
                    if name.startswith("."):
                        name = "(metadata)"
                    per_arr[name][0] += info.compress_size
                    per_arr[name][2] += 1
            g = zarr.open_group(zarr.ZipStore(str(s), mode="r"), mode="r")
            arrays = {k: g[k][...] for k in g.array_keys()}
            for k, a in arrays.items():
                per_arr[k][1] += a.nbytes
            tiles += int(arrays["tile_time"].shape[0])
            cov = arrays["w_all"] > 0
            wsum += float(arrays["w_all"][cov].sum())
            wn += int(cov.sum())
            alt["current"] += compressed_size(arrays, 3, 1)
            alt["zstd9"] += compressed_size(arrays, 9, 1)
            alt["chunk64"] += compressed_size(arrays, 3, 64)
            alt["both"] += compressed_size(arrays, 9, 64)
            fs = arrays["flag_share"]
            used = np.flatnonzero(fs.reshape(fs.shape[0], fs.shape[1], -1).any(axis=(0, 2)))
            sp = dict(arrays, flag_share=fs[:, used])
            alt["sparse"] += compressed_size(sp, 9, 64)
            g.store.close()
        total = sum(v[0] for v in per_arr.values())
        for k, (st, raw, mem) in sorted(per_arr.items(), key=lambda kv: -kv[1][0]):
            arr_rows.append({"product": pdir.name, "array": k, "MB": round(st / 1e6, 2),
                             "share": round(st / total, 3), "raw_MB": round(raw / 1e6, 2),
                             "ratio": round(raw / st, 1) if st else None, "members": mem})
        prod_rows.append({"product": pdir.name, "stores": len(pick), "tiles": tiles,
                          "MB_per_store": round(file_bytes / len(pick) / 1e6, 1),
                          "KB_per_tile": round(file_bytes / max(tiles, 1) / 1e3, 1),
                          "w_all_per_cell": round(wsum / max(wn, 1), 3),
                          **{f"{k}_vs_current": round(v / alt["current"], 3) for k, v in alt.items() if k != "current"}})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("Per array (stored bytes in the sampled stores):")
    print(pd.DataFrame(arr_rows).to_string(index=False))
    print("\nPer product (sizes relative to the current settings, same arrays, lossless):")
    print(pd.DataFrame(prod_rows).to_string(index=False))


if __name__ == "__main__":
    main()
