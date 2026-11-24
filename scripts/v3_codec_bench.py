"""Which zstd level for the v3 tile stores? (read-only; ~1-3 min on a login node)

For the largest store of each product (first --tiles tiles of every field), writes nothing and times
what tiling v3 does when it writes a store (store_chunks, store_codec: pick the better of byte / bit
shuffle on the first chunk, then compress every chunk), single-threaded, at zstd levels 1, 3, 5, 7, 9.
Prints per product and level: compressed MB, size relative to level 9, and seconds per granule
(scaled by the product's tiles per granule). Every level is lossless.

Also prints what chlor_a would take as int16 log10 (step 1e-4 in log10, i.e. 0.023 % relative; range
5e-4 .. 1,900 mg m-3) instead of float32, as a candidate for a further cut.

    python scripts/v3_codec_bench.py [--tiles 128]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
LEVELS = (1, 3, 5, 7, 9)


def write_cost(arrays: dict[str, np.ndarray], level: int) -> tuple[int, float]:
    from oefm.tiling3 import store_chunks, store_codec

    t0 = time.perf_counter()
    nb = 0
    for a in arrays.values():
        ch = store_chunks(a)
        c = store_codec(a, ch, level)
        if a.ndim <= 1:
            nb += len(c.encode(np.ascontiguousarray(a)))
        else:
            nb += sum(len(c.encode(np.ascontiguousarray(a[i:i + ch[0]]))) for i in range(0, a.shape[0], ch[0]))
    return nb, time.perf_counter() - t0


def main(argv=None) -> None:
    import zarr
    from numcodecs import Blosc, blosc

    from oefm.config import load_config
    from oefm.tiling3 import v3_root

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--tiles", type=int, default=128)
    args = ap.parse_args(argv)
    blosc.use_threads = False
    root = Path(args.root) if args.root else v3_root(load_config())
    print(f"{'product':16s} {'level':>5s} {'MB':>8s} {'vs_9':>6s} {'s_per_granule':>13s}", flush=True)
    for pdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "index"):
        stores = sorted(pdir.rglob("*.zarr.zip"), key=lambda p: -p.stat().st_size)
        if not stores:
            continue
        tiles_total = 0
        for s in pdir.rglob("*.zarr.zip"):
            with zipfile.ZipFile(s) as zf:
                tiles_total += len(json.loads(zf.read(".zattrs")).get("tile_ids", []))
        ngran = len({s.name.split("__")[0] for s in pdir.rglob("*.zarr.zip")})
        st = zarr.ZipStore(str(stores[0]), mode="r")
        g = zarr.open_group(st, mode="r")
        arrays = {k: (g[k][:args.tiles] if g[k].ndim > 1 else g[k][...]) for k in g.array_keys()}
        st.close()
        k = arrays["w_all"].shape[0]
        scale = tiles_total / ngran / max(k, 1)          # sample -> one granule
        res = {lv: write_cost(arrays, lv) for lv in LEVELS}
        for lv in LEVELS:
            nb, sec = res[lv]
            print(f"{pdir.name:16s} {lv:5d} {nb * scale / 1e6:8.1f} {nb / res[9][0]:6.3f} {sec * scale:13.1f}", flush=True)
        if "x_chlor_a" in arrays:
            v = arrays["x_chlor_a"].astype("f8")
            ok = np.isfinite(v) & (v > 0)
            q = np.full(v.shape, -32767, "i2")
            q[ok] = np.clip(np.round(np.log10(v[ok]) / 1e-4), -32766, 32767).astype("i2")
            c = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
            f32 = sum(len(c.encode(np.ascontiguousarray(arrays["x_chlor_a"][i:i + 64]))) for i in range(0, k, 64))
            i16 = sum(len(c.encode(np.ascontiguousarray(q[i:i + 64]))) for i in range(0, k, 64))
            print(f"{pdir.name:16s} chlor_a as int16 log10: {i16 / max(f32, 1):.2f} x its float32 size "
                  f"({f32 * scale / 1e6:.1f} -> {i16 * scale / 1e6:.1f} MB per granule)", flush=True)


if __name__ == "__main__":
    main()
