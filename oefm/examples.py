"""Multisensor tile examples for the visual check (step 2.4).

Picks the tile-days seen by the most products within a few hours, and draws, for each, one panel
per observation at ~443 nm (shared colour scale), plus OLCI at native 300 m. Panels are bare (no
titles); `panels.csv` next to each figure lists what each panel shows, left to right.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .tiling import tiles_root


def _open(path: Path):
    import zarr

    return zarr.open_group(zarr.ZipStore(str(path), mode="r"), mode="r")


def pick(df, n: int, max_dt_h: float):
    import pandas as pd

    df = df[df["valid_fraction"] > 0.2].copy()
    df["t"] = pd.to_datetime(df["time_utc"], utc=True, format="ISO8601")
    df["day"] = df["t"].dt.strftime("%Y-%m-%d")
    best = []
    for (tile, day), g in df.groupby(["tile_id", "day"]):
        g = g.sort_values("t")
        if g["product"].nunique() < 2:
            continue
        span = (g["t"].max() - g["t"].min()).total_seconds() / 3600
        if span > max_dt_h:
            g = g[(g["t"] - g["t"].iloc[0]).dt.total_seconds() / 3600 <= max_dt_h]
        best.append((g["product"].nunique(), float(g["valid_fraction"].mean()), tile, day, g))
    best.sort(key=lambda b: (-b[0], -b[1]))
    out, used_days, used_tiles = [], set(), set()
    for b in best:  # first pass: one per day
        if b[3] not in used_days and len(out) < n:
            out.append(b)
            used_days.add(b[3])
            used_tiles.add(b[2])
    for b in best:  # then fill up with other tiles
        if len(out) >= n:
            break
        if b[2] not in used_tiles:
            out.append(b)
            used_tiles.add(b[2])
    return out


def draw(cfg: Config, n: int = 4, max_dt_h: float = 4.0, log=print) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8})
    root = tiles_root(cfg)
    idx = root / "index" / "tile_observations.parquet"
    if not idx.exists():
        log("No tile index yet: run `python -m oefm tiles` first.")
        return []
    df = pd.read_parquet(idx)
    out_dir = root / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for nprod, _, tile, day, g in pick(df, n, max_dt_h):
        panels = []
        for _, r in g.iterrows():
            z = _open(cfg.data_root / r["output_path"])
            wl = np.asarray(z["wavelength"][:])
            k = int(np.argmin(np.abs(wl - (443 if z.attrs["variable"] == "Rrs" else 865))))
            j = int(r["row"])
            panels.append((r, f"{r['product']} {wl[k]:.0f} nm 1.2 km", np.asarray(z["reflectance"][j, k]),
                           np.asarray(z["w_valid"][j])))
            if "reflectance_300m" in z:
                panels.append((r, f"{r['product']} {wl[k]:.0f} nm 300 m", np.asarray(z["reflectance_300m"][j, k]),
                               np.asarray(z["w_valid_300m"][j])))
        rrs = [p for p in panels if not p[0]["product"].endswith("sfrefl")]
        vals = np.concatenate([p[2][np.isfinite(p[2])].ravel() for p in rrs]) if rrs else np.array([0, 1])
        lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0, 1))
        fig, axes = plt.subplots(2, len(panels), figsize=(1.9 * len(panels), 3.9), squeeze=False)
        rows = []
        for c, (r, label, img, wv) in enumerate(panels):
            sf = r["product"].endswith("sfrefl")
            fin = img[np.isfinite(img)]
            vmin, vmax = (np.percentile(fin, [2, 98]) if sf and fin.size else (lo, hi))
            im = axes[0, c].imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
            axes[1, c].imshow(wv, origin="lower", cmap="Greys", vmin=0, interpolation="nearest")
            for a in axes[:, c]:
                a.set_xticks([])
                a.set_yticks([])
            if sf or c == len(panels) - 1:
                fig.colorbar(im, ax=axes[0, c], fraction=0.046, pad=0.02)
            rows.append({"column": c + 1, "panel": label, "time_utc": r["time_utc"],
                         "valid_fraction": r["valid_fraction"], "senz_mean": r["senz_mean"],
                         "top": "reflectance (south-up)", "bottom": "valid pixel-equivalents per cell"})
        fig.tight_layout()
        stem = out_dir / f"{day}_{tile}_{nprod}products"
        fig.savefig(f"{stem}.png", dpi=150)
        plt.close(fig)
        pd.DataFrame(rows).to_csv(f"{stem}_panels.csv", index=False)
        written.append(Path(f"{stem}.png"))
    log(f"examples: {len(written)} figures in {out_dir}")
    return written
