"""Which evaluation samples have usable satellite imagery, per sensor family (step 2.4b).

For every point in config/eval_points.csv (eDNA, SFER, CRCP fish) and, if present, the gitignored
config/eval_points_private.csv (video MaxN; see scripts/make_eval_points.py),
find its 1.2 km cell and count the tile observations of each product family that have a valid
pixel in that cell or its 3 x 3 neighbourhood:
  - `causal`: within the CONTEXT_DAYS before the sample (the causal sequence window), and
  - `sameday`: within +/- 1 day of the sample.
"Usable" = the cell has valid pixel weight under the training mask (oefm/qcpolicy.py), i.e.
actual pixel validity, not acquisition dates. Datasets with `time_precision: month` in
training.partitions.labels (video MaxN) use their calendar month as the window instead, and have
no same-day count. Per dataset and year the summary reports samples with >= 1 and >= 2 usable
PACE observations, samples whose whole causal window lies in the PACE era (date-based), and those
that also have >= 1 usable PACE observation; plus the same >= 1 counts for the other families.
Writes tiles/<version>/index/eval_coverage.csv (one row per point) and eval_coverage_summary.csv
(per dataset x year x family), which answer "how many evaluation samples have usable PACE".
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, Config
from .grid import TileGrid
from .tiling import tiles_root

CONTEXT_DAYS = 30
FAMILIES = ["PACE", "OLCI", "MODIS", "VIIRS"]


def run(cfg: Config, points_csv: Path | None = None, log=print) -> Path | None:
    import pandas as pd
    import zarr

    points_csv = points_csv or REPO_ROOT / "config" / "eval_points.csv"
    idx_path = tiles_root(cfg) / "index" / "tile_observations.parquet"
    if not idx_path.exists():
        log("No tile index yet: run `python -m oefm tiles` first.")
        return None
    files = [points_csv] + [p for p in [points_csv.with_name(points_csv.stem + "_private.csv")] if p.exists()]
    pts = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    log(f"Evaluation points: {len(pts):,} from {', '.join(f.name for f in files)}")
    pts["t"] = pd.to_datetime(pts["time_utc"], utc=True)
    grid = TileGrid(cfg)
    x, y = grid.proj.forward(pts["lon"].to_numpy(), pts["lat"].to_numpy())
    slot = grid.slots(x, y)
    n = grid.tc
    pts["tile_id"] = [grid.ids[s // (n * n)] if s >= 0 else None for s in slot]
    pts["row"] = np.where(slot >= 0, (slot % (n * n)) // n, -1)
    pts["col"] = np.where(slot >= 0, slot % n, -1)
    idx = pd.read_parquet(idx_path, columns=["product", "family", "output_path", "row", "tile_id", "time_utc"])
    idx = idx[~idx["product"].str.endswith("sfrefl")]  # SFREFL is a separate optional view
    idx["t"] = pd.to_datetime(idx["time_utc"], utc=True, format="ISO8601")
    by_tile = {k: g for k, g in idx.groupby("tile_id")}
    cache: dict[str, object] = {}

    def wvalid(path: str, row: int) -> np.ndarray:
        if path not in cache:
            if len(cache) > 256:
                cache.clear()
            cache[path] = zarr.open_group(zarr.ZipStore(str(cfg.data_root / path), mode="r"), mode="r")["w_valid"]
        return cache[path][row]

    out = []
    lo, hi = pd.Timedelta(days=CONTEXT_DAYS), pd.Timedelta(days=1)
    labels = cfg.raw["training"]["partitions"]["labels"]
    pace0 = pd.Timestamp(str(cfg.raw["training"]["partitions"]["pace_start"]), tz="UTC")
    for p in pts.itertuples():
        monthly = labels.get(p.dataset, {}).get("time_precision") == "month"
        if monthly:
            w0 = p.t.normalize().replace(day=1)
            w1 = w0 + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
        else:
            w0, w1 = p.t - lo, p.t
        rec = {"dataset": p.dataset, "sample_id": p.sample_id, "time_utc": p.time_utc, "tile_id": p.tile_id,
               "window": "calendar month" if monthly else f"{CONTEXT_DAYS} d causal",
               "full_window_in_pace_era": bool(w0 >= pace0)}
        counts = defaultdict(int)
        g = by_tile.get(p.tile_id)
        if g is not None:
            g = g[(g["t"] >= min(w0, p.t - hi)) & (g["t"] <= max(w1, p.t + hi))]
            for o in g.itertuples():
                w = wvalid(o.output_path, int(o.row))
                r0, c0 = max(0, p.row - 1), max(0, p.col - 1)
                if not (w[r0:p.row + 2, c0:p.col + 2] > 0).any():
                    continue
                if w0 <= o.t <= w1:
                    counts[f"causal_{o.family}"] += 1
                if not monthly and abs((o.t - p.t).total_seconds()) <= 86400:
                    counts[f"sameday_{o.family}"] += 1
        for fam in FAMILIES:
            rec[f"causal_{fam}"] = counts[f"causal_{fam}"]
            rec[f"sameday_{fam}"] = counts[f"sameday_{fam}"]
        out.append(rec)
    df = pd.DataFrame(out)
    df["year"] = pd.to_datetime(df["time_utc"], utc=True).dt.year
    dest = idx_path.parent / "eval_coverage.csv"
    df.to_csv(dest, index=False)
    rows = []
    for (ds, yr), g in df.groupby(["dataset", "year"]):
        r = {"dataset": ds, "year": yr, "samples": len(g),
             "PACE_ge1": int((g["causal_PACE"] >= 1).sum()), "PACE_ge2": int((g["causal_PACE"] >= 2).sum()),
             "PACE_full_window_era": int(g["full_window_in_pace_era"].sum()),
             "PACE_full_window_era_and_ge1": int((g["full_window_in_pace_era"] & (g["causal_PACE"] >= 1)).sum()),
             "PACE_sameday": int((g["sameday_PACE"] >= 1).sum())}
        for fam in FAMILIES:
            r[f"usable_{fam}_causal30d"] = int((g[f"causal_{fam}"] > 0).sum())
            r[f"usable_{fam}_sameday"] = int((g[f"sameday_{fam}"] > 0).sum())
            r[f"median_{fam}_obs_causal30d"] = float(g[f"causal_{fam}"].median())
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(idx_path.parent / "eval_coverage_summary.csv", index=False)
    cols = ["dataset", "year", "samples", "PACE_ge1", "PACE_ge2", "PACE_full_window_era", "PACE_full_window_era_and_ge1",
            "PACE_sameday", "usable_VIIRS_causal30d", "usable_MODIS_causal30d", "usable_OLCI_causal30d"]
    log(summ[cols].to_string(index=False))
    log(f"Evaluation coverage -> {dest}")
    return dest
