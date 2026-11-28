"""Observation exclusion windows (`processing.exclude_windows` in the config).

A satellite observation is skipped, for training, evaluation embeddings and evaluation matchup
counts alike, when its time falls in a window and its location (tile centre) is inside the window's
lon / lat box (no box: everywhere). First use: the total solar eclipse of 2024-04-08, whose partial
phase darkened the Americas from mid-afternoon to ~21 UTC (PACE SFREFL had 3 % of values below the
valid range). In-situ samples are never excluded, only the imagery.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def windows(cfg) -> list[dict[str, Any]]:
    out = []
    for w in (cfg.raw.get("processing") or {}).get("exclude_windows") or []:
        t0 = pd.Timestamp(w["start"]).tz_convert("UTC") if pd.Timestamp(w["start"]).tzinfo else pd.Timestamp(w["start"], tz="UTC")
        t1 = pd.Timestamp(w["end"]).tz_convert("UTC") if pd.Timestamp(w["end"]).tzinfo else pd.Timestamp(w["end"], tz="UTC")
        out.append({"t0": t0.timestamp(), "t1": t1.timestamp(), "lon": w.get("lon"), "lat": w.get("lat"),
                    "reason": w.get("reason", "")})
    return out


def excluded(cfg, t_unix, lon=None, lat=None) -> np.ndarray:
    """Boolean mask: observation at t_unix (s since the epoch, UTC) and (lon, lat) is excluded.
    lon / lat None: location unknown, so only the time is tested (conservative)."""
    t = np.asarray(t_unix, dtype="f8")
    out = np.zeros(t.shape, bool)
    for w in windows(cfg):
        m = (t >= w["t0"]) & (t <= w["t1"])
        if lon is not None and w["lon"] is not None:
            lo = np.asarray(lon, dtype="f8")
            m &= (lo >= float(w["lon"][0])) & (lo <= float(w["lon"][1]))
        if lat is not None and w["lat"] is not None:
            la = np.asarray(lat, dtype="f8")
            m &= (la >= float(w["lat"][0])) & (la <= float(w["lat"][1]))
        out |= m
    return out


def tile_centres(grids) -> dict[str, tuple[float, float]]:
    """{tile_id: (lon, lat)} of the tile centres of one or more TileGrids."""
    out: dict[str, tuple[float, float]] = {}
    for g in grids if isinstance(grids, (list, tuple)) else [grids]:
        x = (np.asarray(g.ix, dtype="f8") + 0.5) * g.tile_m
        y = (np.asarray(g.iy, dtype="f8") + 0.5) * g.tile_m
        lon, lat = g.proj.inverse(x, y)
        out.update({tid: (float(a), float(b)) for tid, a, b in zip(g.ids, np.ravel(lon), np.ravel(lat))})
    return out


def drop_excluded(cfg, df: pd.DataFrame, grids, t_col: str = "t") -> pd.DataFrame:
    """Rows of a tile-observation frame (tile_id, t in s) outside every exclusion window."""
    if not windows(cfg) or df.empty:
        return df
    cen = tile_centres(grids)
    ll = df["tile_id"].map(cen)
    lon = np.array([v[0] if isinstance(v, tuple) else np.nan for v in ll])
    lat = np.array([v[1] if isinstance(v, tuple) else np.nan for v in ll])
    unknown = np.isnan(lon)
    m = excluded(cfg, df[t_col].to_numpy(), lon, lat) | (unknown & excluded(cfg, df[t_col].to_numpy()))
    return df[~m]
