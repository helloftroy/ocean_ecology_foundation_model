"""Training sequences, partitions and masking (step 2.5).

Everything here is computed on the fly from the tile index, the per-granule tile stores and the
physics cube, using config `training`. Changing partitions, sensors, PACE visibility or masking
needs no rebuild.

Partitions
- Imagery (self-supervised pretraining and the PACE spectral-retention test): every observation
  gets a temporal label (train / val / gap / test, config `training.partitions.temporal`) and every
  tile a geographic label (train / geo_test / buffer). A split selects observations by both labels
  *before* windows are formed, so a sequence only ever contains observations of its own split and
  no held-out observation can enter training as context or target.
      train          tiles 'train', times 'train'      (2018 - 2025, incl. unlabeled 2024 PACE)
      val            tiles 'train', times 'val'        (Jan - Feb 2026)
      test_temporal  tiles 'train', times 'test'       (Apr - Oct 2026; March 2026 is unused)
      test_geo       tiles 'geo_test', times 'train'   (independent, geographically blocked)
  The 2024 biological hold-out applies to labels only (`label_split`), not to imagery.
- Labels (downstream probes): `label_split(dataset, time)` -> 'train' | 'test' from
  `training.partitions.labels`.

Samples (`TileSequences.__getitem__`, `sequence_for_point`) are plain dicts of numpy arrays:
    tile_id, anchor_time (UTC seconds), mode, cell (row, col) for point queries
    obs: list (time order) of dicts
        product, family, time (UTC s), dt_hours (relative to the anchor, <= 0 in causal mode),
        wavelength (B,), reflectance (B, 32, 32) float32 NaN where no valid pixel,
        valid (32, 32) bool, valid_weight (32, 32), strict (32, 32) bool, senz (32, 32),
        flags (32, 32) uint32, band_weight (B, 32, 32) red-band angle weights,
        hidden (bool, whole observation), patch_hidden (32, 32) bool, is_anchor, is_pace,
        [reflectance_300m (B, 128, 128), valid_300m] for OLCI when include_300m
    physics: values (D, V, K, K), day_offset (D,) (<= 0), available (D, V) bool, variables,
             hidden (bool)
    static: lon, lat, depth_m (32, 32); physics_static (S, K, K)
    targets (only with targets=True): pisces: values (3,) PO4/NO3/Si mmol m-3 on the anchor day,
             normalized (3,), coverage, source. Auxiliary TARGETS: never part of obs or physics.
Arrays are south-up (row 0 = south), as stored.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .grid import TileGrid
from .physics import physics_path
from .qcpolicy import red_angle_weight
from .tiling import tiles_root

SPLITS = ("train", "val", "test_temporal", "test_geo")
_DAY = 86400.0


def _ts(d) -> float:
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


# ----------------------------------------------------------------------------- partitions
def temporal_labels(cfg: Config, t_unix: np.ndarray) -> np.ndarray:
    """'train' | 'val' | 'gap' | 'test' | 'none' per UTC time (end dates inclusive)."""
    out = np.full(t_unix.shape, "none", dtype=object)
    for name, (a, b) in cfg.raw["training"]["partitions"]["temporal"].items():
        out[(t_unix >= _ts(a)) & (t_unix < _ts(b) + _DAY)] = name
    return out


def geographic_labels(cfg: Config, grid: TileGrid) -> dict[str, str]:
    """tile_id -> 'train' | 'geo_test' | 'buffer'. Blocks of block_tiles x block_tiles tiles; a
    seeded hash picks test_fraction of the blocks; tiles within buffer_tiles of a test tile (and
    not in a test block) are 'buffer'."""
    g = cfg.raw["training"]["partitions"]["geographic"]
    b, frac, buf, seed = int(g["block_tiles"]), float(g["test_fraction"]), int(g["buffer_tiles"]), int(g["seed"])
    blocks = sorted({(int(ix) // b, int(iy) // b) for ix, iy in zip(grid.ix, grid.iy)})

    def h(bk):
        return int(hashlib.sha256(f"{seed}:{bk[0]}:{bk[1]}".encode()).hexdigest()[:12], 16)

    ranked = sorted(blocks, key=h)
    test_blocks = set(ranked[: max(1, int(round(frac * len(blocks))))])
    lab = {}
    test_xy = [(int(ix), int(iy)) for ix, iy in zip(grid.ix, grid.iy) if (int(ix) // b, int(iy) // b) in test_blocks]
    for tid, ix, iy in zip(grid.ids, grid.ix, grid.iy):
        ix, iy = int(ix), int(iy)
        if (ix // b, iy // b) in test_blocks:
            lab[tid] = "geo_test"
        elif any(max(abs(ix - a), abs(iy - c)) <= buf for a, c in test_xy):
            lab[tid] = "buffer"
        else:
            lab[tid] = "train"
    return lab


def label_split(cfg: Config, dataset: str, time_utc) -> str:
    """Downstream biological labels: 'train' before test_from, 'test' from it on."""
    spec = cfg.raw["training"]["partitions"]["labels"][dataset]
    t = time_utc if isinstance(time_utc, (int, float)) else datetime.fromisoformat(
        str(time_utc).replace("Z", "+00:00")).timestamp()
    return "test" if t >= _ts(spec["test_from"]) else "train"


# ----------------------------------------------------------------------------- data access
class _Stores:
    """LRU cache of open per-granule tile stores (zipped Zarr)."""

    def __init__(self, root: Path, size: int = 256) -> None:
        self.root, self.size, self.c = root, size, OrderedDict()

    def get(self, rel: str):
        import zarr

        if rel in self.c:
            self.c.move_to_end(rel)
            return self.c[rel]
        z = zarr.open_group(zarr.ZipStore(str(self.root / rel), mode="r"), mode="r")
        self.c[rel] = z
        if len(self.c) > self.size:
            self.c.popitem(last=False)
        return z


class TileSequences:
    """Sequences of tile observations for one split.

    split: 'train' | 'val' | 'test_temporal' | 'test_geo' | 'all' (no partition filter; used for
           embedding extraction at evaluation points)
    mode: 'causal' (window [t - causal_days, t]) or 'retro' (t +/- retro_half_days)
    length: maximum observations per sequence (8 / 16 / 24); the anchor is always kept
    anchors: 'all' (every eligible observation) or 'pace' (PACE observations only: the spectral-
             retention set)
    sensors: products available to the encoder (default training.sensors.include)
    pace: 'visible' | 'hidden' (PACE observations returned with hidden=True) | 'drop' (removed)
    selection: 'recent' (deterministic: the anchor and the most recent / nearest others) or
               'random' (random subset; training augmentation)
    masking: True to apply training.masking, a dict to override it, None/False for no masking
    """

    def __init__(self, cfg: Config, split: str = "train", mode: str = "causal", length: int = 16,
                 anchors: str = "all", sensors: list[str] | None = None, pace: str = "visible",
                 selection: str = "recent", masking: bool | dict | None = None, include_300m: bool = False,
                 physics: bool = True, targets: bool = False, seed: int = 0, index=None) -> None:
        import pandas as pd

        if split not in SPLITS + ("all",):
            raise ValueError(f"split must be one of {SPLITS + ('all',)}")
        self.cfg, self.split, self.mode, self.length = cfg, split, mode, int(length)
        self.pace, self.selection, self.include_300m = pace, selection, include_300m
        tr = cfg.raw["training"]
        self.seq = tr["sequence"]
        self.mask_cfg = ({**tr["masking"], **masking} if isinstance(masking, dict) else tr["masking"]) if masking else None
        self.rng = np.random.default_rng(seed)
        self.grid = TileGrid(cfg)
        self.root = tiles_root(cfg)
        self.stores = _Stores(cfg.data_root)
        df = index if index is not None else pd.read_parquet(self.root / "index" / "tile_observations.parquet")
        self.sensors = list(sensors or tr["sensors"]["include"])
        df = df[df["product"].isin(self.sensors)]
        if pace == "drop":
            df = df[df["family"] != "PACE"]
        df = df[df["valid_fraction"] >= float(self.seq["min_valid_fraction"])].copy()
        df["t"] = pd.to_datetime(df["time_utc"], utc=True, format="ISO8601").astype("int64") / 1e9
        df["t_label"] = temporal_labels(cfg, df["t"].to_numpy())
        geo = geographic_labels(cfg, self.grid)
        df["g_label"] = df["tile_id"].map(geo).fillna("none")
        self.geo = geo
        if split != "all":
            want_g, want_t = {"train": ("train", "train"), "val": ("train", "val"),
                              "test_temporal": ("train", "test"), "test_geo": ("geo_test", "train")}[split]
            df = df[(df["g_label"] == want_g) & (df["t_label"] == want_t)]
        self.obs = df.sort_values(["tile_id", "t"]).reset_index(drop=True)
        self._by_tile = {k: g.index.to_numpy() for k, g in self.obs.groupby("tile_id", sort=False)}
        self._times = self.obs["t"].to_numpy()
        cand = self.obs if anchors == "all" else self.obs[self.obs["family"] == "PACE"]
        self.anchors = self._eligible(cand.index.to_numpy())
        self._phys = None
        self.physics_on = physics
        self.targets_on = targets
        self._tg = None

    # ------------------------------------------------------------------ windows
    def _window(self, rows: np.ndarray, t: float) -> np.ndarray:
        if self.mode == "causal":
            lo, hi = t - float(self.seq["causal_days"]) * _DAY, t
        else:
            h = float(self.seq["retro_half_days"]) * _DAY
            lo, hi = t - h, t + h
        times = self._times[rows]
        a, b = np.searchsorted(times, lo, "left"), np.searchsorted(times, hi, "right")
        return rows[a:b]

    def _eligible(self, cand: np.ndarray) -> np.ndarray:
        keep = []
        need = int(self.seq["min_observations"])
        tiles = self.obs["tile_id"].to_numpy()
        for i in cand:
            rows = self._by_tile[tiles[i]]
            if self._window(rows, self._times[i]).size >= need:
                keep.append(i)
        return np.asarray(keep, dtype=int)

    def __len__(self) -> int:
        return int(self.anchors.size)

    def _select(self, win: np.ndarray, anchor: int) -> np.ndarray:
        if win.size <= self.length:
            return win
        others = win[win != anchor]
        if self.selection == "random":
            pick = self.rng.choice(others, self.length - 1, replace=False)
        else:
            d = np.abs(self._times[others] - self._times[anchor])
            pick = others[np.argsort(d, kind="stable")[: self.length - 1]]
        return np.sort(np.r_[pick, anchor])

    # ------------------------------------------------------------------ reading
    def _read_obs(self, i: int, anchor_t: float, cell: tuple[int, int] | None = None) -> dict[str, Any]:
        r = self.obs.iloc[i]
        z = self.stores.get(r["output_path"])
        j = int(r["row"])
        wl = np.asarray(z["wavelength"][:], dtype="f4")
        refl = np.asarray(z["reflectance"][j])
        wv = np.asarray(z["w_valid"][j])
        senz = np.asarray(z["senz"][j])
        o = {"product": r["product"], "family": r["family"], "time": float(r["t"]),
             "dt_hours": (float(r["t"]) - anchor_t) / 3600.0, "wavelength": wl, "reflectance": refl,
             "valid": wv > 0, "valid_weight": wv, "strict": np.asarray(z["w_strict"][j]) > 0, "senz": senz,
             "flags": np.asarray(z["flags_or"][j]),
             "band_weight": red_angle_weight(self.cfg, r["product"], wl[:, None, None], senz[None, :, :]),
             "is_pace": r["family"] == "PACE", "hidden": False, "patch_hidden": np.zeros(senz.shape, bool),
             "granule_key": r["granule_key"], "output_path": r["output_path"],
             "flag_meanings": list(z.attrs["flag_meanings"]), "flag_masks": list(z.attrs["flag_masks"])}
        if self.include_300m and "reflectance_300m" in z:
            o["reflectance_300m"] = np.asarray(z["reflectance_300m"][j])
            o["valid_300m"] = np.asarray(z["w_valid_300m"][j]) > 0
        return o

    def _physics(self, tile_id: str, anchor_t: float) -> dict[str, Any] | None:
        if not self.physics_on:
            return None
        if self._phys is None:
            import zarr

            p = physics_path(self.cfg)
            if not p.exists():
                self.physics_on = False
                return None
            z = zarr.open_group(str(p), mode="r")
            self._phys = (z, date.fromisoformat(z.attrs["day0"]), {t: k for k, t in enumerate(z.attrs["tile_ids"])})
        z, day0, tix = self._phys
        n = int(self.seq["physics_lookback_days"])
        d_anchor = (datetime.fromtimestamp(anchor_t, timezone.utc).date() - day0).days
        d_from = d_anchor - n + 1
        a, b = max(0, d_from), min(z["values"].shape[0], d_anchor + 1)
        k = tix[tile_id]
        vals = np.full((n,) + z["values"].shape[2:], np.nan, dtype="f4")
        if b > a:
            vals[a - d_from: b - d_from] = z["values"][a:b, k]
        return {"values": vals, "day_offset": np.arange(-n + 1, 1), "available": np.isfinite(vals).any(axis=(2, 3)),
                "variables": [v["name"] for v in z.attrs["variables"]], "hidden": False,
                "static": np.asarray(z["static"][k])}

    def _targets(self, tile_id: str, anchor_t: float) -> dict[str, Any] | None:
        if not self.targets_on:
            return None
        if self._tg is None:
            import zarr

            from .targets import targets_path

            p = targets_path(self.cfg)
            if not p.exists():
                self.targets_on = False
                return None
            z = zarr.open_group(str(p), mode="r")
            self._tg = (z, date.fromisoformat(z.attrs["day0"]), {t: k for k, t in enumerate(z.attrs["tile_ids"])})
        z, day0, tix = self._tg
        d = (datetime.fromtimestamp(anchor_t, timezone.utc).date() - day0).days
        if not 0 <= d < z["values"].shape[0]:
            return None
        k = tix[tile_id]
        return {"pisces": {"values": np.asarray(z["values"][d, k]), "normalized": np.asarray(z["normalized"][d, k]),
                           "coverage": float(z["coverage"][d, k]), "source": int(z["source"][d]),
                           "variables": list(z.attrs["variables"])}}

    def _static(self, tile_id: str) -> dict[str, np.ndarray]:
        import zarr

        if not hasattr(self, "_st"):
            z = zarr.open_group(zarr.ZipStore(str(self.root / "grid" / "static.zarr.zip"), mode="r"), mode="r")
            self._st = (z, {t: k for k, t in enumerate(z.attrs["tile_ids"])})
        z, tix = self._st
        k = tix[tile_id]
        return {"lon": np.asarray(z["lon"][k]), "lat": np.asarray(z["lat"][k]), "depth_m": np.asarray(z["depth_m"][k])}

    def _assemble(self, tile_id: str, rows: np.ndarray, anchor_t: float, anchor_row: int | None) -> dict[str, Any]:
        obs = [self._read_obs(int(i), anchor_t) for i in rows]
        for i, o in zip(rows, obs):
            o["is_anchor"] = anchor_row is not None and int(i) == int(anchor_row)
            if self.pace == "hidden" and o["is_pace"]:
                o["hidden"] = True
        s = {"tile_id": tile_id, "anchor_time": anchor_t, "mode": self.mode, "split": self.split, "obs": obs,
             "physics": self._physics(tile_id, anchor_t), "static": self._static(tile_id),
             "targets": self._targets(tile_id, anchor_t)}
        if self.mask_cfg:
            apply_masking(s, self.mask_cfg, self.rng)
        return s

    def __getitem__(self, k: int) -> dict[str, Any]:
        i = int(self.anchors[k])
        tile = self.obs.at[i, "tile_id"]
        win = self._window(self._by_tile[tile], self._times[i])
        return self._assemble(tile, self._select(win, i), float(self._times[i]), i)

    # ------------------------------------------------------------------ evaluation points
    def sequence_for_point(self, lat: float, lon: float, time_utc: str | float) -> dict[str, Any] | None:
        """Causal sequence ending at a sample time for the tile containing (lat, lon); no anchor
        observation. Use split='all' so partitions do not filter imagery at evaluation points."""
        t = time_utc if isinstance(time_utc, (int, float)) else datetime.fromisoformat(
            str(time_utc).replace("Z", "+00:00")).timestamp()
        x, y = self.grid.proj.forward(np.array([lon]), np.array([lat]))
        s = int(self.grid.slots(x, y)[0])
        if s < 0:
            return None
        n = self.grid.tc
        tile = self.grid.ids[s // (n * n)]
        rows = self._by_tile.get(tile, np.zeros(0, int))
        win = rows[(self._times[rows] >= t - float(self.seq["causal_days"]) * _DAY) & (self._times[rows] <= t)]
        if win.size > self.length:
            win = win[np.argsort(t - self._times[win], kind="stable")[: self.length]]
        out = self._assemble(tile, np.sort(win), float(t), None)
        out["cell"] = ((s % (n * n)) // n, s % n)
        return out

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict[str, Any]:
        a = self.obs.loc[self.anchors]
        years = (a["t"] // (365.25 * _DAY) + 1970).astype(int)
        return {"split": self.split, "mode": self.mode, "tiles": int(a["tile_id"].nunique()),
                "anchors": int(len(a)), "pace_anchors": int((a["family"] == "PACE").sum()),
                "anchors_by_year": {int(k): int(v) for k, v in years.value_counts().sort_index().items()},
                "observations": int(len(self.obs)),
                "first": str(a["time_utc"].min()) if len(a) else None, "last": str(a["time_utc"].max()) if len(a) else None}


# ----------------------------------------------------------------------------- masking
def apply_masking(s: dict[str, Any], m: dict[str, Any], rng: np.random.Generator) -> None:
    """The five masking types (Objective 3), keeping >= min_visible_observations real observations
    visible. Sets obs['hidden'], obs['patch_hidden'] and physics['hidden'] in place."""
    obs = s["obs"]
    fams = sorted({o["family"] for o in obs})
    if len(fams) > 1 and rng.random() < m["drop_sensor_prob"]:
        f = fams[rng.integers(len(fams))]
        for o in obs:
            if o["family"] == f:
                o["hidden"] = True
    for o in obs:
        if rng.random() < m["drop_time_prob"]:
            o["hidden"] = True
    if any(o["is_pace"] for o in obs) and any(not o["is_pace"] for o in obs) and rng.random() < m["pace_mask_prob"]:
        for o in obs:
            if o["is_pace"]:
                o["hidden"] = True
    need = int(m["min_visible_observations"])
    hidden = [k for k, o in enumerate(obs) if o["hidden"]]
    rng.shuffle(hidden)
    while sum(not o["hidden"] for o in obs) < min(need, len(obs)) and hidden:
        obs[hidden.pop()]["hidden"] = False
    p = int(m["spatial_patch_cells"])
    for o in obs:
        if o["hidden"]:
            continue
        n = o["valid"].shape[0]
        g = n // p
        pick = rng.random((g, g)) < m["spatial_patch_ratio"]
        o["patch_hidden"] = np.kron(pick, np.ones((p, p), bool))[:n, :n]
    if s.get("physics") is not None and rng.random() < m["physics_mask_prob"]:
        s["physics"]["hidden"] = True
