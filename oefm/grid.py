"""The fixed tile grid (step 2.3): a local equal-area projection, 38.4 km tiles, 1.2 km and 300 m cells.

Projection: spherical Lambert azimuthal equal-area (LAEA) on the authalic sphere, centred on the
pilot domain (config `processing.grid`). Equal-area on the sphere; against the WGS84 ellipsoid
the area error over the domain is well below 1%. Written out here so nothing beyond numpy is
needed; the equivalent PROJ string is in the config.

Tile (ix, iy) covers x in [ix, ix + 1) * tile_m and y in [iy, iy + 1) * tile_m, so a tile's ID
depends only on the projection and the tile size, never on which tiles are selected or on the
order they were built. Cells are numbered row-major from the tile's south-west corner:
cell (row j, column i) has centre x = (ix * T + i + 0.5) * cell_m, y = (iy * T + j + 0.5) * cell_m.
Row 0 is the southern edge (arrays are stored south-up; flip for north-up display).

Global configs have several zones (config `zones`), each its own LAEA centre; `TileGrid(cfg, zone)`
is one zone's tile set and its tile IDs carry the zone as a prefix (`<zone>:X+03Y-12`), so IDs stay
unique across zones. `grids(cfg)` returns every zone's grid (one unnamed grid for regional configs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config, Region

R_AUTHALIC_M = 6371007.181


@dataclass(frozen=True)
class Laea:
    lat_0: float
    lon_0: float
    r: float = R_AUTHALIC_M

    def forward(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        lam = np.radians(np.asarray(lon, dtype="f8") - self.lon_0)
        phi = np.radians(np.asarray(lat, dtype="f8"))
        p0 = np.radians(self.lat_0)
        cosc = np.sin(p0) * np.sin(phi) + np.cos(p0) * np.cos(phi) * np.cos(lam)
        k = np.sqrt(2.0 / np.clip(1.0 + cosc, 1e-12, None))
        x = self.r * k * np.cos(phi) * np.sin(lam)
        y = self.r * k * (np.cos(p0) * np.sin(phi) - np.sin(p0) * np.cos(phi) * np.cos(lam))
        return x, y

    def inverse(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, dtype="f8"), np.asarray(y, dtype="f8")
        p0 = np.radians(self.lat_0)
        rho = np.hypot(x, y)
        c = 2.0 * np.arcsin(np.clip(rho / (2.0 * self.r), -1, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.arcsin(np.clip(np.cos(c) * np.sin(p0) + np.where(rho > 0, y * np.sin(c) * np.cos(p0) / rho, 0), -1, 1))
        lam = np.arctan2(x * np.sin(c), rho * np.cos(p0) * np.cos(c) - y * np.sin(p0) * np.sin(c))
        lon = np.degrees(lam) + self.lon_0
        return (lon + 180.0) % 360.0 - 180.0, np.degrees(phi)   # [-180, 180) across the antimeridian

    def proj4(self) -> str:
        return f"+proj=laea +lat_0={self.lat_0} +lon_0={self.lon_0} +R={self.r} +units=m +no_defs"


def tile_id(ix: int, iy: int, zone: str | None = None) -> str:
    return (f"{zone}:" if zone else "") + f"X{ix:+03d}Y{iy:+03d}"


def grids(cfg: Config) -> list["TileGrid"]:
    """Every zone's grid (global configs), or the one regional grid."""
    return [TileGrid(cfg, z) for z in cfg.zones] if cfg.zones else [TileGrid(cfg)]


class TileGrid:
    """The pilot tile set and fast lookups from projected coordinates to (tile, cell)."""

    def __init__(self, cfg: Config, zone: str | None = None) -> None:
        g = cfg.raw["processing"]["grid"]
        if cfg.zones and zone is None:
            raise ValueError("this config has zones: use TileGrid(cfg, zone) or grids(cfg)")
        if zone is not None and zone not in cfg.zones:
            raise ValueError(f"unknown zone {zone!r}")
        self.zone = zone
        c = cfg.zones[zone] if zone else g
        self.proj = Laea(float(c["lat_0"]), float(c["lon_0"]))
        self.cell_m = float(g["cell_m"])
        self.tc = int(g["tile_cells"])
        self.tile_m = self.cell_m * self.tc
        self.olci_cell_m = float(g["olci_cell_m"])
        self.olci_tc = int(round(self.tile_m / self.olci_cell_m))
        if abs(self.olci_tc * self.olci_cell_m - self.tile_m) > 1e-6:
            raise ValueError("olci_cell_m must divide the tile size")
        self.regions = [r for r in cfg.regions.values() if zone is None or r.zone == zone]
        if not self.regions:
            raise ValueError(f"zone {zone!r} has no regions")
        self._select(cfg.domain(zone))

    # ------------------------------------------------------------------ selection
    def _select(self, dom: Region) -> None:
        # Tile index range covering the domain (sample its edges densely; the projection bends lines).
        t = np.linspace(0, 1, 200)
        lons = np.r_[dom.west + t * (dom.east - dom.west), np.full(200, dom.east),
                     dom.east - t * (dom.east - dom.west), np.full(200, dom.west)]
        lats = np.r_[np.full(200, dom.south), dom.south + t * (dom.north - dom.south),
                     np.full(200, dom.north), dom.north - t * (dom.north - dom.south)]
        x, y = self.proj.forward(lons, lats)
        ix0, ix1 = int(np.floor(x.min() / self.tile_m)), int(np.floor(x.max() / self.tile_m))
        iy0, iy1 = int(np.floor(y.min() / self.tile_m)), int(np.floor(y.max() / self.tile_m))
        keep = []
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                lon, lat = self.cell_centres(ix, iy)
                inside = np.zeros(lon.shape, bool)
                for r in self.regions:
                    inside |= r.contains(lon, lat)
                if inside.any():
                    keep.append((ix, iy, [r.key for r in self.regions if r.contains(lon, lat).any()]))
        self.ix = np.array([k[0] for k in keep], dtype="i4")
        self.iy = np.array([k[1] for k in keep], dtype="i4")
        self.tile_regions = [k[2] for k in keep]
        self.ids = [tile_id(a, b, self.zone) for a, b in zip(self.ix, self.iy)]
        self.ix0, self.iy0 = int(self.ix.min()), int(self.iy.min())
        self.lut = np.full((int(self.iy.max()) - self.iy0 + 1, int(self.ix.max()) - self.ix0 + 1), -1, dtype="i4")
        self.lut[self.iy - self.iy0, self.ix - self.ix0] = np.arange(len(self.ids), dtype="i4")

    @property
    def n_tiles(self) -> int:
        return len(self.ids)

    def cell_centres(self, ix: int, iy: int, cell_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(lon, lat) of the cell centres of tile (ix, iy), shape (n, n), row 0 = south."""
        cm = cell_m or self.cell_m
        n = int(round(self.tile_m / cm))
        c = (np.arange(n) + 0.5) * cm
        xx, yy = np.meshgrid(ix * self.tile_m + c, iy * self.tile_m + c)
        return self.proj.inverse(xx, yy)

    def lonlat_box(self, pad_deg: float = 0.05) -> Region:
        """A lon/lat box containing every selected tile (for cutting swaths before projecting). A zone
        across the antimeridian gets the full longitude range (correct, only less tight)."""
        lons, lats = [], []
        for ix, iy in zip(self.ix, self.iy):
            for cx in (0, 1):
                for cy in (0, 1):
                    lo, la = self.proj.inverse((ix + cx) * self.tile_m, (iy + cy) * self.tile_m)
                    lons.append(float(lo))
                    lats.append(float(la))
        w, e = min(lons) - pad_deg, max(lons) + pad_deg
        if e - w > 180:   # wrapped: the tiles straddle +/-180
            w, e = -180.0, 180.0
        return Region("tiles", "tile set", w, min(lats) - pad_deg, e, max(lats) + pad_deg)

    def slots(self, x: np.ndarray, y: np.ndarray, cell_m: float | None = None) -> np.ndarray:
        """Flat slot = tile_number * n * n + row * n + col for projected points; -1 outside the tile set."""
        cm = cell_m or self.cell_m
        n = int(round(self.tile_m / cm))
        with np.errstate(invalid="ignore"):
            cx = np.floor(x / cm)
            cy = np.floor(y / cm)
        ok = np.isfinite(cx) & np.isfinite(cy)
        cx = np.where(ok, cx, 0).astype("i8")
        cy = np.where(ok, cy, 0).astype("i8")
        tx, ty = np.floor_divide(cx, n), np.floor_divide(cy, n)
        li, lj = tx - self.ix0, ty - self.iy0
        inb = ok & (li >= 0) & (lj >= 0) & (lj < self.lut.shape[0]) & (li < self.lut.shape[1])
        tile = np.full(x.shape, -1, dtype="i8")
        tile[inb] = self.lut[lj[inb], li[inb]]
        out = tile * n * n + (cy - ty * n) * n + (cx - tx * n)
        out[tile < 0] = -1
        return out

    def table(self) -> "pd.DataFrame":  # noqa: F821
        import pandas as pd

        rows = []
        for k, (ix, iy, tid) in enumerate(zip(self.ix, self.iy, self.ids)):
            lon, lat = self.proj.inverse((ix + 0.5) * self.tile_m, (iy + 0.5) * self.tile_m)
            rows.append({"tile": k, "tile_id": tid, "zone": self.zone or "", "ix": int(ix), "iy": int(iy),
                         "x0_m": float(ix * self.tile_m), "y0_m": float(iy * self.tile_m),
                         "lon_centre": round(float(lon), 5), "lat_centre": round(float(lat), 5),
                         "regions": " ".join(self.tile_regions[k])})
        return pd.DataFrame(rows)

    def spec(self) -> dict:
        return {"projection": self.proj.proj4(), "zone": self.zone, "cell_m": self.cell_m, "tile_cells": self.tc,
                "tile_m": self.tile_m, "olci_cell_m": self.olci_cell_m, "olci_tile_cells": self.olci_tc,
                "n_tiles": self.n_tiles, "cell_order": "row-major from the south-west corner (row 0 = south)"}


def write_static(cfg: Config, grid: "TileGrid | list[TileGrid]", out_dir: Path, log=print) -> dict[str, Path]:
    """Tile table (CSV + parquet) and static per-cell fields: lon/lat centres, GEBCO depth, land.
    Several grids (zones): one table and one store, tiles in zone order."""
    import json

    import pandas as pd
    import zarr

    from .inspect import gebco_depth

    gl = grid if isinstance(grid, list) else [grid]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = gl[0].tc
    tabs, lons, lats, depths = [], [], [], []
    have_depth = True
    for gr in gl:
        t = gr.table()
        depth_fn = gebco_depth(cfg, cfg.domain(gr.zone) if gr.zone else None)
        lon = np.zeros((gr.n_tiles, n, n), "f4")
        lat = np.zeros_like(lon)
        depth = np.full_like(lon, np.nan)
        for k, (ix, iy) in enumerate(zip(gr.ix, gr.iy)):
            lo, la = gr.cell_centres(int(ix), int(iy))
            lon[k], lat[k] = lo, la
            if depth_fn is not None:
                depth[k] = depth_fn(lo.ravel(), la.ravel()).reshape(n, n)
        if depth_fn is not None:
            t["ocean_cell_fraction"] = np.round((depth > 0).reshape(gr.n_tiles, -1).mean(axis=1), 4)
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    t["median_depth_m"] = np.round(np.nanmedian(np.where(depth > 0, depth, np.nan).reshape(gr.n_tiles, -1), axis=1), 1)
        else:
            have_depth = False
        tabs.append(t)
        lons.append(lon)
        lats.append(lat)
        depths.append(depth)
    tab = pd.concat(tabs, ignore_index=True)
    tab["tile"] = np.arange(len(tab))
    lon, lat, depth = np.concatenate(lons), np.concatenate(lats), np.concatenate(depths)
    ids = [i for gr in gl for i in gr.ids]
    if not have_depth:
        log("GEBCO not found: static depth left empty (re-run `grid` after GEBCO is downloaded)")
    tab.to_csv(out_dir / "tiles.csv", index=False)
    tab.to_parquet(out_dir / "tiles.parquet", index=False)
    store = zarr.ZipStore(str(out_dir / "static.zarr.zip.tmp"), mode="w")
    root = zarr.group(store=store)
    for name, arr in (("lon", lon), ("lat", lat), ("depth_m", depth)):
        root.create_dataset(name, data=arr, chunks=(1, n, n))
    spec = gl[0].spec() if len(gl) == 1 else {"zones": [g.spec() for g in gl], "n_tiles": len(ids)}
    root.attrs.update({"tile_ids": ids, **spec, "depth_source": "GEBCO_2026 nearest cell, positive down"})
    store.close()
    (out_dir / "static.zarr.zip.tmp").replace(out_dir / "static.zarr.zip")
    (out_dir / "grid.json").write_text(json.dumps(spec, indent=1))
    log(f"Tile grid: {len(ids)} tiles of {gl[0].tile_m / 1000:.1f} km in {len(gl)} zone(s) -> {out_dir}")
    return {"tiles": out_dir / "tiles.parquet", "static": out_dir / "static.zarr.zip"}
