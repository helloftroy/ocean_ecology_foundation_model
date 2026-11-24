"""catalog_v1: the fixed global tile catalog (tiling v3). Tile IDs are permanent identifiers.

Zones: 15-degree latitude bands from 75 S to 75 N, each split into equal longitude sectors so a
zone is at most ~1,700 km wide at its equatorward edge (24 sectors at the equator, 12 at 60-75),
plus one polar cap per hemisphere poleward of 75 degrees. Every zone has its own spherical Lambert
azimuthal equal-area projection centred on the zone (authalic sphere, as oefm/grid.py). Linear scale
error is < 0.5 % within band zones and < 0.9 % at the cap rims (equal area everywhere).

Tiles: 38.4 km squares (32 x 32 cells of 1.2 km; 128 x 128 OLCI cells of 300 m) aligned to the
zone's projected origin: tile (ix, iy) covers x in [ix, ix + 1) * 38,400 m, y likewise. A tile belongs
to a zone when its footprint intersects the zone's lat/lon cell, so tiles crossing a zone border
belong to both zones (~4 % duplication, no gaps). Tile ID `<zone>:X+003Y-012`.

Points (samples, label matchups) always use their own zone: the zone whose lat/lon cell contains
the point, never a neighbouring zone's border tile. `point_tile(lat, lon)`.

The definition (`spec()`) is written next to the tiles (catalog.json); changing anything here means
a new catalog version and new tile IDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .grid import Laea

CATALOG_VERSION = "catalog_v1"
CELL_M = 1200.0
TILE_CELLS = 32
TILE_M = CELL_M * TILE_CELLS
OLCI_CELL_M = 300.0
OLCI_TILE_CELLS = 128
BAND_DEG = 15
CAP_LAT = 75.0
MAX_ZONE_KM = 1700.0
KM_PER_DEG = 111.195


@dataclass(frozen=True)
class Zone:
    key: str
    south: float
    north: float
    west: float
    east: float
    lat_0: float
    lon_0: float

    @property
    def proj(self) -> Laea:
        return _proj(self.lat_0, self.lon_0)

    @property
    def is_cap(self) -> bool:
        return self.south <= -90 or self.north >= 90

    def contains(self, lat, lon):
        lat, lon = np.asarray(lat, "f8"), np.asarray(lon, "f8")
        lon = (lon + 180.0) % 360.0 - 180.0
        inlat = (lat >= self.south) & ((lat < self.north) | (self.north >= 90))
        if self.is_cap:
            return inlat
        east_ok = (lon < self.east) | (self.east >= 180)
        return inlat & (lon >= self.west) & east_ok


@lru_cache(maxsize=None)
def _proj(lat_0: float, lon_0: float) -> Laea:
    return Laea(lat_0, lon_0)


def _sectors(south: float) -> int:
    eq = min(abs(south), abs(south + BAND_DEG))
    return int(math.ceil(360.0 * math.cos(math.radians(eq)) * KM_PER_DEG / MAX_ZONE_KM))


@lru_cache(maxsize=1)
def zones() -> tuple[Zone, ...]:
    out = [Zone("ZS", -90.0, -CAP_LAT, -180.0, 180.0, -90.0, 0.0)]
    for south in range(-int(CAP_LAT), int(CAP_LAT), BAND_DEG):
        n = _sectors(south)
        width = 360.0 / n
        for i in range(n):
            w = -180.0 + i * width
            out.append(Zone(f"Z{south:+03d}_{i:02d}", float(south), float(south + BAND_DEG), w, w + width,
                            south + BAND_DEG / 2.0, w + width / 2.0))
    out.append(Zone("ZN", CAP_LAT, 90.0, -180.0, 180.0, 90.0, 0.0))
    return tuple(out)


@lru_cache(maxsize=1)
def zone_map() -> dict[str, Zone]:
    return {z.key: z for z in zones()}


def zone_of(lat, lon) -> np.ndarray:
    """Zone key of each point (its own zone)."""
    lat = np.atleast_1d(np.asarray(lat, "f8"))
    lon = (np.atleast_1d(np.asarray(lon, "f8")) + 180.0) % 360.0 - 180.0
    out = np.full(lat.shape, "", dtype=object)
    out[lat >= CAP_LAT] = "ZN"
    out[lat < -CAP_LAT] = "ZS"
    band = np.floor(lat / BAND_DEG) * BAND_DEG
    mid = (lat >= -CAP_LAT) & (lat < CAP_LAT)
    for south in np.unique(band[mid]):
        m = mid & (band == south)
        n = _sectors(int(south))
        i = np.clip(np.floor((lon[m] + 180.0) / (360.0 / n)).astype(int), 0, n - 1)
        out[m] = [f"Z{int(south):+03d}_{k:02d}" for k in i]
    return out


def tile_id(zone: str, ix: int, iy: int) -> str:
    return f"{zone}:X{int(ix):+04d}Y{int(iy):+04d}"


def parse_tile_id(tid: str) -> tuple[str, int, int]:
    zone, rest = tid.split(":")
    x, y = rest[1:].split("Y")
    return zone, int(x), int(y)


def tile_lonlat(zone: Zone, ix, iy, n: int = TILE_CELLS) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre lon/lat of tile (ix, iy) at n x n cells (row 0 = south)."""
    cm = TILE_M / n
    c = (np.arange(n) + 0.5) * cm
    xx, yy = np.meshgrid(ix * TILE_M + c, iy * TILE_M + c)
    return zone.proj.inverse(xx, yy)


def is_member(zone: Zone, ix, iy) -> np.ndarray:
    """Whether tiles (ix, iy) intersect the zone's lat/lon cell (sampled on a 9 x 9 grid incl. edges)."""
    ix, iy = np.atleast_1d(ix), np.atleast_1d(iy)
    t = np.linspace(0, 1, 9)
    ox, oy = np.meshgrid(t, t)
    x = (ix[:, None] + ox.ravel()[None]) * TILE_M
    y = (iy[:, None] + oy.ravel()[None]) * TILE_M
    lon, lat = zone.proj.inverse(x, y)
    return zone.contains(lat, lon).any(axis=1)


def zones_near(lat, lon, margin_deg: float = 0.5) -> list[str]:
    """Zones whose cell (expanded by margin, ~one tile plus) contains any of the points: the zones a
    swath can contribute border tiles to."""
    lat, lon = np.asarray(lat, "f8").ravel(), np.asarray(lon, "f8").ravel()
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[ok], lon[ok]
    keys = set()
    for dlat in (-margin_deg, 0.0, margin_deg):
        la = np.clip(lat + dlat, -90, 90)
        for dlon in (-1.0, 0.0, 1.0):
            sc = margin_deg / np.maximum(np.cos(np.radians(la)), 0.05)
            keys |= set(zone_of(la, lon + dlon * sc).tolist())
    keys.discard("")
    return sorted(keys)


def point_tile(lat: float, lon: float, n: int = TILE_CELLS) -> tuple[str, int, int]:
    """(tile ID in the point's own zone, row, col) of the n x n cell containing the point."""
    z = zone_map()[zone_of(lat, lon)[0]]
    x, y = z.proj.forward(np.array([lon]), np.array([lat]))
    cm = TILE_M / n
    cx, cy = int(np.floor(x[0] / cm)), int(np.floor(y[0] / cm))
    ix, iy = cx // n, cy // n
    return tile_id(z.key, ix, iy), cy - iy * n, cx - ix * n


def spec() -> dict:
    return {"version": CATALOG_VERSION, "projection": "spherical LAEA per zone, R = 6371007.181 m (authalic)",
            "cell_m": CELL_M, "tile_cells": TILE_CELLS, "tile_m": TILE_M, "olci_cell_m": OLCI_CELL_M,
            "olci_tile_cells": OLCI_TILE_CELLS, "band_deg": BAND_DEG, "cap_lat": CAP_LAT,
            "max_zone_km": MAX_ZONE_KM, "tile_id": "<zone>:X{ix:+04d}Y{iy:+04d}",
            "membership": "tile footprint intersects the zone lat/lon cell (9 x 9 samples); border tiles in both zones",
            "point_rule": "a point uses the tile of its own zone",
            "cell_order": "row-major from the tile's south-west corner (row 0 = south)",
            "zones": [z.__dict__ for z in zones()]}
