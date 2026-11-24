"""Read config/pilot.yaml: regions, time windows, products, and where data lives.

A config may `extends: <other.yaml>` (relative to itself): the other file is loaded first and this
one's top-level keys replace it, except `processing`, `training` and `satellite`, which are merged one
level deep. Global configs (config/biogoship.yaml) also have `zones`: each a local LAEA projection
centre {lat_0, lon_0}; every region then names its `zone` and may carry its own `windows`
([{name, start, end}]), used instead of the global time windows for that region's searches.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "pilot.yaml"
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Region:
    key: str
    name: str
    west: float
    south: float
    east: float
    north: float
    notes: str = ""
    zone: str | None = None
    windows: tuple = ()

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    def area_km2(self) -> float:
        """Exact area of a lon/lat rectangle on a sphere of mean Earth radius."""
        lat = math.sin(math.radians(self.north)) - math.sin(math.radians(self.south))
        return EARTH_RADIUS_KM**2 * math.radians(self.east - self.west) * lat

    def width_km(self) -> float:
        mid = math.radians((self.south + self.north) / 2)
        return EARTH_RADIUS_KM * math.radians(self.east - self.west) * math.cos(mid)

    def height_km(self) -> float:
        return EARTH_RADIUS_KM * math.radians(self.north - self.south)

    def contains(self, lon, lat):
        """Works on scalars and numpy arrays."""
        return (lon >= self.west) & (lon <= self.east) & (lat >= self.south) & (lat <= self.north)


@dataclass(frozen=True)
class Window:
    """A continuous date range [start, end], both inclusive, with the names that formed it."""

    start: date
    end: date
    names: tuple[str, ...] = field(default=())

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    def label(self) -> str:
        return f"{self.start:%Y%m%d}-{self.end:%Y%m%d}"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    today: date

    # ---------------------------------------------------------------- paths
    @property
    def data_root(self) -> Path:
        root = Path(os.getenv("OEFM_DATA_ROOT") or self.raw.get("data_root", "data"))
        return root if root.is_absolute() else REPO_ROOT / root

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def manifest_path(self) -> Path:
        return self.data_root / "manifest" / "manifest.sqlite"

    @property
    def reports_dir(self) -> Path:
        return self.data_root / "reports"

    # -------------------------------------------------------------- regions
    @property
    def regions(self) -> dict[str, Region]:
        out = {}
        for key, spec in self.raw["regions"].items():
            w, s, e, n = (float(v) for v in spec["bbox"])
            if not (w < e and s < n):
                raise ValueError(f"region {key}: bbox must be [west, south, east, north]")
            wins = tuple(Window(self._date(x["start"]), self._date(x["end"]), (x.get("name", "window"),))
                         for x in spec.get("windows", []) or [])
            out[key] = Region(key, spec.get("name", key), w, s, e, n, (spec.get("notes") or "").strip(),
                              spec.get("zone"), tuple(merge_windows(list(wins))))
        return out

    @property
    def zones(self) -> dict[str, dict]:
        """Zone key -> {lat_0, lon_0, ...}; empty for single-projection (regional) configs."""
        return dict(self.raw.get("zones") or {})

    def domain(self, zone: str | None = None) -> Region:
        """Union bounding box of all regions (of one zone) plus the configured margin."""
        regions = [r for r in self.regions.values() if zone is None or r.zone == zone]
        m = float(self.raw.get("domain_margin_deg", 0.0))
        return Region(
            "domain",
            "Pilot domain (union of regions + margin)",
            min(r.west for r in regions) - m,
            min(r.south for r in regions) - m,
            max(r.east for r in regions) + m,
            max(r.north for r in regions) + m,
        )

    # -------------------------------------------------------------- windows
    def _date(self, value: Any) -> date:
        if isinstance(value, date):
            return value
        if str(value).lower() == "today":
            return self.today
        return date.fromisoformat(str(value))

    def region_windows(self, region: Region) -> list[Window]:
        """The region's own windows if it has any, else the global satellite windows."""
        return list(region.windows) if region.windows else self.satellite_windows()

    def satellite_windows(self) -> list[Window]:
        """Continuous observation windows (core + historical + events), merged where they touch.
        Configs without time.core (global, per-region windows): the union of the region windows."""
        t = self.raw.get("time") or {}
        if "core" not in t:
            return merge_windows([w for r in self.regions.values() for w in r.windows])
        windows = [Window(self._date(t["core"]["start"]), self._date(t["core"]["end"]), ("core",))]
        hist = t.get("historical", {})
        for year in hist.get("years", []):
            if hist.get("full_years"):
                windows.append(Window(date(year, 1, 1), date(year, 12, 31), (f"{year}_full",)))
                continue
            for w in hist.get("windows", []):
                start = date.fromisoformat(f"{year}-{w['start']}")
                end = date.fromisoformat(f"{year}-{w['end']}")
                windows.append(Window(start, end, (f"{year}_{w['name']}",)))
        for ev in self.events:
            windows.append(Window(self._date(ev["start"]), self._date(ev["end"]), (ev["name"],)))
        return merge_windows(windows)

    @property
    def events(self) -> list[dict]:
        """time.events plus the events in every file listed under time.event_files (paths relative
        to the config file), e.g. the generated eDNA cruise windows."""
        t = self.raw.get("time") or {}
        out = list(t.get("events", []) or [])
        for name in t.get("event_files", []) or []:
            f = (self.path.parent / name) if not Path(name).is_absolute() else Path(name)
            with open(f) as fh:
                out += list((yaml.safe_load(fh) or {}).get("events", []) or [])
        return out

    def physical_windows(self) -> list[Window]:
        """Satellite windows extended back by physical_lead_days, merged again."""
        lead = timedelta(days=int((self.raw.get("time") or {}).get("physical_lead_days", 0)))
        return merge_windows([Window(w.start - lead, w.end, w.names) for w in self.satellite_windows()])

    # ------------------------------------------------------------- products
    def core_start(self) -> date:
        """Start of the 2024-onward 'core' period (status and report columns); 2024-01-01 when the
        config has no global time.core (global configs with per-region windows)."""
        core = (self.raw.get("time") or {}).get("core")
        return self._date(core["start"]) if core else date(2024, 1, 1)

    @property
    def satellite_products(self) -> dict[str, dict]:
        """Satellite products, minus `exclude_products` (a config extending another can drop some)."""
        skip = set(self.raw.get("exclude_products") or [])
        return {k: {**v, "key": k, "kind": "satellite", "source": v.get("source", "nasa_earthdata")}
                for k, v in self.raw.get("satellite", {}).items() if k not in skip}

    @property
    def physical_products(self) -> dict[str, dict]:
        return {k: {**v, "key": k, "kind": "physical"} for k, v in self.raw.get("physical", {}).items()}

    @property
    def target_products(self) -> dict[str, dict]:
        """Model products used only as auxiliary training targets (never encoder inputs). Stored
        and downloaded like physical products, but kept out of crops and the physics cube."""
        return {k: {**v, "key": k, "kind": "physical", "role": "target"} for k, v in self.raw.get("targets", {}).items()}

    @property
    def products(self) -> dict[str, dict]:
        return {**self.satellite_products, **self.physical_products, **self.target_products}

    def select_products(self, keys: list[str] | None) -> dict[str, dict]:
        """Products by key; 'satellite' / 'physical' / family names (PACE, OLCI, ...) also work."""
        allp = self.products
        if not keys:
            return allp
        out: dict[str, dict] = {}
        for k in keys:
            low = k.lower()
            hits = {pk: p for pk, p in allp.items()
                    if pk == k or p["kind"] == low or str(p.get("family", "")).lower() == low
                    or str(p.get("source", "")).lower() == low}
            if not hits:
                raise SystemExit(f"Unknown product '{k}'. Known: {', '.join(allp)}")
            out.update(hits)
        return out


def merge_windows(windows: list[Window]) -> list[Window]:
    """Sort and merge windows that overlap or touch (end + 1 day >= next start)."""
    merged: list[Window] = []
    for w in sorted(windows, key=lambda w: (w.start, w.end)):
        if merged and w.start <= merged[-1].end + timedelta(days=1):
            last = merged[-1]
            merged[-1] = Window(last.start, max(last.end, w.end), last.names + w.names)
        else:
            merged.append(w)
    return merged


MERGED_KEYS = ("processing", "training", "satellite")


def _read(path: Path) -> dict:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("extends", None)
    if not base:
        return raw
    out = _read((path.parent / base) if not Path(base).is_absolute() else Path(base))
    for k, v in raw.items():
        if k in MERGED_KEYS and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, today: date | None = None) -> Config:
    path = Path(path or os.getenv("OEFM_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    return Config(raw=_read(path), path=path, today=today or datetime.now(timezone.utc).date())


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
