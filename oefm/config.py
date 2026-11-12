"""Read config/pilot.yaml: regions, time windows, products, and where data lives."""

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
            out[key] = Region(key, spec.get("name", key), w, s, e, n, spec.get("notes", "").strip())
        return out

    def domain(self) -> Region:
        """Union bounding box of all regions plus the configured margin (for physical subsets)."""
        regions = list(self.regions.values())
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

    def satellite_windows(self) -> list[Window]:
        """Continuous observation windows (core + historical + events), merged where they touch."""
        t = self.raw["time"]
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
        t = self.raw["time"]
        out = list(t.get("events", []) or [])
        for name in t.get("event_files", []) or []:
            f = (self.path.parent / name) if not Path(name).is_absolute() else Path(name)
            with open(f) as fh:
                out += list((yaml.safe_load(fh) or {}).get("events", []) or [])
        return out

    def physical_windows(self) -> list[Window]:
        """Satellite windows extended back by physical_lead_days, merged again."""
        lead = timedelta(days=int(self.raw["time"].get("physical_lead_days", 0)))
        return merge_windows([Window(w.start - lead, w.end, w.names) for w in self.satellite_windows()])

    # ------------------------------------------------------------- products
    @property
    def satellite_products(self) -> dict[str, dict]:
        return {k: {**v, "key": k, "kind": "satellite", "source": v.get("source", "nasa_earthdata")}
                for k, v in self.raw.get("satellite", {}).items()}

    @property
    def physical_products(self) -> dict[str, dict]:
        return {k: {**v, "key": k, "kind": "physical"} for k, v in self.raw.get("physical", {}).items()}

    @property
    def products(self) -> dict[str, dict]:
        return {**self.satellite_products, **self.physical_products}

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


def load_config(path: str | Path | None = None, today: date | None = None) -> Config:
    path = Path(path or os.getenv("OEFM_CONFIG") or DEFAULT_CONFIG)
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=path, today=today or datetime.now(timezone.utc).date())


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
