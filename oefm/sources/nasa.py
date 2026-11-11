"""NASA Earthdata (OB.DAAC, PO.DAAC, GES DISC) via earthaccess: search, inventory, sessions.

Searching CMR needs no login. Downloading needs a NASA Earthdata account, given either as
EARTHDATA_USERNAME / EARTHDATA_PASSWORD, EARTHDATA_TOKEN, or a ~/.netrc entry for
urs.earthdata.nasa.gov. GES DISC (IMERG) additionally needs the "NASA GESDISC DATA ARCHIVE"
application approved in the Earthdata profile.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any

from ..config import Config, Window

_login_lock = threading.Lock()
_logged_in = False
_local = threading.local()


def _earthaccess():
    import earthaccess  # imported lazily: unit tests and the region report need no network stack

    return earthaccess


# ----------------------------------------------------------------------------- login / session
def login() -> None:
    global _logged_in
    with _login_lock:
        if _logged_in:
            return
        ea = _earthaccess()
        strategies = []
        if os.getenv("EARTHDATA_TOKEN") or (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")):
            strategies.append("environment")
        strategies.append("netrc")
        last = None
        for strategy in strategies:
            try:
                auth = ea.login(strategy=strategy)
                if auth is not None and getattr(auth, "authenticated", False):
                    _logged_in = True
                    return
            except Exception as exc:  # noqa: BLE001 - try the next strategy
                last = exc
        raise RuntimeError(
            "NASA Earthdata login failed. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD (or EARTHDATA_TOKEN) "
            f"in credentials.sh, or add urs.earthdata.nasa.gov to ~/.netrc. Last error: {last}"
        )


def session():
    """A requests session carrying Earthdata auth; one per thread."""
    s = getattr(_local, "session", None)
    if s is None:
        login()
        s = _earthaccess().get_requests_https_session()
        _local.session = s
    return s


# ----------------------------------------------------------------------------- collections
def describe_collection(short_name: str, version: str | None = None, provider: str | None = None) -> list[dict]:
    """Versions, providers and temporal extents of a CMR collection (all versions if version is None)."""
    kw: dict[str, Any] = {"short_name": short_name}
    if provider:
        kw["provider"] = provider
    out = []
    for c in _earthaccess().search_datasets(**kw):
        umm, meta = c["umm"], c["meta"]
        if version and umm.get("Version") != version:
            continue
        ranges = (umm.get("TemporalExtents") or [{}])[0].get("RangeDateTimes") or [{}]
        out.append({
            "short_name": umm.get("ShortName"),
            "version": umm.get("Version"),
            "provider": meta.get("provider-id"),
            "concept_id": meta.get("concept-id"),
            "begin": ranges[0].get("BeginningDateTime"),
            "end": ranges[-1].get("EndingDateTime"),
            "title": umm.get("EntryTitle"),
            "doi": (umm.get("DOI") or {}).get("DOI"),
            "processing_level": (umm.get("ProcessingLevel") or {}).get("Id"),
        })
    return out


def all_versions(short_name: str, provider: str | None = None) -> list[str]:
    return sorted({c["version"] for c in describe_collection(short_name, provider=provider)})


# ----------------------------------------------------------------------------- granules
def search(product: dict, bbox: tuple[float, float, float, float] | None, start: date, end: date) -> list:
    kw: dict[str, Any] = {
        "short_name": product["short_name"],
        "temporal": (f"{start:%Y-%m-%d}T00:00:00Z", f"{end:%Y-%m-%d}T23:59:59Z"),
        "count": -1,
    }
    if product.get("version"):
        kw["version"] = product["version"]
    if product.get("provider"):
        kw["provider"] = product["provider"]
    if bbox is not None:
        kw["bounding_box"] = tuple(bbox)
    return _earthaccess().search_data(**kw)


def granule_row(g, product: dict, cfg: Config) -> dict[str, Any] | None:
    """Flatten one CMR granule into a manifest row (None if it has no downloadable file)."""
    umm = g["umm"]
    links = [u for u in g.data_links(access="external") if u.startswith("http")]
    links = [u for u in links if not u.endswith((".md5", ".sha256", ".xml", ".cmr.json", ".dmrpp"))]
    if not links:
        return None
    url = links[0]
    filename = PurePosixPath(url.split("?")[0]).name
    rng = (umm.get("TemporalExtent") or {}).get("RangeDateTime") or {}
    t0 = rng.get("BeginningDateTime") or (umm.get("TemporalExtent") or {}).get("SingleDateTime")
    t1 = rng.get("EndingDateTime") or t0

    size, exact, algo, cksum = _size_and_checksum(umm, filename, g)
    spatial = ((umm.get("SpatialExtent") or {}).get("HorizontalSpatialDomain") or {}).get("Geometry") or {}
    w, s, e, n = _bounds(spatial)
    when = _parse_time(t0)
    sub = f"{when:%Y}/{when:%m}" if when else "undated"
    return {
        "granule_key": f"{product['key']}:{filename}",
        "product": product["key"],
        "kind": product["kind"],
        "family": product.get("family", product.get("variable", "")),
        "source": "nasa_earthdata",
        "short_name": product["short_name"],
        "version": product.get("version"),
        "provider": g["meta"].get("provider-id"),
        "concept_id": g["meta"].get("concept-id"),
        "granule_ur": umm.get("GranuleUR"),
        "filename": filename,
        "url": url,
        "time_start": t0,
        "time_end": t1,
        "west": w, "south": s, "east": e, "north": n,
        "footprint": spatial or None,
        "size_bytes": size,
        "size_exact": exact,
        "checksum_algo": algo,
        "checksum_expected": cksum,
        "request": None,
        "local_path": f"raw/{product['key']}/{sub}/{filename}",
    }


def _size_and_checksum(umm: dict, filename: str, g) -> tuple[int | None, int, str | None, str | None]:
    infos = (umm.get("DataGranule") or {}).get("ArchiveAndDistributionInformation") or []
    info = next((i for i in infos if i.get("Name") == filename), infos[0] if infos else {})
    algo = cksum = None
    if info.get("Checksum"):
        algo = str(info["Checksum"].get("Algorithm", "")).lower().replace("-", "") or None
        cksum = info["Checksum"].get("Value")
    if info.get("SizeInBytes"):
        return int(info["SizeInBytes"]), 1, algo, cksum
    unit = {"KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12, "B": 1}.get(str(info.get("SizeUnit", "MB")).upper(), 1e6)
    if info.get("Size") is not None:
        return int(float(info["Size"]) * unit), 0, algo, cksum
    try:
        return int(g.size() * 1e6), 0, algo, cksum  # earthaccess reports MB
    except Exception:  # noqa: BLE001
        return None, 0, algo, cksum


def _bounds(geom: dict) -> tuple[float | None, ...]:
    lons: list[float] = []
    lats: list[float] = []
    for r in geom.get("BoundingRectangles") or []:
        lons += [r["WestBoundingCoordinate"], r["EastBoundingCoordinate"]]
        lats += [r["SouthBoundingCoordinate"], r["NorthBoundingCoordinate"]]
    for poly in geom.get("GPolygons") or []:
        for p in (poly.get("Boundary") or {}).get("Points") or []:
            lons.append(p["Longitude"])
            lats.append(p["Latitude"])
    if not lons:
        return (None, None, None, None)
    return (min(lons), min(lats), max(lons), max(lats))


def _parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


# ----------------------------------------------------------------------------- inventory
def inventory(product: dict, cfg: Config, log=print) -> list[dict]:
    """All granules of one product over the pilot regions and windows, deduplicated by file name."""
    is_sat = product["kind"] == "satellite"
    windows = cfg.satellite_windows() if is_sat else cfg.physical_windows()
    if product.get("only_after"):
        windows = _after(windows, _collection_end(cfg.products[product["only_after"]]), log, product["key"])
    targets = list(cfg.regions.values()) if is_sat else [cfg.domain()]
    rows: dict[str, dict] = {}
    for win in windows:
        for region in targets:
            granules = search(product, region.bbox, win.start, win.end)
            for g in granules:
                row = granule_row(g, product, cfg)
                if row is None:
                    continue
                prev = rows.get(row["granule_key"])
                regions = {region.key} if is_sat else set(cfg.regions)
                if prev:
                    regions |= set(prev["regions"].split(","))
                    # CMR may list one file under two GranuleURs (e.g. MODIS Aqua 2021-2025)
                    if row["concept_id"] not in prev["concept_id"].split(";"):
                        prev["concept_id"] += ";" + row["concept_id"]
                    prev["regions"] = ",".join(sorted(regions))
                    continue
                row["regions"] = ",".join(sorted(regions))
                row["windows"] = win.label()
                rows[row["granule_key"]] = row
            log(f"  {product['key']:18s} {win.label()} {region.key:14s} {len(granules):6d} CMR records")
    return list(rows.values())


def _collection_end(base: dict) -> date | None:
    info = describe_collection(base["short_name"], base.get("version"), base.get("provider"))
    ends = [c["end"] for c in info if c.get("end")]
    if not ends:
        return None
    return _parse_time(max(ends)).date()


def _after(windows: list[Window], cutoff: date | None, log, key: str) -> list[Window]:
    if cutoff is None:
        log(f"  {key}: base collection has no end date; nothing to fill")
        return []
    out = []
    for w in windows:
        if w.end <= cutoff:
            continue
        out.append(Window(max(w.start, cutoff + timedelta(days=1)), w.end, w.names))
    log(f"  {key}: only dates after {cutoff} (end of the base collection)")
    return out


def dump(obj: Any) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))
