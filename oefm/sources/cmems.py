"""Copernicus Marine (altimetry, GlobCurrent) via the Copernicus Marine Toolbox.

Copernicus Marine serves analysis-ready stores rather than files, so a "granule" here is one
monthly NetCDF subset of the pilot domain with every variable of the dataset (nothing derived).
The multi-year (reprocessed) dataset is used up to its last day; the near-real-time dataset only
after that. Credentials: COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD.
Catalogue lookups and size estimates need no login.
"""

from __future__ import annotations

import calendar
import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import Config


def _cm():
    import copernicusmarine

    return copernicusmarine


@lru_cache(maxsize=None)
def dataset_info(dataset_id: str) -> dict[str, Any]:
    """Time range (dates), variables, grid step and version label of a dataset."""
    cat = _cm().describe(dataset_id=dataset_id, disable_progress_bar=True)
    for prod in cat.products:
        for ds in prod.datasets:
            if ds.dataset_id != dataset_id:
                continue
            version = ds.versions[-1]
            for part in version.parts:
                for svc in part.services:
                    if svc.service_name not in ("arco-time-series", "arco-geo-series") or not svc.variables:
                        continue
                    tmin = tmax = step = None
                    for c in svc.variables[0].coordinates:
                        if c.coordinate_id == "time":
                            tmin, tmax = c.minimum_value, c.maximum_value
                        if c.coordinate_id in ("longitude", "x"):
                            step = c.step
                    to_date = lambda ms: datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date() if ms else None  # noqa: E731
                    return {
                        "product_id": prod.product_id,
                        "dataset_id": dataset_id,
                        "version": version.label,
                        "start": to_date(tmin),
                        "end": to_date(tmax),
                        "variables": [v.short_name for v in svc.variables],
                        "step_deg": step,
                    }
    raise RuntimeError(f"Copernicus Marine dataset {dataset_id} not found in the catalogue")


def _months(start: date, end: date):
    d = date(start.year, start.month, 1)
    while d <= end:
        last = date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
        yield max(d, start), min(last, end)
        d = last + timedelta(days=1)


def plan_requests(product: dict, cfg: Config) -> list[dict[str, Any]]:
    """Monthly subset requests covering the physical windows, MY first then NRT."""
    infos = [dataset_info(d) for d in product["datasets"]]
    out = []
    for win in cfg.physical_windows():
        for m0, m1 in _months(win.start, win.end):
            cursor = m0
            for info in infos:
                if cursor > m1:
                    break
                if info["start"] and info["start"] > m1:
                    continue
                if info["end"] and info["end"] < cursor:
                    continue
                s = max(cursor, info["start"] or cursor)
                e = min(m1, info["end"] or m1)
                if s > e:
                    continue
                out.append({"dataset": info, "start": s, "end": e, "window": win.label()})
                cursor = e + timedelta(days=1)
            if cursor <= m1:
                out.append({"dataset": None, "start": cursor, "end": m1, "window": win.label()})
    return out


def inventory(product: dict, cfg: Config, log=print, estimate_sizes: bool = True) -> list[dict]:
    dom = cfg.domain()
    rows = []
    gaps = 0
    for req in plan_requests(product, cfg):
        info = req["dataset"]
        if info is None:
            gaps += (req["end"] - req["start"]).days + 1
            log(f"  {product['key']}: no dataset covers {req['start']}..{req['end']} (recorded as missing)")
            continue
        variables = None if product.get("variables", "all") == "all" else list(product["variables"])
        fname = f"{info['dataset_id']}_{req['start']:%Y%m%d}_{req['end']:%Y%m%d}.nc"
        request = {
            "dataset_id": info["dataset_id"],
            "dataset_version": info["version"],
            "variables": variables,
            "minimum_longitude": dom.west, "maximum_longitude": dom.east,
            "minimum_latitude": dom.south, "maximum_latitude": dom.north,
            "start_datetime": f"{req['start']:%Y-%m-%d}T00:00:00",
            "end_datetime": f"{req['end']:%Y-%m-%d}T23:59:59",
        }
        if product.get("max_depth_m") is not None:  # 3-D model fields: keep only the top levels
            request["minimum_depth"] = 0.0
            request["maximum_depth"] = float(product["max_depth_m"])
        have_login = bool(os.getenv("COPERNICUSMARINE_SERVICE_USERNAME"))
        size = _estimate_size(request) if (estimate_sizes and have_login) else None
        if size is None:
            size = _grid_estimate(info, dom, req["start"], req["end"], variables)
        rows.append({
            "granule_key": f"{product['key']}:{fname}",
            "product": product["key"],
            "kind": "physical",
            "family": product.get("variable", ""),
            "source": "copernicus_marine",
            "short_name": info["dataset_id"],
            "version": info["version"],
            "provider": info["product_id"],
            "concept_id": None,
            "granule_ur": None,
            "filename": fname,
            "url": f"copernicusmarine://{info['dataset_id']}",
            "time_start": request["start_datetime"] + "Z",
            "time_end": request["end_datetime"] + "Z",
            "west": dom.west, "south": dom.south, "east": dom.east, "north": dom.north,
            "footprint": None,
            "size_bytes": size,
            "size_exact": 0,
            "checksum_algo": None,
            "checksum_expected": None,
            "request": request,
            "local_path": f"raw/{product['key']}/{req['start']:%Y}/{fname}",
            "regions": ",".join(sorted(cfg.regions)),
            "windows": req["window"],
        })
    log(f"  {product['key']:18s} {len(rows)} monthly subsets; {gaps} days not covered by any dataset")
    return rows


def _estimate_size(request: dict) -> int | None:
    try:
        resp = _cm().subset(**_subset_kwargs(request), dry_run=True, disable_progress_bar=True)
        mb = getattr(resp, "file_size", None)
        return int(mb * 1e6) if mb else None
    except Exception:  # noqa: BLE001 - an estimate is optional
        return None


def _grid_estimate(info: dict, dom, start: date, end: date, variables) -> int | None:
    """Uncompressed float32 size of the subset (upper bound; NetCDF compression makes files smaller)."""
    step = info.get("step_deg")
    if not step:
        return None
    nx = int((dom.east - dom.west) / step) + 2
    ny = int((dom.north - dom.south) / step) + 2
    nvar = len(variables or info["variables"])
    return int(nx * ny * nvar * 4 * ((end - start).days + 1))


def _subset_kwargs(request: dict) -> dict:
    kw = {k: v for k, v in request.items() if k not in ("dataset_version",) and v is not None}
    kw["dataset_version"] = request.get("dataset_version")
    kw["coordinates_selection_method"] = "outside"  # cover the whole box, never clip inside it
    return {k: v for k, v in kw.items() if v is not None}


def download(row: dict, dest: Path) -> None:
    """Fetch one subset to dest atomically (temporary name, then rename)."""
    if not (os.getenv("COPERNICUSMARINE_SERVICE_USERNAME") and os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")):
        raise RuntimeError("Set COPERNICUSMARINE_SERVICE_USERNAME / _PASSWORD in credentials.sh")
    import json

    request = json.loads(row["request"]) if isinstance(row["request"], str) else row["request"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = dest.name + ".part.nc"
    (dest.parent / tmp_name).unlink(missing_ok=True)
    _cm().subset(
        **_subset_kwargs(request),
        output_directory=str(dest.parent),
        output_filename=tmp_name,
        overwrite=True,
        disable_progress_bar=True,
    )
    (dest.parent / tmp_name).replace(dest)
