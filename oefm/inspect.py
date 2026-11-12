"""Inspect downloaded files: structure, wavelengths, flags, fill/negative values, resolution, maps.

Nothing here changes or filters data. Candidate quality masks are *evaluated* (how many
pixels each would keep, by region and by water depth) so a mask policy can be chosen later.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, Region

# OB.DAAC l2_flags names. The default L3 ocean-colour binning mask (l2bin flaguse for OC) and
# two looser alternatives; STRAYLIGHT is what usually removes near-shore and near-cloud pixels.
MASKS: dict[str, list[str]] = {
    "minimal": ["LAND", "CLDICE", "ATMFAIL", "NAVFAIL", "HILT"],
    "l3_no_straylight": ["ATMFAIL", "LAND", "HIGLINT", "HILT", "HISATZEN", "CLDICE", "COCCOLITH",
                         "HISOLZEN", "LOWLW", "CHLFAIL", "NAVWARN", "MAXAERITER", "ATMWARN", "NAVFAIL"],
    "l3_default": ["ATMFAIL", "LAND", "HIGLINT", "HILT", "HISATZEN", "STRAYLIGHT", "CLDICE", "COCCOLITH",
                   "HISOLZEN", "LOWLW", "CHLFAIL", "NAVWARN", "MAXAERITER", "ATMWARN", "NAVFAIL"],
}
DEPTH_CLASSES = [("coastal_0_20m", 0, 20), ("shelf_20_200m", 20, 200), ("offshore_gt200m", 200, 1e5)]
GLOBAL_ATTRS = ["title", "product_name", "processing_version", "processing_level", "instrument", "platform",
                "time_coverage_start", "time_coverage_end", "spatialResolution", "date_created",
                "geospatial_lat_min", "geospatial_lat_max", "geospatial_lon_min", "geospatial_lon_max",
                "id", "source", "references", "product_version", "gds_version_id"]


# ----------------------------------------------------------------------------- generic structure
def structure(path: Path) -> dict[str, Any]:
    """Every group/variable with dims, shape, dtype, units, fill and scaling; selected global attrs."""
    import netCDF4

    out: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size, "global_attrs": {}, "variables": {}}
    with netCDF4.Dataset(path) as ds:
        for a in GLOBAL_ATTRS:
            if a in ds.ncattrs():
                out["global_attrs"][a] = _plain(ds.getncattr(a))

        def walk(g, prefix=""):
            for name, v in g.variables.items():
                attrs = {a: _plain(v.getncattr(a)) for a in v.ncattrs()
                         if a in ("units", "long_name", "standard_name", "_FillValue", "scale_factor", "add_offset",
                                  "valid_min", "valid_max", "flag_meanings", "flag_masks")}
                out["variables"][prefix + name] = {"dims": list(v.dimensions), "shape": list(v.shape),
                                                   "dtype": str(v.dtype), **attrs}
            for gname, sub in g.groups.items():
                walk(sub, f"{prefix}{gname}/")

        walk(ds)
    return out


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# ----------------------------------------------------------------------------- L2 swaths
class L2Swath:
    """Lazy access to an OB.DAAC Level-2 file (MODIS, VIIRS, OLCI, PACE AOP/SFREFL)."""

    def __init__(self, path: Path) -> None:
        import netCDF4

        self.path = path
        self.ds = netCDF4.Dataset(path)
        self.geo = self.ds["geophysical_data"]
        nav = self.ds["navigation_data"]
        self.lat = np.asarray(nav["latitude"][:], dtype="f4")
        self.lon = np.asarray(nav["longitude"][:], dtype="f4")
        self.lat[np.abs(self.lat) > 90] = np.nan
        self.lon[np.abs(self.lon) > 180] = np.nan
        self.kind, self.bands = self._find_bands()

    def close(self) -> None:
        self.ds.close()

    def _find_bands(self) -> tuple[str, list[tuple[float, str, int | None]]]:
        """(variable family, [(wavelength_nm, variable_name, index_in_3d_or_None)])."""
        names = list(self.geo.variables)
        for fam in ("Rrs", "rhos"):
            if fam in names and self.geo[fam].ndim == 3:
                sbp = self.ds["sensor_band_parameters"]
                wl_name = next((n for n in ("wavelength_3d", "wavelength") if n in sbp.variables
                                and sbp[n].shape[0] == self.geo[fam].shape[2]), None)
                wls = np.asarray(sbp[wl_name][:], dtype=float) if wl_name else np.arange(self.geo[fam].shape[2])
                return fam, [(float(w), fam, i) for i, w in enumerate(wls)]
            per_band = sorted(
                (float(m.group(1)), n, None) for n in names if (m := re.fullmatch(fam + r"_(\d+(?:\.\d+)?)", n))
            )
            if per_band:
                return fam, per_band
        return "none", []

    @property
    def wavelengths(self) -> list[float]:
        return [b[0] for b in self.bands]

    def window(self, box: Region) -> tuple[slice, slice] | None:
        m = box.contains(self.lon, self.lat)
        if not m.any():
            return None
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]
        return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)

    def band(self, wl: float, rs: slice, cs: slice) -> tuple[np.ndarray, float]:
        """Band nearest to wl (nm) inside the row/col window, NaN where fill."""
        i = int(np.argmin([abs(b[0] - wl) for b in self.bands]))
        w, name, idx = self.bands[i]
        v = self.geo[name]
        data = v[rs, cs, idx] if idx is not None else v[rs, cs]
        return np.ma.filled(np.ma.asarray(data, dtype="f4"), np.nan), w

    def flags(self, rs: slice, cs: slice) -> tuple[np.ndarray, dict[str, int]]:
        v = self.geo["l2_flags"]
        v.set_auto_maskandscale(False)
        bits = np.asarray(v[rs, cs]).astype(np.int64)
        names = str(v.getncattr("flag_meanings")).split()
        masks = [int(x) for x in np.atleast_1d(v.getncattr("flag_masks"))]
        return bits, dict(zip(names, masks))

    def pixel_spacing_km(self, rs: slice, cs: slice, inside: np.ndarray | None = None) -> dict[str, float]:
        """Median distance between neighbouring pixels: over the region, and along the swath centre.

        Medians, not single pairs: OLCI geolocation repeats some values and VIIRS rows jump at
        scan / bow-tie boundaries, so any one pair can read 0 km or 12 km.
        """
        lat, lon = self.lat[rs, cs], self.lon[rs, cs]
        out: dict[str, float] = {}
        scan = _hav(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        track = _hav(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
        if inside is not None:
            scan = scan[inside[:, :-1] & inside[:, 1:]]
            track = track[inside[:-1, :] & inside[1:, :]]
        scan, track = scan[np.isfinite(scan) & (scan > 0)], track[np.isfinite(track) & (track > 0)]
        if scan.size:
            out["along_scan_km"] = float(np.median(scan))
        if track.size:
            out["along_track_km"] = float(np.median(track))
        c = self.lat.shape[1] // 2  # swath centre (nadir)
        nad = _hav(self.lat[:-1, c], self.lon[:-1, c], self.lat[1:, c], self.lon[1:, c])
        nad_s = _hav(self.lat[:, c - 1], self.lon[:, c - 1], self.lat[:, c], self.lon[:, c])
        nad, nad_s = nad[np.isfinite(nad) & (nad > 0)], nad_s[np.isfinite(nad_s) & (nad_s > 0)]
        if nad.size and nad_s.size:
            out["nadir_scan_km"] = float(np.median(nad_s))
            out["nadir_track_km"] = float(np.median(nad))
        return out


def _hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = (np.radians(np.asarray(x, dtype="f8")) for x in (a1, o1, a2, o2))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def mask_bits(names_to_bits: dict[str, int], wanted: list[str]) -> int:
    bits = 0
    for n in wanted:
        bits |= names_to_bits.get(n, 0)
    return bits


def inspect_l2(path: Path, product: dict, cfg: Config, depth=None) -> dict[str, Any]:
    """Per-region statistics of one L2 granule (no pixels are discarded)."""
    sw = L2Swath(path)
    try:
        ref_wl = 555.0 if sw.kind == "Rrs" else 865.0
        res: dict[str, Any] = {
            "file": path.name, "product": product["key"], "variable": sw.kind,
            "n_bands": len(sw.bands), "wavelengths_nm": sw.wavelengths,
            "swath_shape": list(sw.lat.shape), "time_start": _attr(sw.ds, "time_coverage_start"),
            "processing_version": _attr(sw.ds, "processing_version"),
            "units": _units(sw), "regions": {},
        }
        if sw.kind == "rhos":
            res["sargassum_bands"] = sargassum_band_check(sw.wavelengths)
        for region in cfg.regions.values():
            win = sw.window(region)
            if win is None:
                continue
            rs, cs = win
            inreg = region.contains(sw.lon[rs, cs], sw.lat[rs, cs])
            ref, ref_used = sw.band(ref_wl, rs, cs)
            bits, fmap = sw.flags(rs, cs)
            n = int(inreg.sum())
            stats: dict[str, Any] = {"n_pixels": n, "ref_wavelength_nm": ref_used,
                                     **{k: round(v, 3) for k, v in sw.pixel_spacing_km(rs, cs, inreg).items()}}
            stats["flag_fraction"] = {name: round(float(((bits & b) != 0)[inreg].mean()), 4) for name, b in fmap.items() if b}
            finite = np.isfinite(ref) & inreg
            stats["ref_fill_fraction"] = round(1 - finite.sum() / max(n, 1), 4)
            stats["ref_negative_fraction"] = round(float((ref[finite] < 0).mean()) if finite.any() else float("nan"), 4)
            stats["mask_valid_fraction"] = {}
            d = depth(sw.lon[rs, cs], sw.lat[rs, cs]) if depth is not None else None
            for mname, flist in MASKS.items():
                keep = ((bits & mask_bits(fmap, flist)) == 0) & finite
                stats["mask_valid_fraction"][mname] = round(float(keep[inreg].mean()) if n else float("nan"), 4)
                if d is not None:
                    for cname, lo, hi in DEPTH_CLASSES:
                        zone = inreg & (d >= lo) & (d < hi)
                        if zone.any():
                            stats.setdefault("mask_valid_by_depth", {}).setdefault(mname, {})[cname] = \
                                round(float(keep[zone].mean()), 4)
            # Fill / negative fraction of every band, only for pixels that are not land or cloud.
            water = inreg & ((bits & mask_bits(fmap, ["LAND", "CLDICE"])) == 0)
            per_band = []
            for wl, name, idx in sw.bands:
                v = sw.geo[name]
                arr = np.ma.filled(np.ma.asarray(v[rs, cs, idx] if idx is not None else v[rs, cs], dtype="f4"), np.nan)
                vals = arr[water]
                fin = np.isfinite(vals)
                per_band.append({"nm": wl, "fill": round(1 - fin.mean(), 4) if vals.size else None,
                                 "negative": round(float((vals[fin] < 0).mean()), 4) if fin.any() else None,
                                 "median": float(np.nanmedian(vals)) if fin.any() else None})
            stats["bands_over_clear_water"] = per_band
            stats["missing_bands"] = [b["nm"] for b in per_band if b["fill"] == 1.0]
            res["regions"][region.key] = stats
        return res
    finally:
        sw.close()


def sargassum_band_check(wls: list[float]) -> dict[str, Any]:
    """Which rhos bands exist for red/NIR/SWIR floating-algae indices (FAI/AFAI-type baselines)."""
    wls = np.asarray(wls)
    targets = {"red_645": 645, "red_667": 667, "red_edge_709": 709, "nir_748": 748, "nir_859": 859,
               "nir_869": 869, "swir_1240": 1240, "swir_1640": 1640, "swir_2130": 2130}
    out = {}
    for name, t in targets.items():
        i = int(np.argmin(np.abs(wls - t)))
        out[name] = {"nearest_nm": float(wls[i]), "offset_nm": round(float(wls[i] - t), 1)}
    out["n_bands_600_900nm"] = int(((wls >= 600) & (wls <= 900)).sum())
    out["n_bands_swir_gt1000nm"] = int((wls > 1000).sum())
    return out


def _attr(ds, name):
    return _plain(ds.getncattr(name)) if name in ds.ncattrs() else None


def _units(sw: L2Swath) -> str | None:
    if not sw.bands:
        return None
    v = sw.geo[sw.bands[0][1]]
    return v.getncattr("units") if "units" in v.ncattrs() else None


# ----------------------------------------------------------------------------- gridded products
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")


def inspect_grid(path: Path, product: dict, cfg: Config) -> dict[str, Any]:
    import xarray as xr

    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        lat = next(n for n in LAT_NAMES if n in ds.coords or n in ds.variables)
        lon = next(n for n in LON_NAMES if n in ds.coords or n in ds.variables)
        dom = cfg.domain()
        lons = ds[lon].values
        wrap = float(np.nanmax(lons)) > 180
        w, e = (dom.west % 360, dom.east % 360) if wrap else (dom.west, dom.east)
        latv = ds[lat].values
        lat_sl = slice(dom.south, dom.north) if latv[0] < latv[-1] else slice(dom.north, dom.south)
        sub = ds.sel({lat: lat_sl, lon: slice(w, e)})
        res: dict[str, Any] = {
            "file": path.name, "product": product["key"], "lat_name": lat, "lon_name": lon,
            "lon_convention": "0-360" if wrap else "-180-180",
            "grid_step_deg": float(abs(np.diff(latv[:2]))[0]) if latv.size > 1 else None,
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "time": [str(t) for t in ds["time"].values[:8]] if "time" in ds.coords else None,
            "variables": {},
        }
        for name, v in sub.data_vars.items():
            if lat not in v.dims or lon not in v.dims:
                continue
            arr = np.asarray(v.values, dtype="f8") if v.dtype.kind in "fiu" else None
            info = {"dims": list(v.dims), "units": v.attrs.get("units"), "long_name": v.attrs.get("long_name"),
                    "fill_value": _plain(v.encoding.get("_FillValue"))}
            if arr is not None and arr.size:
                fin = np.isfinite(arr)
                info.update({"domain_nan_fraction": round(1 - fin.mean(), 4),
                             "min": float(np.nanmin(arr)) if fin.any() else None,
                             "max": float(np.nanmax(arr)) if fin.any() else None,
                             "mean": float(np.nanmean(arr)) if fin.any() else None})
            res["variables"][name] = info
        return res
    finally:
        ds.close()


# ----------------------------------------------------------------------------- bathymetry helper
def gebco_depth(cfg: Config):
    """A function (lon, lat) -> depth in metres (positive down) from the local GEBCO file, or None."""
    files = sorted((cfg.raw_dir / "gebco").glob("GEBCO_*.nc"))
    if not files:
        return None
    import netCDF4

    dom = cfg.domain()
    with netCDF4.Dataset(files[-1]) as ds:
        lat, lon = ds["lat"][:], ds["lon"][:]
        i0, i1 = np.searchsorted(lat, [dom.south, dom.north])
        j0, j1 = np.searchsorted(lon, [dom.west, dom.east])
        elev = np.asarray(ds["elevation"][i0:i1, j0:j1], dtype="f4")
        lat, lon = np.asarray(lat[i0:i1]), np.asarray(lon[j0:j1])

    def depth(qlon, qlat):
        i = np.clip(np.searchsorted(lat, qlat), 0, lat.size - 1)
        j = np.clip(np.searchsorted(lon, qlon), 0, lon.size - 1)
        d = -elev[i, j]
        d[~np.isfinite(qlat) | ~np.isfinite(qlon)] = np.nan
        return d

    return depth


# ----------------------------------------------------------------------------- maps
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 9, "axes.linewidth": 0.6})
    return plt


def _draw_regions(ax, cfg: Config):
    for r in cfg.regions.values():
        ax.plot([r.west, r.east, r.east, r.west, r.west], [r.south, r.south, r.north, r.north, r.south],
                lw=0.8, color="k")
        ax.text(r.west + 0.05, r.north - 0.05, r.key, fontsize=7, va="top")


def map_l2(path: Path, product: dict, cfg: Config, out_png: Path, stride: int = 0) -> list[Path]:
    """Two bare panels (no titles): reference band and flag classes, over the pilot domain."""
    plt = _mpl()
    sw = L2Swath(path)
    try:
        dom = cfg.domain()
        win = sw.window(dom)
        if win is None:
            return []
        rs, cs = win
        lat, lon = sw.lat[rs, cs], sw.lon[rs, cs]
        s = stride or max(1, int(max(lat.shape) / 1500))
        ref, wl = sw.band(555.0 if sw.kind == "Rrs" else 865.0, rs, cs)
        bits, fmap = sw.flags(rs, cs)
        cls = np.full(bits.shape, 0, "i1")                          # 0 valid under l3_default
        cls[(bits & mask_bits(fmap, MASKS["l3_default"])) != 0] = 1  # other flags
        cls[(bits & fmap.get("STRAYLIGHT", 0)) != 0] = 2
        cls[(bits & fmap.get("CLDICE", 0)) != 0] = 3
        cls[(bits & fmap.get("LAND", 0)) != 0] = 4
        out = []
        for suffix, data, kw in [
            (f"{sw.kind}{int(round(wl))}", ref, {"cmap": "viridis",
                                                "vmin": np.nanpercentile(ref, 2) if np.isfinite(ref).any() else 0,
                                                "vmax": np.nanpercentile(ref, 98) if np.isfinite(ref).any() else 1}),
            ("flags", cls.astype("f4"), {"cmap": plt.matplotlib.colors.ListedColormap(
                ["#2a7fbf", "#f2b134", "#d95f02", "#d9d9d9", "#6b6b6b"]), "vmin": -0.5, "vmax": 4.5}),
        ]:
            fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
            with warnings.catch_warnings():
                # Swath coordinates are not monotonic (bow-tie rows, curved edges); "nearest"
                # cells are fine for a diagnostic map, so matplotlib's warning is just noise.
                warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")
                m = ax.pcolormesh(lon[::s, ::s], lat[::s, ::s], data[::s, ::s], shading="nearest",
                                  rasterized=True, **kw)
            cb = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.01)
            if suffix == "flags":
                cb.set_ticks(range(5))
                cb.set_ticklabels(["valid (L3 mask)", "other flag", "STRAYLIGHT", "CLDICE", "LAND"])
            else:
                cb.set_label(f"{sw.kind} {wl:.0f} nm")
            _draw_regions(ax, cfg)
            ax.set_xlim(dom.west, dom.east)
            ax.set_ylim(dom.south, dom.north)
            ax.set_aspect(1 / math.cos(math.radians((dom.south + dom.north) / 2)))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            p = out_png.with_name(f"{out_png.stem}_{suffix}.png")
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            out.append(p)
        return out
    finally:
        sw.close()


def map_grid(path: Path, product: dict, cfg: Config, out_png: Path, variable: str | None = None) -> list[Path]:
    import xarray as xr

    plt = _mpl()
    preferred = ["analysed_sst", "ws", "precipitation", "adt", "uo", "elevation"]
    ds = xr.open_dataset(path, decode_timedelta=False)
    try:
        lat = next(n for n in LAT_NAMES if n in ds.variables)
        lon = next(n for n in LON_NAMES if n in ds.variables)
        name = variable or next((v for v in preferred if v in ds.data_vars), None) or next(iter(ds.data_vars))
        v = ds[name]
        for d in v.dims:
            if d not in (lat, lon):
                v = v.isel({d: 0})
        dom = cfg.domain()
        lons = ds[lon].values
        if float(np.nanmax(lons)) > 180:
            v = v.assign_coords({lon: ((v[lon] + 180) % 360) - 180}).sortby(lon)
        latv = v[lat].values
        v = v.sel({lat: slice(dom.south, dom.north) if latv[0] < latv[-1] else slice(dom.north, dom.south),
                   lon: slice(dom.west, dom.east)})
        v = v.transpose(lat, lon)
        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
        m = ax.pcolormesh(v[lon], v[lat], v.values, shading="nearest", cmap="viridis", rasterized=True)
        fig.colorbar(m, ax=ax, fraction=0.03, pad=0.01).set_label(f"{name} ({v.attrs.get('units', '')})")
        _draw_regions(ax, cfg)
        ax.set_xlim(dom.west, dom.east)
        ax.set_ylim(dom.south, dom.north)
        ax.set_aspect(1 / math.cos(math.radians((dom.south + dom.north) / 2)))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        p = out_png.with_name(f"{out_png.stem}_{name}.png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        return [p]
    finally:
        ds.close()


def write_json(obj: Any, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj, indent=1, default=_plain))
