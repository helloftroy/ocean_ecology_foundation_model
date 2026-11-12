"""Sensor-zenith (HISATZEN) diagnostic: what relaxing the HISATZEN mask would recover, and at what cost.

OB.DAAC L2 files flag HISATZEN where the sensor zenith exceeds l2gen's `satzen` threshold (60 deg
in every file checked), but they do not store the angle itself. It is reconstructed here from each
file's own geolocation: a pixel's great-circle distance from the sub-satellite point plus the
orbit altitude give the view zenith at the ground. The sub-satellite point is the swath-centre
column for the cross-track scanners (MODIS, VIIRS, OCI; OCI's +-20 deg along-track tilt is added),
and the column with the smallest pixel spacing for OLCI, whose swath is offset to the west.

The HISATZEN bit is an independent check on both the flag decoding and the geometry: if both are
right, the flag edge falls on the reconstructed 60-degree line.

Five outputs, all per sensor (product), none of which removes or alters any pixel:

1. decode check:  HISATZEN bit from each file's flag_meanings/flag_masks, l2gen's threshold, and
                  how well the flag matches the reconstructed angle;
2. coverage:      valid water pixels per region under the L3 default mask with and without
                  HISATZEN, i.e. what relaxing it recovers;
3. maps:          reconstructed zenith and reflectance over the domain with the flag edge drawn;
4. Rrs vs angle:  reflectance distribution per band in zenith bins (extremes, negatives);
5. overlap:       same-day matchups against another sensor viewing at < 45 deg, with the ratio
                  binned by the target sensor's zenith: an angle-dependent bias shows as a ratio
                  that drifts between the low- and high-angle classes.
"""

from __future__ import annotations

import json
import math
import os
import warnings
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, Region, load_config
from .inspect import MASKS, L2Swath, mask_bits

EARTH_KM = 6371.0088
# Nominal orbit altitudes (km). A 5 km error moves the 60-degree line by about 0.1 degree.
ALTITUDE_KM = {"MODIS": 705.0, "VIIRS": 824.0, "OLCI": 814.5, "PACE": 676.5}
ZEN_EDGES = [0, 15, 30, 45, 55, 60, 65, 75]
OVERLAP_CLASSES = [(0, 30), (30, 45), (45, 60), (60, 90)]
REFERENCE_MAX_ZEN = 45.0
LOW_ANGLE_MAX = 45     # target classes up to this angle form the baseline of the angle effect
TARGETS = {"Rrs": [412, 443, 490, 555, 665], "rhos": [443, 555, 667, 865, 1618]}
SAMPLE_PER_BIN = 400  # reflectance values kept per (granule, band, zenith bin) for pooled quantiles
BIN_DEG = 0.02        # matchup bins (diagnostic only, same as the overlap check)
RELAXED = "l3_no_hisatzen"


# ----------------------------------------------------------------------------- geometry
def _central_angle(a1, o1, a2, o2):
    a1, o1, a2, o2 = (np.radians(np.asarray(x, dtype="f8")) for x in (a1, o1, a2, o2))
    h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def nadir_column(sw: L2Swath, family: str) -> int:
    ncol = sw.lat.shape[1]
    if family != "OLCI":
        return ncol // 2
    rows = slice(None, None, max(1, sw.lat.shape[0] // 200))
    lat, lon = sw.lat[rows], sw.lon[rows]
    d = _central_angle(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    d[~(d > 0)] = np.nan  # OLCI geolocation repeats some values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sp = np.nanmedian(d, axis=0)
    sp = np.convolve(np.nan_to_num(sp, nan=np.nanmax(sp)), np.ones(31) / 31, "same")
    e = max(20, ncol // 20)  # the outermost columns have edge artefacts
    return int(np.argmin(sp[e:-e])) + e


def sensor_zenith(sw: L2Swath, family: str, rs: slice, cs: slice, nadir: int | None = None) -> np.ndarray:
    """View zenith (deg) at the ground for rows rs, columns cs of an L2 swath."""
    h = ALTITUDE_KM[family]
    nadir = nadir_column(sw, family) if nadir is None else nadir
    lat, lon = sw.lat[rs, cs], sw.lon[rs, cs]
    g = _central_angle(lat, lon, sw.lat[rs, nadir][:, None], sw.lon[rs, nadir][:, None])
    nav = sw.ds["navigation_data"]
    if family == "PACE" and "tilt" in nav.variables:
        t = np.radians(np.abs(np.ma.filled(nav["tilt"][rs], 0.0).astype("f8")))[:, None]
        ga = np.arcsin(np.clip((EARTH_KM + h) / EARTH_KM * np.sin(t), -1, 1)) - t  # along-track offset
        g = np.arccos(np.clip(np.cos(g) * np.cos(ga), -1, 1))
    eta = np.arctan2(EARTH_KM * np.sin(g), EARTH_KM + h - EARTH_KM * np.cos(g))  # nadir angle at sensor
    return np.degrees(g + eta).astype("f4")


def l2gen_params(sw: L2Swath) -> dict[str, Any]:
    try:
        grp = sw.ds["processing_control"]["input_parameters"]
    except (KeyError, IndexError):
        return {}
    out = {}
    for k in ("satzen", "masksatzen"):
        if k in grp.ncattrs():
            try:
                out[k] = float(str(grp.getncattr(k)).strip())
            except ValueError:
                out[k] = str(grp.getncattr(k))
    return out


# ----------------------------------------------------------------------------- per granule
def granule(args: tuple[str, str, dict, str]) -> dict[str, Any]:
    cfg_path, path, row, today = args
    cfg = load_config(cfg_path, date.fromisoformat(today))
    product = cfg.satellite_products[row["product"]]
    family = product["family"]
    rec: dict[str, Any] = {"granule_key": row["granule_key"], "product": row["product"], "family": family,
                           "time_start": row["time_start"], "path": row["local_path"]}
    try:
        sw = L2Swath(Path(path))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    try:
        dom = cfg.domain()
        win = sw.window(dom)
        if win is None:
            rec["error"] = "swath does not reach the domain"
            return rec
        rs, cs = win
        nadir = nadir_column(sw, family)
        zen = sensor_zenith(sw, family, rs, cs, nadir)
        bits, fmap = sw.flags(rs, cs)
        params = l2gen_params(sw)
        thr = params.get("satzen", 60.0)
        bit = fmap.get("HISATZEN")
        rec.update({"hisatzen_bit": bit, "l2gen_satzen": params.get("satzen"),
                    "l2gen_masksatzen": params.get("masksatzen"), "nadir_column": nadir,
                    "n_columns": int(sw.lat.shape[1]), "zen_max_domain": round(float(np.nanmax(zen)), 2)})
        lon, lat = sw.lon[rs, cs], sw.lat[rs, cs]
        ok = np.isfinite(zen) & np.isfinite(lon)

        # 1. decode check
        if bit:
            flag = (bits & bit) != 0
            rec["flagged_fraction_domain"] = round(float(flag[ok].mean()), 4)
            rec["flag_matches_angle"] = round(float(((zen >= thr) == flag)[ok].mean()), 4)
            edge = (flag[:, 1:] != flag[:, :-1]) & ok[:, 1:] & ok[:, :-1]
            ze = np.r_[zen[:, 1:][edge], zen[:, :-1][edge]]
            if ze.size:
                rec["flag_edge_zen_median"] = round(float(np.median(ze)), 2)
                rec["flag_edge_zen_p05"] = round(float(np.percentile(ze, 5)), 2)
                rec["flag_edge_zen_p95"] = round(float(np.percentile(ze, 95)), 2)
        else:
            flag = np.zeros(bits.shape, bool)

        # 2. coverage per region
        water_all = ok & ((bits & mask_bits(fmap, ["LAND"])) == 0)
        ref, _ = sw.band(555.0 if sw.kind == "Rrs" else 865.0, rs, cs)
        finite = np.isfinite(ref)
        regions = []
        for region in cfg.regions.values():
            inreg = region.contains(lon, lat) & water_all
            if not inreg.any():
                continue
            r = {"region": region.key, "n_water": int(inreg.sum()),
                 "hisatzen_fraction": round(float(flag[inreg].mean()), 4),
                 "zen_median": round(float(np.median(zen[inreg])), 2), "zen_max": round(float(zen[inreg].max()), 2)}
            for m in ("minimal", "l3_default", RELAXED):
                keep = inreg & finite & ((bits & mask_bits(fmap, MASKS[m])) == 0)
                r[f"valid_{m}"] = int(keep.sum())
            regions.append(r)
        rec["regions"] = regions

        # 4. reflectance vs zenith (water, valid under the relaxed mask, whole domain)
        keep = water_all & ((bits & mask_bits(fmap, MASKS[RELAXED])) == 0)
        zb = np.digitize(zen, ZEN_EDGES) - 1
        rng = np.random.default_rng(zlib.crc32(row["granule_key"].encode()))
        bands = {}
        for wl in TARGETS.get(sw.kind, []):
            v, wl_used = sw.band(float(wl), rs, cs)
            if abs(wl_used - wl) > 15:
                continue
            per = {}
            for b in range(len(ZEN_EDGES) - 1):
                sel = keep & (zb == b) & np.isfinite(v)
                n = int(sel.sum())
                if n == 0:
                    continue
                vals = v[sel]
                sample = vals if n <= SAMPLE_PER_BIN else rng.choice(vals, SAMPLE_PER_BIN, replace=False)
                per[b] = {"n": n, "neg": int((vals < 0).sum()), "sample": [round(float(x), 7) for x in sample]}
            bands[str(wl)] = {"wl_used": wl_used, "bins": per}
        rec["bands"] = bands
        rec["kind"] = sw.kind
    finally:
        sw.close()
    return rec


# ----------------------------------------------------------------------------- overlap pairs
def _binned(cfg: Config, path: Path, family: str, region: Region, wavelengths: list[int]):
    sw = L2Swath(path)
    try:
        win = sw.window(region)
        if win is None:
            return None
        rs, cs = win
        lon, lat = sw.lon[rs, cs], sw.lat[rs, cs]
        zen = sensor_zenith(sw, family, rs, cs)
        bits, fmap = sw.flags(rs, cs)
        ok = region.contains(lon, lat) & np.isfinite(zen) & ((bits & mask_bits(fmap, MASKS[RELAXED])) == 0)
        nx = int(round((region.east - region.west) / BIN_DEG))
        ny = int(round((region.north - region.south) / BIN_DEG))
        idx = (np.clip(((lat - region.south) / BIN_DEG).astype(int), 0, ny - 1) * nx
               + np.clip(((lon - region.west) / BIN_DEG).astype(int), 0, nx - 1))

        def mean(values, m):
            s = np.bincount(idx[m], weights=values[m], minlength=nx * ny)
            n = np.bincount(idx[m], minlength=nx * ny)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(n > 0, s / n, np.nan)

        out = {"zen": mean(zen.astype("f8"), ok)}
        for wl in wavelengths:
            v, wl_used = sw.band(float(wl), rs, cs)
            out[wl] = mean(v.astype("f8"), ok & np.isfinite(v)) if abs(wl_used - wl) <= 15 else None
        return out
    finally:
        sw.close()


def pair(args: tuple[str, dict, str]) -> list[dict[str, Any]]:
    """Both directions of one same-day pair: each sensor as target, the other as low-angle reference."""
    cfg_path, p, today = args
    cfg = load_config(cfg_path, date.fromisoformat(today))
    region = cfg.regions[p["region"]]
    wls = TARGETS["Rrs"]
    try:
        a = _binned(cfg, cfg.data_root / p["path_a"], p["family_a"], region, wls)
        b = _binned(cfg, cfg.data_root / p["path_b"], p["family_b"], region, wls)
    except Exception as exc:  # noqa: BLE001
        return [{"error": f"{type(exc).__name__}: {exc}", **{k: p[k] for k in ("path_a", "path_b")}}]
    if a is None or b is None:
        return []
    out = []
    for tgt, ref, tname, rname in ((a, b, p["product_a"], p["product_b"]), (b, a, p["product_b"], p["product_a"])):
        base = np.isfinite(tgt["zen"]) & (ref["zen"] < REFERENCE_MAX_ZEN)
        for lo, hi in OVERLAP_CLASSES:
            cls = base & (tgt["zen"] >= lo) & (tgt["zen"] < hi)
            for wl in wls:
                if tgt[wl] is None or ref[wl] is None:
                    continue
                m = cls & np.isfinite(tgt[wl]) & np.isfinite(ref[wl]) & (tgt[wl] > 0) & (ref[wl] > 0)
                if m.sum() < 5:
                    continue
                lr = np.log(tgt[wl][m] / ref[wl][m])
                out.append({"target": tname, "reference": rname, "region": p["region"], "date": p["date"],
                            "dt_h": round(p["dt_h"], 2), "zen_class": f"{lo}-{hi}", "band": wl,
                            "n_cells": int(m.sum()), "log_ratios": [round(float(x), 4) for x in lr[:500]]})
    return out


# ----------------------------------------------------------------------------- driver
def _spread(df, n: int):
    """n rows spread evenly over time (df sorted by time)."""
    if len(df) <= n:
        return df
    return df.iloc[np.linspace(0, len(df) - 1, n).round().astype(int)]


def run(cfg: Config, per_product: int = 40, max_pairs: int = 30, max_dt_h: float = 3.0,
        workers: int = 0, log=print) -> Path:
    import pandas as pd

    obs_path = cfg.data_root / "index" / "observations.csv"
    if not obs_path.exists():
        raise SystemExit("Build the observation index first: python -m oefm index")
    obs = pd.read_csv(obs_path)
    obs = obs.sort_values("time_start")
    out_dir = cfg.reports_dir / "zenith"
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(32, (os.cpu_count() or 2) - 1))

    # Granules: per product, spread over time, preferring ones with some valid water in a region.
    g = (obs[obs["valid_minimal"].fillna(0) > 0.05]
         .drop_duplicates("granule_key"))
    picks = pd.concat([_spread(d, per_product) for _, d in g.groupby("product")]) if len(g) else g
    jobs = [(str(cfg.path), str(cfg.data_root / r["path"]),
             {"granule_key": r["granule_key"], "product": r["product"], "time_start": r["time_start"],
              "local_path": r["path"]}, cfg.today.isoformat()) for _, r in picks.iterrows()]
    log(f"Zenith diagnostic: {len(jobs)} granules from {picks['product'].nunique() if len(picks) else 0} products")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(granule, jobs, chunksize=2))
    (out_dir / "granules.jsonl").write_text("\n".join(json.dumps(r, default=float) for r in recs) + "\n")

    # Same-day pairs of different platforms (PACE AOP only; SFREFL shares its geometry).
    rrs = obs[(obs["variable"] == "Rrs")].copy()
    rrs["t"] = pd.to_datetime(rrs["time_start"], utc=True)
    m = rrs.merge(rrs, on=["region", "date"], suffixes=("_a", "_b"))
    m = m[(m["product_a"] < m["product_b"]) & (m["platform_a"] != m["platform_b"])]
    m["dt_h"] = (m["t_a"] - m["t_b"]).abs().dt.total_seconds() / 3600
    m = m[m["dt_h"] <= max_dt_h]
    m["score"] = m[["valid_cover_minimal_a", "valid_cover_minimal_b"]].min(axis=1)
    m = m[m["score"] > 0.05]
    pairs = pd.concat([_spread(d.sort_values("t_a"), max_pairs) for _, d in m.groupby(["product_a", "product_b"])]) \
        if len(m) else m
    pjobs = [(str(cfg.path), {"region": r["region"], "date": r["date"], "dt_h": r["dt_h"],
                              "path_a": r["path_a"], "path_b": r["path_b"],
                              "product_a": r["product_a"], "product_b": r["product_b"],
                              "family_a": r["family_a"], "family_b": r["family_b"]}, cfg.today.isoformat())
             for _, r in pairs.iterrows()]
    log(f"Zenith overlap: {len(pjobs)} same-day pairs (dt <= {max_dt_h} h)")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        prs = [x for chunk in pool.map(pair, pjobs, chunksize=2) for x in chunk]

    with warnings.catch_warnings():
        # medians of all-empty columns (e.g. no HISATZEN edge for OLCI) are reported as blanks
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = summarise(recs, prs, out_dir)
    figures(cfg, recs, prs, out_dir)
    dest = out_dir / "zenith_summary.json"
    dest.write_text(json.dumps(summary, indent=1, default=float))
    log(f"Zenith diagnostic written to {out_dir}")
    return dest


# ----------------------------------------------------------------------------- summaries
def summarise(recs: list[dict], prs: list[dict], out_dir: Path) -> dict[str, Any]:
    import pandas as pd

    ok = [r for r in recs if "error" not in r]
    # 1. decode
    dec = pd.DataFrame([{k: r.get(k) for k in ("product", "hisatzen_bit", "l2gen_satzen", "l2gen_masksatzen",
                                                 "flagged_fraction_domain", "flag_matches_angle",
                                                 "flag_edge_zen_median", "flag_edge_zen_p05", "flag_edge_zen_p95",
                                                 "zen_max_domain")} for r in ok])
    decode = []
    if len(dec):
        for prod, d in dec.groupby("product"):
            decode.append({
                "product": prod, "granules": len(d),
                "hisatzen_bits": sorted({int(x) for x in d["hisatzen_bit"].dropna()}),
                "l2gen_satzen": sorted({float(x) for x in d["l2gen_satzen"].dropna()}),
                "l2gen_masksatzen": sorted({float(x) for x in d["l2gen_masksatzen"].dropna()}),
                "flag_matches_angle_median": _r(d["flag_matches_angle"].median()),
                "flag_matches_angle_min": _r(d["flag_matches_angle"].min()),
                "flag_edge_zen_median": _r(d["flag_edge_zen_median"].median(), 2),
                "granules_with_flag_edge": int(d["flag_edge_zen_median"].notna().sum()),
                "zen_max": _r(d["zen_max_domain"].max(), 1),
            })
    pd.DataFrame(decode).to_csv(out_dir / "decode_check.csv", index=False)

    # 2. coverage
    cov_rows = [{"product": r["product"], **x} for r in ok for x in r.get("regions", [])]
    cov = pd.DataFrame(cov_rows)
    coverage = []
    if len(cov):
        for (prod, reg), d in cov.groupby(["product", "region"]):
            w = d["n_water"].sum()
            l3, rel, mn = d["valid_l3_default"].sum(), d[f"valid_{RELAXED}"].sum(), d["valid_minimal"].sum()
            coverage.append({
                "product": prod, "region": reg, "granules": len(d), "water_pixels": int(w),
                "hisatzen_fraction": _r((d["hisatzen_fraction"] * d["n_water"]).sum() / max(w, 1)),
                "valid_l3_default": _r(l3 / max(w, 1)), f"valid_{RELAXED}": _r(rel / max(w, 1)),
                "valid_minimal": _r(mn / max(w, 1)),
                "recovered_x": _r(rel / l3, 2) if l3 else None,
                "granules_l3_empty_relaxed_not": int(((d["valid_l3_default"] == 0) & (d[f"valid_{RELAXED}"] > 0)).sum()),
            })
    pd.DataFrame(coverage).to_csv(out_dir / "coverage_by_sensor_region.csv", index=False)

    # 4. reflectance vs zenith
    pooled: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "neg": 0, "sample": []})
    for r in ok:
        for wl, info in r.get("bands", {}).items():
            for b, s in info["bins"].items():
                k = (r["product"], r.get("kind"), int(wl), int(b))
                pooled[k]["n"] += s["n"]
                pooled[k]["neg"] += s["neg"]
                pooled[k]["sample"] += s["sample"]
    rvz = []
    for (prod, kind, wl, b), s in sorted(pooled.items()):
        x = np.asarray(s["sample"])
        rvz.append({"product": prod, "variable": kind, "band": wl,
                    "zen_bin": f"{ZEN_EDGES[b]}-{ZEN_EDGES[b + 1]}", "zen_mid": (ZEN_EDGES[b] + ZEN_EDGES[b + 1]) / 2,
                    "n_pixels": s["n"], "negative_fraction": _r(s["neg"] / max(s["n"], 1)),
                    "p01": float(np.percentile(x, 1)), "p25": float(np.percentile(x, 25)),
                    "median": float(np.median(x)), "p75": float(np.percentile(x, 75)),
                    "p99": float(np.percentile(x, 99))})
    pd.DataFrame(rvz).to_csv(out_dir / "reflectance_vs_zenith.csv", index=False)

    # 5. overlap
    ov: dict[tuple, dict] = defaultdict(lambda: {"pairs": set(), "cells": 0, "lr": []})
    for p in prs:
        if "error" in p:
            continue
        k = (p["target"], p["reference"], p["band"], p["zen_class"])
        ov[k]["pairs"].add((p["date"], p["region"]))
        ov[k]["cells"] += p["n_cells"]
        ov[k]["lr"] += p["log_ratios"]
    orows = []
    for (t, ref, wl, zc), s in sorted(ov.items()):
        lr = np.asarray(s["lr"])
        orows.append({"target": t, "reference": ref, "band": wl, "target_zen_class": zc,
                      "pairs": len(s["pairs"]), "cells": s["cells"],
                      "median_ratio": _r(math.exp(np.median(lr)), 3),
                      "ratio_p25": _r(math.exp(np.percentile(lr, 25)), 3),
                      "ratio_p75": _r(math.exp(np.percentile(lr, 75)), 3)})
    odf = pd.DataFrame(orows)
    odf.to_csv(out_dir / "overlap_ratio_by_target_zenith.csv", index=False)
    # Angle effect: ratio in each class relative to the same target/reference/band when the target
    # views at < 45 deg. This divides out the sensors' fixed offset, leaving the angle dependence.
    effect = []
    low = {f"{lo}-{hi}" for lo, hi in OVERLAP_CLASSES if hi <= LOW_ANGLE_MAX}
    for (t, ref, wl), classes in _group(ov).items():
        base = [x for zc, s in classes.items() if zc in low for x in s["lr"]]
        if len(base) < 50:
            continue
        b0 = float(np.median(base))
        for zc, s in classes.items():
            effect.append({"target": t, "reference": ref, "band": wl, "target_zen_class": zc,
                           "pairs": len(s["pairs"]), "cells": s["cells"],
                           "ratio_vs_low": _r(math.exp(float(np.median(s["lr"])) - b0), 3)})
    edf = pd.DataFrame(effect).sort_values(["target", "band", "reference", "target_zen_class"]) if effect \
        else pd.DataFrame()
    edf.to_csv(out_dir / "angle_effect.csv", index=False)
    by_target = []
    if len(edf):
        for (t, wl, zc), d in edf.groupby(["target", "band", "target_zen_class"]):
            by_target.append({"target": t, "band": int(wl), "target_zen_class": zc, "references": len(d),
                              "pairs": int(d["pairs"].sum()), "cells": int(d["cells"].sum()),
                              "ratio_vs_low_median": _r(d["ratio_vs_low"].median(), 3),
                              "ratio_vs_low_min": _r(d["ratio_vs_low"].min(), 3),
                              "ratio_vs_low_max": _r(d["ratio_vs_low"].max(), 3)})
    pd.DataFrame(by_target).to_csv(out_dir / "angle_effect_by_target.csv", index=False)
    return {"granules": len(recs), "granule_errors": len(recs) - len(ok), "pairs_records": len(prs),
            "decode": decode, "coverage": coverage, "angle_effect_by_target": by_target}


def _group(ov: dict) -> dict:
    g: dict[tuple, dict] = defaultdict(dict)
    for (t, ref, wl, zc), s in ov.items():
        g[(t, ref, wl)][zc] = s
    return g


def _r(x, nd: int = 4):
    try:
        return None if x is None or not np.isfinite(x) else round(float(x), nd)
    except TypeError:
        return None


# ----------------------------------------------------------------------------- figures
def figures(cfg: Config, recs: list[dict], prs: list[dict], out_dir: Path) -> list[Path]:
    """Bare panels (no titles): per product a zenith map and reflectance map with the HISATZEN
    edge, reflectance vs zenith, and the overlap angle effect."""
    import pandas as pd

    from .inspect import _draw_regions, _mpl

    plt = _mpl()
    out: list[Path] = []
    ok = [r for r in recs if "error" not in r]
    for prod in sorted({r["product"] for r in ok}):
        cands = [r for r in ok if r["product"] == prod]
        best = max(cands, key=lambda r: (r.get("flag_edge_zen_median") is not None,
                                         sum(x["n_water"] * x["hisatzen_fraction"] for x in r.get("regions", []))))
        try:
            out += _map(cfg, best, out_dir / f"{prod}_example", plt, _draw_regions)
        except Exception:  # noqa: BLE001 - a missing map must not stop the diagnostic
            pass
    rvz = pd.read_csv(out_dir / "reflectance_vs_zenith.csv") if (out_dir / "reflectance_vs_zenith.csv").stat().st_size > 2 else None
    if rvz is not None and len(rvz):
        for prod, d in rvz.groupby("product"):
            bands = sorted(d["band"].unique())
            fig, axes = plt.subplots(1, len(bands), figsize=(2.2 * len(bands), 2.3), dpi=150, squeeze=False)
            for ax, wl in zip(axes[0], bands):
                x = d[d["band"] == wl].sort_values("zen_mid")
                ax.fill_between(x["zen_mid"], x["p25"], x["p75"], color="#2a7fbf", alpha=0.25, lw=0)
                ax.plot(x["zen_mid"], x["median"], color="#2a7fbf", lw=1.2, marker="o", ms=2.5)
                ax.axvline(60, color="k", lw=0.6, ls="--")
                ax.axhline(0, color="#888888", lw=0.5)
                ax.set_xlabel("sensor zenith (deg)")
                ax.set_ylabel(f"{x['variable'].iloc[0]} {wl} nm")
            fig.tight_layout()
            p = out_dir / f"{prod}_reflectance_vs_zenith.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            out.append(p)
    eff = out_dir / "angle_effect.csv"
    if eff.exists() and eff.stat().st_size > 2:
        e = pd.read_csv(eff)
        mids = {f"{lo}-{hi}": (lo + min(hi, 75)) / 2 for lo, hi in OVERLAP_CLASSES}
        for tgt, d in e.groupby("target"):
            bands = sorted(d["band"].unique())
            fig, axes = plt.subplots(1, len(bands), figsize=(2.2 * len(bands), 2.3), dpi=150, squeeze=False,
                                     sharey=True)
            for ax, wl in zip(axes[0], bands):
                for ref, x in d[d["band"] == wl].groupby("reference"):
                    x = x.assign(mid=x["target_zen_class"].map(mids)).sort_values("mid")
                    ax.plot(x["mid"], x["ratio_vs_low"], marker="o", ms=2.5, lw=1, label=ref)
                ax.axhline(1, color="k", lw=0.6)
                ax.axvline(60, color="k", lw=0.6, ls="--")
                ax.set_xlabel("target zenith (deg)")
                ax.set_ylabel(f"ratio vs < {LOW_ANGLE_MAX} deg, {wl} nm")
            axes[0][-1].legend(fontsize=6, frameon=False)
            fig.tight_layout()
            p = out_dir / f"{tgt}_overlap_angle_effect.png"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            out.append(p)
    return out


def _map(cfg: Config, rec: dict, stem: Path, plt, draw_regions) -> list[Path]:
    product = cfg.satellite_products[rec["product"]]
    sw = L2Swath(cfg.data_root / rec["path"])
    try:
        dom = cfg.domain()
        win = sw.window(dom)
        if win is None:
            return []
        rs, cs = win
        zen = sensor_zenith(sw, product["family"], rs, cs)
        bits, fmap = sw.flags(rs, cs)
        flag = ((bits & fmap.get("HISATZEN", 0)) != 0).astype("f4")
        ref, wl = sw.band(443.0 if sw.kind == "Rrs" else 865.0, rs, cs)
        land = (bits & mask_bits(fmap, ["LAND", "CLDICE"])) != 0
        ref = np.where(land, np.nan, ref)
        lat, lon = sw.lat[rs, cs], sw.lon[rs, cs]
        s = max(1, int(max(lat.shape) / 1200))
        out = []
        fin = np.isfinite(ref)
        for suffix, data, kw, label in [
            ("zenith", zen, {"cmap": "magma", "vmin": 0, "vmax": 75}, "sensor zenith (deg)"),
            (f"{sw.kind}{int(round(wl))}", ref,
             {"cmap": "viridis", "vmin": np.nanpercentile(ref, 2) if fin.any() else 0,
              "vmax": np.nanpercentile(ref, 98) if fin.any() else 1}, f"{sw.kind} {wl:.0f} nm"),
        ]:
            fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mesh = ax.pcolormesh(lon[::s, ::s], lat[::s, ::s], data[::s, ::s], shading="nearest",
                                     rasterized=True, **kw)
                if flag.any() and (~flag.astype(bool)).any():
                    ax.contour(lon[::s, ::s], lat[::s, ::s], flag[::s, ::s], levels=[0.5], colors="w",
                               linewidths=0.9)
            fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.01).set_label(label)
            draw_regions(ax, cfg)
            ax.set_xlim(dom.west, dom.east)
            ax.set_ylim(dom.south, dom.north)
            ax.set_aspect(1 / math.cos(math.radians((dom.south + dom.north) / 2)))
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            p = stem.with_name(f"{stem.name}_{suffix}.png")
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
            out.append(p)
        return out
    finally:
        sw.close()
