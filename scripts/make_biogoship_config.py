"""Generate config/biogoship.yaml: zones, corridor regions and windows around the Bio-GO-SHIP samples.

Input: the benchmark bundle's samples.csv (biogoship_granite_kegg/data/results/benchmark_export).
Every sample with a time and a position is covered (the 1,416 paired-benchmark samples and the
extended set).

- Zones: per cruise, consecutive samples (by time) while all lie within --zone-km of the zone's
  centroid. Each zone is one local LAEA projection (scale error < 1 % within 1,500 km).
- Regions (corridor boxes): consecutive samples of a zone while the box diagonal stays under
  --chunk-km and the time span under --chunk-days, never across the antimeridian. Box = the
  samples' extent + --margin-deg (latitude; longitude scaled by 1 / cos(lat)).
- Windows per region:
    cruise:   first sample day - --before days to last sample day + --after days (the causal
              30 d and retrospective +/- 15 d windows of every sample in the box);
    season_1, season_2: --season-days each, starting 4 and 8 months after the zone's median
              sample day, moved to the PACE era (2024-03-05 to 2025-12-31, all four sensor
              families) and never overlapping the zone's cruise windows.
  Imagery only: no label enters pretraining (decision log, "PACE era vs the 2024 hold-out year").

    python scripts/make_biogoship_config.py --samples <bundle>/samples.csv [--out config/biogoship.yaml]
"""

from __future__ import annotations

import argparse
import math
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PACE_START, PACE_END = date(2024, 3, 5), date(2025, 12, 31)
R_KM = 6371.0


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def centroid(lat, lon) -> tuple[float, float]:
    v = unit(np.asarray(lat), np.asarray(lon)).mean(0)
    v /= np.linalg.norm(v)
    return float(np.degrees(np.arcsin(v[2]))), float(np.degrees(np.arctan2(v[1], v[0])))


def gc_km(lat0, lon0, lat, lon) -> np.ndarray:
    a = unit(np.atleast_1d(lat), np.atleast_1d(lon)) @ unit([lat0], [lon0])[0]
    return R_KM * np.arccos(np.clip(a, -1, 1))


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def zones_for(c) -> list[np.ndarray]:
    """Index arrays (into the cruise frame, time-sorted) of consecutive samples per zone."""
    out, cur = [], [0]
    for i in range(1, len(c)):
        trial = cur + [i]
        la, lo = centroid(c["lat"].iloc[trial], c["lon"].iloc[trial])
        if gc_km(la, lo, c["lat"].iloc[trial], c["lon"].iloc[trial]).max() <= ARGS.zone_km:
            cur = trial
        else:
            out.append(np.array(cur))
            cur = [i]
    out.append(np.array(cur))
    return out


def chunks_for(z) -> list[np.ndarray]:
    out, cur = [], [0]
    for i in range(1, len(z)):
        trial = cur + [i]
        la, lo = z["lat"].iloc[trial].to_numpy(), z["lon"].iloc[trial].to_numpy()
        wraps = lo.max() - lo.min() > 180
        diag = gc_km(la.min(), lo.min(), la.max(), lo.max())[0] if not wraps else np.inf
        span = (z["day"].iloc[trial].max() - z["day"].iloc[trial].min()).days
        if diag <= ARGS.chunk_km and span <= ARGS.chunk_days:
            cur = trial
        else:
            out.append(np.array(cur))
            cur = [i]
    out.append(np.array(cur))
    return out


def box(lat, lon) -> list[float]:
    m = ARGS.margin_deg
    s, n = max(-90.0, float(lat.min()) - m), min(90.0, float(lat.max()) + m)
    mid = max(abs(s), abs(n))
    if mid >= 85:
        return [-180.0, round(s, 3), 180.0, round(n, 3)]
    ml = m / max(math.cos(math.radians(mid)), 0.05)
    w, e = max(-180.0, float(lon.min()) - ml), min(180.0, float(lon.max()) + ml)
    return [round(w, 3), round(s, 3), round(e, 3), round(n, 3)]


def season_window(start: date, cruise: list[tuple[date, date]]) -> tuple[date, date]:
    """The same month-day in the first PACE-era year where the window fits and avoids the cruise."""
    for year in (2024, 2025):
        try:
            d0 = start.replace(year=year)
        except ValueError:  # 29 February
            d0 = start.replace(year=year, day=28)
        d1 = d0 + timedelta(days=ARGS.season_days - 1)
        if d0 >= PACE_START and d1 <= PACE_END and all(d1 < a or d0 > b for a, b in cruise):
            return d0, d1
    d0 = PACE_START
    return d0, d0 + timedelta(days=ARGS.season_days - 1)


def main(argv=None) -> None:
    global ARGS
    import pandas as pd
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", default=str(REPO / "config" / "biogoship.yaml"))
    ap.add_argument("--zone-km", type=float, default=1500.0)
    ap.add_argument("--chunk-km", type=float, default=300.0)
    ap.add_argument("--chunk-days", type=int, default=7)
    ap.add_argument("--margin-deg", type=float, default=0.3)
    ap.add_argument("--before", type=int, default=30)
    ap.add_argument("--after", type=int, default=15)
    ap.add_argument("--season-days", type=int, default=45)
    ARGS = ap.parse_args(argv)

    s = pd.read_csv(ARGS.samples)
    s["t"] = pd.to_datetime(s["collection_time_utc"], utc=True, errors="coerce")
    s = s[s["t"].notna() & s["lat"].notna() & s["lon"].notna()].copy()
    s["day"] = s["t"].dt.date
    zones, regions = {}, {}
    rows = []
    for cruise, c in s.sort_values("t").groupby("cruise", sort=False):
        c = c.reset_index(drop=True)
        for zi, zix in enumerate(zones_for(c), 1):
            z = c.iloc[zix].reset_index(drop=True)
            zkey = f"{slug(cruise)}_z{zi}"
            la0, lo0 = centroid(z["lat"], z["lon"])
            zones[zkey] = {"lat_0": round(la0, 3), "lon_0": round(lo0, 3), "cruise": cruise,
                           "n_samples": int(len(z)), "radius_km": round(float(gc_km(la0, lo0, z["lat"], z["lon"]).max()), 1)}
            chunks = chunks_for(z)
            cruise_w = [(z["day"].iloc[ix].min() - timedelta(days=ARGS.before),
                         z["day"].iloc[ix].max() + timedelta(days=ARGS.after)) for ix in chunks]
            mid = sorted(z["day"])[len(z) // 2]
            seasons = [season_window(mid + timedelta(days=d), cruise_w) for d in (122, 243)]
            for ci, (ix, (a, b)) in enumerate(zip(chunks, cruise_w), 1):
                ch = z.iloc[ix]
                rkey = f"{zkey}_r{ci:02d}"
                wins = [{"name": "cruise", "start": a.isoformat(), "end": b.isoformat()}]
                wins += [{"name": f"season_{k}", "start": d0.isoformat(), "end": d1.isoformat()}
                         for k, (d0, d1) in enumerate(seasons, 1)]
                regions[rkey] = {"name": f"{cruise} corridor {zi}.{ci}", "zone": zkey,
                                 "bbox": box(ch["lat"].to_numpy(), ch["lon"].to_numpy()), "windows": wins,
                                 "notes": f"{len(ch)} samples, {ch['day'].min()} to {ch['day'].max()}"}
                rows.append({"cruise": cruise, "zone": zkey, "region": rkey, "n": len(ch),
                             "cruise_days": (b - a).days + 1})
    cfg = {
        "extends": "pilot.yaml",
        "project": "ocean_ecology_foundation_model",
        "phase": "biogoship_global",
        "data_root": "data_biogoship",
        "domain_margin_deg": 0.5,
        "time": {"physical_lead_days": 0},
        "physical": {},
        "targets": {},
        "processing": {"stats_dir": "data/processed/regional/tiles/v2/train_stats"},
        "runs_root": "data/runs",
        "zones": zones,
        "regions": regions,
    }
    head = (f"# Generated by scripts/make_biogoship_config.py from {Path(ARGS.samples).name} "
            f"({len(s):,} samples with time and position). Do not edit by hand; rerun the script.\n"
            "# Satellite products, processing and training settings come from pilot.yaml (extends).\n"
            "# Stage 1 is optical only, so no physical products here. GEBCO: link data/raw/gebco into\n"
            "# data_biogoship/raw/. Normalisation statistics and training runs: Florida's (stats_dir, runs_root).\n")
    Path(ARGS.out).write_text(head + yaml.safe_dump(cfg, sort_keys=False, width=120))
    r = pd.DataFrame(rows)
    print(f"{len(s):,} samples, {s['cruise'].nunique()} cruises -> {len(zones)} zones, {len(regions)} regions; "
          f"max zone radius {max(z['radius_km'] for z in zones.values()):.0f} km")
    print(r.groupby("cruise").agg(zones=("zone", "nunique"), regions=("region", "size"), samples=("n", "sum"),
                                  cruise_window_days=("cruise_days", "sum")).to_string())
    print(f"-> {ARGS.out}")


ARGS = None

if __name__ == "__main__":
    main()
