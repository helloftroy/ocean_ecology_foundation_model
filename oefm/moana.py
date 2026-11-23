"""NASA PACE MOANA phytoplankton community matchups: the baseline for the PACE phytoplankton test.

MOANA (Lange et al. 2020) estimates Prochlorococcus, Synechococcus and picoeukaryote cell
abundances (cells/mL) from the same OCI spectra the encoder sees. If a frozen embedding cannot
predict eDNA phytoplankton groups as well as MOANA does, it is losing PACE information.

Product: PACE_OCI_L4M_MOANA (OB_CLOUD), daily mapped 4 km files (~6 MB each, from 2024-03-05),
domain 70 S-70 N, 85 W-25 E (Atlantic only: samples west of 85 W get no matchup). Only the days
within --days of a PACE-era eDNA or SFER sample are downloaded, to raw/pace_oci_moana/.

Matchup per sample and day: median of the valid pixels (> 0, not fill) in the 3 x 3 pixels around
the sample (~12 km, as the 3 x 3 tile cells of the embeddings). Per window +/- k days (k = 1 and
--days): mean over the days with a value of log10(1 + cells/mL) for each group, and the number of
days used. -> <data_root>/processed/regional/eval/moana_matchups.csv.gz (keyed like the embeddings:
lat|lon|time_utc), read by oefm.model.focused (step phyto).

    python -m oefm.moana [--days 3] [--version 3.2]      # on 'service' (needs the internet)
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

SHORT_NAME = "PACE_OCI_L4M_MOANA"
PACE_START = date(2024, 3, 5)
# variable names: 3.1 and the data-structure page use the short form; later versions may spell them out
VARIABLES = {"pro": ("prococcus_moana", "prochlorococcus_moana"),
             "syn": ("syncoccus_moana", "synechococcus_moana"),
             "pico": ("picoeuk_moana", "picoeukaryotes_moana")}
DATASETS = ("edna_16S", "edna_18S", "sfer")


def sample_points(cfg):
    import pandas as pd

    from .targets import eval_dir

    frames = []
    for ds in DATASETS:
        f = eval_dir(cfg) / "matched" / f"{ds}.csv.gz"
        if f.exists():
            frames.append(pd.read_csv(f, dtype={"sample_id": str}, usecols=["dataset", "sample_id", "lat", "lon", "time_utc"]))
    if not frames:
        raise SystemExit("no matched evaluation sets: run targets.sbatch STEPS=evalsets first")
    pts = pd.concat(frames, ignore_index=True)
    pts["day"] = pd.to_datetime(pts["time_utc"], utc=True).dt.date
    pts = pts[pts["day"] >= PACE_START].reset_index(drop=True)
    pts["key"] = pts["lat"].round(5).astype(str) + "|" + pts["lon"].round(5).astype(str) + "|" + pts["time_utc"].astype(str)
    return pts


def needed_days(pts, k: int) -> list[date]:
    days = {d + timedelta(days=o) for d in pts["day"] for o in range(-k, k + 1)}
    return sorted(d for d in days if d >= PACE_START)


def find_granules(days: list[date], version: str | None, log=print) -> dict[date, tuple[str, str]]:
    """{day: (file name, https url)} for the daily 4 km files."""
    from .sources import nasa

    out: dict[date, tuple[str, str]] = {}
    prod = {"short_name": SHORT_NAME, "version": version, "provider": "OB_CLOUD"}
    # one CMR search per month spanned (fewer requests than per day)
    months = sorted({(d.year, d.month) for d in days})
    want = set(days)
    for y, m in months:
        start = date(y, m, 1)
        end = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
        for g in nasa.search(prod, None, start, end):
            links = [u for u in g.data_links(access="external") if u.endswith(".nc")]
            for u in links:
                name = u.rsplit("/", 1)[-1]
                if ".L4m.DAY." not in name or ".4km." not in name:
                    continue
                d = datetime.strptime(name.split(".")[1], "%Y%m%d").date()
                if d in want:
                    out[d] = (name, u)
    log(f"moana: {len(out)} of {len(days)} needed days have a daily 4 km file")
    return out


def download(files: dict[date, tuple[str, str]], dest: Path, log=print) -> dict[date, Path]:
    from .download import http_fetch
    from .sources import nasa

    dest.mkdir(parents=True, exist_ok=True)
    sess = nasa.session()
    got = {}
    for d, (name, url) in sorted(files.items()):
        p = dest / name
        if not p.exists():
            try:
                http_fetch(url, p, sess, 300.0, [])
            except Exception as exc:  # noqa: BLE001 - report and continue; rerun to retry
                log(f"moana: {name} failed ({exc})")
                continue
        got[d] = p
    log(f"moana: {len(got)} files in {dest}")
    return got


def _pick(ds, names):
    for n in names:
        if n in ds.variables:
            return n
    return None


def extract_day(path: Path, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """3 x 3 pixel median per sample (NaN outside the domain or without valid pixels)."""
    import netCDF4

    out = {g: np.full(len(lat), np.nan) for g in VARIABLES}
    with netCDF4.Dataset(path) as ds:
        la, lo = ds["lat"][:].astype(float), ds["lon"][:].astype(float)
        ii = np.abs(la[None, :] - lat[:, None]).argmin(1)
        jj = np.abs(lo[None, :] - lon[:, None]).argmin(1)
        inside = (lat >= la.min()) & (lat <= la.max()) & (lon >= lo.min()) & (lon <= lo.max())
        i0, i1 = max(int(ii.min()) - 1, 0), min(int(ii.max()) + 2, len(la))
        j0, j1 = max(int(jj.min()) - 1, 0), min(int(jj.max()) + 2, len(lo))
        for g, names in VARIABLES.items():
            v = _pick(ds, names)
            if v is None:
                continue
            ds[v].set_auto_mask(True)
            block = np.ma.filled(ds[v][i0:i1, j0:j1].astype(float), np.nan)
            block[block <= 0] = np.nan
            for s in np.where(inside)[0]:
                a, b = ii[s] - i0, jj[s] - j0
                w = block[max(a - 1, 0):a + 2, max(b - 1, 0):b + 2]
                if np.isfinite(w).any():
                    out[g][s] = np.nanmedian(w)
    return out


def matchups(pts, files: dict[date, Path], k: int):
    """Per sample: log10(1 + cells/mL) averaged over the days within +/- 1 and +/- k days."""
    lat, lon = pts["lat"].to_numpy(float), pts["lon"].to_numpy(float)
    per_day = {d: extract_day(p, lat, lon) for d, p in sorted(files.items())}
    res = pts[["dataset", "sample_id", "lat", "lon", "time_utc", "key"]].copy()
    for win in sorted({1, k}):
        nd = np.zeros(len(pts), int)
        for g in VARIABLES:
            vals = np.full((len(pts), 2 * win + 1), np.nan)
            for s, d0 in enumerate(pts["day"]):
                for j, o in enumerate(range(-win, win + 1)):
                    d = d0 + timedelta(days=o)
                    if d in per_day:
                        vals[s, j] = per_day[d][g][s]
            with np.errstate(all="ignore"):
                lv = np.log10(1 + vals)
                res[f"moana_{g}_pm{win}"] = np.nanmean(lv, axis=1)
            nd = np.maximum(nd, np.isfinite(vals).sum(1))
        res[f"moana_ndays_pm{win}"] = nd
    return res


def main(argv=None) -> None:
    from .config import load_config
    from .targets import eval_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=3, help="matchup half-window (days); +/- 1 d is always added")
    ap.add_argument("--version", default="3.2", help="collection version (CMR listed only 3.2 on 2026-11-23)")
    args = ap.parse_args(argv)
    cfg = load_config()
    pts = sample_points(cfg)
    print(f"moana: {len(pts)} PACE-era samples ({pts.groupby('dataset').size().to_dict()})", flush=True)
    files = find_granules(needed_days(pts, args.days), args.version)
    got = download(files, cfg.data_root / "raw" / "pace_oci_moana")
    out = matchups(pts, got, args.days)
    dest = eval_dir(cfg) / "moana_matchups.csv.gz"
    out.to_csv(dest, index=False)
    for win in sorted({1, args.days}):
        ok = out[f"moana_ndays_pm{win}"] > 0
        print(f"moana +/-{win} d: {int(ok.sum())} of {len(out)} samples with a matchup; by dataset "
              f"{out[ok].groupby('dataset').size().to_dict()}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
