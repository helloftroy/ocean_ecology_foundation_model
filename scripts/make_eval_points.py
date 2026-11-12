"""Write config/eval_points.csv: one row per evaluation observation (where and when), for counting
which evaluation samples have usable satellite (especially PACE) imagery on Ptolemy.

The evaluation tables live outside this repo and are not on Ptolemy, so the compact point list is
generated on the laptop and committed (coordinates rounded to 0.001 deg, ~100 m, well inside one
1.2 km cell). Rerun when an evaluation table changes:

    python scripts/make_eval_points.py --root ..

Datasets: SEUS MBON 16S and 18S surface samples (eDNA; lat/lon swapped in the source TSVs and
corrected here), SFER bottle samples <= 5 m with nutrients or chlorophyll, NOAA CRCP reef-fish
visual census (one row per primary sample unit), and the GF video MaxN events. Only rows inside
a pilot region and from 2018 on are kept.

The repository is public. Datasets in PRIVATE (the GF video MaxN events, whose sharing status is
not established) are written to config/eval_points_private.csv instead, which is gitignored: copy
it to the same place on Ptolemy by hand (scp). `python -m oefm evalcover` reads both files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from make_edna_windows import sample_table  # noqa: E402
from oefm.config import load_config  # noqa: E402

PRIVATE = {"fish_maxn"}


def edna(root: Path, marker: str) -> pd.DataFrame:
    g = sample_table(root / f"eDNA_data_sources/processed/{marker}_keyfield_eventdatezulu_materialsampleid_metadata_replicate_averaged.csv",
                     root / f"eDNA_data_sources/seusmbon_{marker}_metadata.tsv")
    g = g[g["surface"]]
    return pd.DataFrame({"dataset": f"edna_{marker}", "sample_id": g["group_id"].astype(str), "lat": g["lat"],
                         "lon": g["lon"], "time_utc": pd.to_datetime(g["eventDateZulu"], utc=True, errors="coerce")})


def sfer(root: Path) -> pd.DataFrame:
    s = pd.read_csv(root / "physiochem_data_sources/SFER_physicochem_clean_wide.csv", low_memory=False)
    s = s[pd.to_numeric(s["depth"], errors="coerce") <= 5]
    has = s[["po4", "no3", "si", "chl"]].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    s = s[has]
    return pd.DataFrame({"dataset": "sfer", "sample_id": s["keyfield"].astype(str), "lat": s["lat_dec"],
                         "lon": s["lon_dec"], "time_utc": pd.to_datetime(s["datetime"], utc=True, errors="coerce")})


def crcp(root: Path) -> pd.DataFrame:
    c = pd.read_csv(root / "eDNA_data_sources/fish_other_data_sources/CRCP_Reef_Fish_Surveys_Florida_0941_6202_3050.csv",
                    usecols=["time", "latitude", "longitude", "PRIMARY_SAMPLE_UNIT"], low_memory=False)
    c = c[c["time"] != "UTC"]  # ERDDAP units row
    c = c.drop_duplicates(["PRIMARY_SAMPLE_UNIT", "time"])
    return pd.DataFrame({"dataset": "fish_crcp", "sample_id": c["PRIMARY_SAMPLE_UNIT"].astype(str) + "_" + c["time"].str[:10],
                         "lat": pd.to_numeric(c["latitude"], errors="coerce"), "lon": pd.to_numeric(c["longitude"], errors="coerce"),
                         "time_utc": pd.to_datetime(c["time"], utc=True, errors="coerce")})


def maxn(root: Path) -> pd.DataFrame:
    m = pd.read_csv(root / "multimodal_granite/fish/gf_maxn_top10_granite_manifest_nmf_ecology.csv",
                    usecols=["fish_event_id", "fish_time", "fish_lat", "fish_lon"]).drop_duplicates("fish_event_id")
    return pd.DataFrame({"dataset": "fish_maxn", "sample_id": m["fish_event_id"].astype(str), "lat": m["fish_lat"],
                         "lon": m["fish_lon"], "time_utc": pd.to_datetime(m["fish_time"], utc=True, errors="coerce")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="folder holding eDNA_data_sources, physiochem_data_sources, multimodal_granite")
    ap.add_argument("--out", type=Path, default=REPO / "config" / "eval_points.csv")
    args = ap.parse_args()
    cfg = load_config(REPO / "config" / "pilot.yaml")
    parts = []
    for fn in (lambda r: edna(r, "16S"), lambda r: edna(r, "18S"), sfer, crcp, maxn):
        try:
            parts.append(fn(args.root))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"skipped a dataset: {type(exc).__name__}: {exc}")
    df = pd.concat(parts, ignore_index=True).dropna(subset=["lat", "lon", "time_utc"])
    df = df[df["time_utc"] >= pd.Timestamp("2018-01-01", tz="UTC")]
    inreg = pd.Series(False, index=df.index)
    for r in cfg.regions.values():
        inreg |= (df["lon"] >= r.west) & (df["lon"] <= r.east) & (df["lat"] >= r.south) & (df["lat"] <= r.north)
    df = df[inreg].drop_duplicates(["dataset", "sample_id"])
    df["lat"], df["lon"] = df["lat"].round(3), df["lon"].round(3)
    df["time_utc"] = df["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.sort_values(["dataset", "time_utc"])
    priv = df["dataset"].isin(PRIVATE)
    df[~priv].to_csv(args.out, index=False)
    if priv.any():
        df[priv].to_csv(args.out.with_name(args.out.stem + "_private.csv"), index=False)
    print(df.groupby("dataset").size().to_string())
    print(f"{len(df):,} points -> {args.out}")


if __name__ == "__main__":
    main()
