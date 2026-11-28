"""Granite (ibm-granite/granite-geospatial-ocean) eDNA embeddings -> external feature sets for the probes.

The Granite bundle (multimodal_granite/eDNA_processing/edna_physchem_granite_embeddings_bundle.zip) has,
per eDNA event, one 512-d embedding: the encoder's last hidden state mean-pooled over a whole 42 x 42
Sentinel-3 OLCI tile (300 m, ~12.6 km), the tile nearest in time (<= 5 d) and within 25 km of the sample
(metadata: match_day_gap, match_distance_km). `edna_event_id` equals our eDNA `sample_id`.

Writes two npz files for `python -m oefm.model.evaluate --extra-features ...`, keyed like the probes
(lat|lon|time_utc of the evaluation targets):
  granite_tile.npz    "granite/tile": every matched sample, as the Granite work used it;
  granite_strict.npz  "granite/strict": only samples inside their tile (distance <= --max-km, default
                      6.3 km = half the tile width) and <= --max-days (default 3, our retro3 window)
                      from the overpass; other samples NaN, so every feature set in that run is scored
                      on the same strict subset.
Prints the counts per dataset and split.

    python scripts/granite_features.py --bundle ../multimodal_granite/eDNA_processing/edna_physchem_granite_embeddings_bundle.zip \
        --targets ../eval_targets --out ../eval_results/granite
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--targets", required=True, help="folder with edna_16S.csv.gz, edna_18S.csv.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-km", type=float, default=6.3)
    ap.add_argument("--max-days", type=float, default=3.0)
    args = ap.parse_args(argv)
    with zipfile.ZipFile(args.bundle) as zf:
        meta = pd.read_csv(io.BytesIO(zf.read("metadata.csv")), dtype={"edna_event_id": str})
        emb = np.load(io.BytesIO(zf.read("embeddings.npy")))
    if len(meta) != emb.shape[0]:
        raise SystemExit(f"metadata rows {len(meta)} != embeddings {emb.shape[0]}")
    row_of = {s: i for i, s in enumerate(meta["edna_event_id"])}
    frames = []
    for f in sorted(Path(args.targets).glob("edna_*.csv.gz")):
        if "_components" in f.name:
            continue
        t = pd.read_csv(f, dtype={"sample_id": str}, usecols=["sample_id", "dataset", "lat", "lon", "time_utc", "label_split"])
        frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    t["key"] = t["lat"].round(5).astype(str) + "|" + t["lon"].round(5).astype(str) + "|" + t["time_utc"].astype(str)
    t["row"] = t["sample_id"].map(row_of)
    t = t[t["row"].notna()].copy()
    t["row"] = t["row"].astype(int)
    t["km"] = meta["match_distance_km"].to_numpy()[t["row"]]
    t["days"] = meta["match_day_gap"].to_numpy()[t["row"]]
    t["strict"] = (t["km"] <= args.max_km) & (t["days"] <= args.max_days)
    u = t.drop_duplicates("key")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X = emb[u["row"].to_numpy()].astype("f4")
    np.savez(out / "granite_tile.npz", key=u["key"].to_numpy().astype(str), datasets=np.array(sorted(t["dataset"].unique())), **{"granite/tile": X})
    Xs = X.copy()
    Xs[~u["strict"].to_numpy()] = np.nan
    np.savez(out / "granite_strict.npz", key=u["key"].to_numpy().astype(str), datasets=np.array(sorted(t["dataset"].unique())), **{"granite/strict": Xs})
    print(f"Granite: {len(meta):,} events, dim {emb.shape[1]}; matched to {len(u):,} evaluation keys")
    print(f"strict: distance <= {args.max_km} km and gap <= {args.max_days} d")
    print(t.groupby(["dataset", "label_split"]).agg(samples=("key", "size"), strict=("strict", "sum"),
                                                     km_median=("km", "median"), days_median=("days", "median"))
          .round(2).to_string())
    print(f"-> {out / 'granite_tile.npz'}, {out / 'granite_strict.npz'}")


if __name__ == "__main__":
    main()
