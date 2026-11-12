"""Write config/edna_cruise_windows.yaml: satellite windows around the SEUS MBON eDNA samples.

Every surface sample inside a pilot region, from the first historical year up to CONTEXT_DAYS
into the core window, gets a window from CONTEXT_DAYS before to AFTER_DAYS after its sampling day, so its
full causal optical context is downloaded. Sample days less than MERGE_GAP days apart become one
event (one per cruise, in practice). Later samples are covered by the core window.

The eDNA tables live outside this repo and are not on Ptolemy, so the YAML is committed. Rerun
only when the eDNA metadata changes:

    python scripts/make_edna_windows.py \
        --groups ../eDNA_data_sources/processed/16S_keyfield_eventdatezulu_materialsampleid_metadata_replicate_averaged.csv \
        --samples ../eDNA_data_sources/seusmbon_16S_metadata.tsv

NB the source TSV has decimalLatitude and decimalLongitude swapped (latitude holds ~-82). This
script detects that from the values and corrects it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from oefm.config import load_config  # noqa: E402

CONTEXT_DAYS = 30
AFTER_DAYS = 3
MERGE_GAP = 7


def sample_table(groups_csv: Path, samples_tsv: Path) -> pd.DataFrame:
    g = pd.read_csv(groups_csv, low_memory=False)
    raw = (pd.read_csv(samples_tsv, sep="\t", low_memory=False,
                       usecols=["samp_name", "decimalLatitude", "decimalLongitude"])
           .drop_duplicates("samp_name").set_index("samp_name"))
    first = g["sample_names"].str.split(";").str[0]
    a = pd.to_numeric(first.map(raw["decimalLatitude"]), errors="coerce")
    b = pd.to_numeric(first.map(raw["decimalLongitude"]), errors="coerce")
    swapped = a.abs().median() > 60 and b.abs().median() < 60  # "latitude" of ~-82 is a longitude
    g["lat"], g["lon"] = (b, a) if swapped else (a, b)
    g["day"] = pd.to_datetime(g["eventDateZulu"].astype(str).str[:10], errors="coerce").dt.date
    g["surface"] = g["depth_category"].astype(str).str.lower().str.startswith("surf")
    g.attrs["swapped"] = bool(swapped)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", required=True, type=Path)
    ap.add_argument("--samples", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "config" / "edna_cruise_windows.yaml")
    args = ap.parse_args()

    cfg = load_config(REPO / "config" / "pilot.yaml")
    t = cfg.raw["time"]
    first_year = min(t["historical"]["years"])
    core_start = date.fromisoformat(str(t["core"]["start"]))
    g = sample_table(args.groups, args.samples)
    inreg = pd.Series(False, index=g.index)
    for r in cfg.regions.values():
        inreg |= (g["lon"] >= r.west) & (g["lon"] <= r.east) & (g["lat"] >= r.south) & (g["lat"] <= r.north)
    sel = g[inreg & g["surface"] & g["day"].notna()]
    # Samples early in the core window still need context from before it.
    sel = sel[(sel["day"] >= date(first_year, 1, 1)) & (sel["day"] < core_start + timedelta(days=CONTEXT_DAYS))]
    days = sorted(set(sel["day"]))

    events, cur = [], None
    for d in days:
        if cur and (d - cur[1]).days <= MERGE_GAP:
            cur[1] = d
            cur[2] += int((sel["day"] == d).sum())
        else:
            if cur:
                events.append(cur)
            cur = [d, d, int((sel["day"] == d).sum())]
    if cur:
        events.append(cur)
    out = {
        "generated_by": "scripts/make_edna_windows.py",
        "source": {"groups": args.groups.name, "samples": args.samples.name,
                   "lat_lon_swapped_in_source": g.attrs["swapped"]},
        "rule": f"surface samples in pilot regions, {first_year}..{core_start + timedelta(days=CONTEXT_DAYS - 1)}; "
                f"window = first sample day - {CONTEXT_DAYS} d .. last sample day + {AFTER_DAYS} d; "
                f"sample days <= {MERGE_GAP} d apart merged",
        "events": [{"name": f"edna_{a:%Y%m%d}", "start": (a - timedelta(days=CONTEXT_DAYS)).isoformat(),
                    "end": (b + timedelta(days=AFTER_DAYS)).isoformat(),
                    "why": f"SEUS MBON eDNA cruise {a}..{b}, {n} surface sample groups"} for a, b, n in events],
    }
    args.out.write_text("# Generated; do not edit by hand (see scripts/make_edna_windows.py)\n"
                        + yaml.safe_dump(out, sort_keys=False, width=110))
    print(f"{len(sel)} samples on {len(days)} days -> {len(events)} events written to {args.out}")


if __name__ == "__main__":
    main()
