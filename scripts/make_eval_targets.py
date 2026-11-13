"""Build the downstream evaluation targets with label splits fixed BEFORE any training (step 2.7).

Runs on the laptop (the label tables are not on Ptolemy) and writes one CSV per dataset plus
fit_info.json into --out (default ../eval_targets/, outside the public repo). Copy that folder to
data/processed/regional/eval/ on Ptolemy (cluster_instructions.md, step 2.7).

Label split per dataset: config training.partitions.labels (eDNA and fish: test from 2024;
SFER: test from 2025). Every fitted transformation uses TRAINING labels only and is then applied
unchanged to the test labels:
- eDNA (16S, 18S; surface samples in the pilot regions, 2018+): ASV prevalence filter (>= 2% of
  training samples), Hellinger transform (sqrt of relative abundance), NMF for each k in --k
  (sklearn, nndsvda, seed 7, cd/Frobenius, max_iter 2000, tol 1e-4: the settings of
  multimodal_granite/eDNA_processing/run_hellinger_nmf.py), PCA of the Hellinger matrix
  (composition), and the dominant genera (top --genera by training mean relative abundance).
- SFER (<= 5 m): PO4, NO3, Si, fluorometric Chl as measured (no fitting).
- Fish CRCP: per survey unit, the --species most prevalent species in training: presence and
  log1p(count). Fish MaxN: the top-10 species columns of the Granite manifest (as already chosen
  there; month-precision dates).

Rerun when a source table changes:  python scripts/make_eval_targets.py --root ..
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from oefm.config import load_config  # noqa: E402
from oefm.loader import label_split  # noqa: E402

PREVALENCE = 0.02
NMF_KW = {"init": "nndsvda", "solver": "cd", "beta_loss": "frobenius", "max_iter": 2000, "tol": 1e-4,
          "random_state": 7}


def points(root_repo: Path) -> pd.DataFrame:
    files = [root_repo / "config" / "eval_points.csv", root_repo / "config" / "eval_points_private.csv"]
    return pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)


def with_split(cfg, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = df.copy()
    df["label_split"] = [label_split(cfg, dataset, t) for t in df["time_utc"]]
    return df


def genus(tax: str) -> str:
    """Genus, or the deepest informative rank above it. SILVA-style 16S ('g__Synechococcus') and
    PR2-style 18S (Domain;Supergroup;Division;Subdivision;Class;Order;Family;Genus;Species, with
    '_X' placeholders) are both handled."""
    if "__" not in str(tax):
        ranks = [r.strip() for r in str(tax).replace(":", ";").split(";")][:8]
        names = ["d", "sg", "dv", "sd", "c", "o", "f", "g"]
        for i in range(len(ranks) - 1, 0, -1):
            v = ranks[i]
            if v and not v.endswith("_X") and not v.endswith("_XX") and v.lower() not in ("unknown", "plas"):
                return f"{names[i]}__{v}"
        return "unassigned"
    parts = dict(p.split("__", 1) for p in str(tax).split(";") if "__" in p)
    for rank in ("g", "f", "o", "c", "p"):
        v = parts.get(rank, "").strip()
        if v and not v.lower().startswith("uncultured") and v.lower() not in ("unknown", "unassigned"):
            return f"{rank}__{v}"
    return "unassigned"


def edna(cfg, root: Path, pts: pd.DataFrame, marker: str, ks: list[int], n_genera: int, n_pcs: int):
    from sklearn.decomposition import NMF

    ds = f"edna_{marker}"
    meta = with_split(cfg, pts[pts["dataset"] == ds], ds).set_index("sample_id")
    tab = pd.read_csv(root / f"eDNA_data_sources/processed/{marker}_keyfield_eventdatezulu_materialsampleid_by_asv_mean_counts_prevalence_ge2pct.tsv.gz",
                      sep="\t", index_col=0)
    cols = [c for c in tab.columns if c in meta.index]
    X = tab[cols].T.astype("f8")  # samples x ASVs
    meta = meta.loc[X.index]
    train = (meta["label_split"] == "train").to_numpy()
    rel = X.div(X.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    prev = (X[train] > 0).mean(axis=0)
    keep = prev[prev >= PREVALENCE].index
    Xk = X[keep]
    relk = Xk.div(Xk.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    H = np.sqrt(relk.to_numpy())
    out = meta[["dataset", "lat", "lon", "time_utc", "label_split"]].copy()
    info = {"dataset": ds, "samples": int(len(X)), "train": int(train.sum()), "test": int((~train).sum()),
            "asvs_in_source": int(X.shape[1]), "asvs_kept_train_prevalence": int(len(keep)), "nmf": {}}
    comp_dir = {}
    for k in ks:
        m = NMF(n_components=k, **NMF_KW)
        W_train = m.fit_transform(H[train])
        W = np.zeros((len(H), k))
        W[train] = W_train
        W[~train] = m.transform(H[~train])
        for c in range(k):
            out[f"nmf_k{k}_{c + 1}"] = W[:, c]
        info["nmf"][k] = {"reconstruction_err_train": float(m.reconstruction_err_), "n_iter": int(m.n_iter_)}
        comp_dir[k] = pd.DataFrame(m.components_.T, index=keep, columns=[f"NMF{c + 1}" for c in range(k)])
    # composition: PCA of the Hellinger matrix, fitted on training samples
    mu = H[train].mean(axis=0)
    _, sv, vt = np.linalg.svd(H[train] - mu, full_matrices=False)
    P = (H - mu) @ vt[:n_pcs].T
    for c in range(n_pcs):
        out[f"pc_{c + 1}"] = P[:, c]
    info["pca_explained_variance_train"] = [float(x) for x in (sv[:n_pcs] ** 2 / (sv ** 2).sum())]
    # dominant genera: relative abundance (of all ASVs in the source table), top by training mean
    tax = pd.read_csv(root / f"eDNA_data_sources/seusmbon_P01to41_{marker}_taxonomy.tsv", sep="\t", index_col=0)["Taxon"]
    g = rel.T.groupby(tax.reindex(rel.columns).fillna("").map(genus)).sum().T
    g = g.drop(columns=["unassigned"], errors="ignore")
    top = g[train].mean(axis=0).sort_values(ascending=False).index[:n_genera]
    for name in top:
        out[f"taxon_{name}"] = g[name].to_numpy()
    info["dominant_genera"] = list(top)
    out.index.name = "sample_id"
    return out.reset_index(), info, comp_dir


def sfer(cfg, root: Path, pts: pd.DataFrame):
    s = pd.read_csv(root / "physiochem_data_sources/SFER_physicochem_clean_wide.csv", low_memory=False)
    s = s.drop_duplicates("keyfield").set_index("keyfield")
    p = with_split(cfg, pts[pts["dataset"] == "sfer"], "sfer").set_index("sample_id")
    out = p[["dataset", "lat", "lon", "time_utc", "label_split"]].copy()
    for v in ("po4", "no3", "si", "chl"):
        out[v] = pd.to_numeric(s.reindex(out.index)[v], errors="coerce")
    out.index.name = "sample_id"
    return out.reset_index(), {"dataset": "sfer", "samples": int(len(out)),
                               "train": int((out["label_split"] == "train").sum()),
                               "test": int((out["label_split"] == "test").sum()),
                               "non_missing": {v: int(out[v].notna().sum()) for v in ("po4", "no3", "si", "chl")}}


def crcp(cfg, root: Path, pts: pd.DataFrame, n_species: int):
    c = pd.read_csv(root / "eDNA_data_sources/fish_other_data_sources/CRCP_Reef_Fish_Surveys_Florida_0941_6202_3050.csv",
                    usecols=["time", "PRIMARY_SAMPLE_UNIT", "SCIENTIFIC_NAME", "NUM"], low_memory=False)
    c = c[c["time"] != "UTC"]
    c["sample_id"] = c["PRIMARY_SAMPLE_UNIT"].astype(str) + "_" + c["time"].str[:10]
    c["NUM"] = pd.to_numeric(c["NUM"], errors="coerce").fillna(0)
    counts = c.pivot_table(index="sample_id", columns="SCIENTIFIC_NAME", values="NUM", aggfunc="sum", fill_value=0)
    p = with_split(cfg, pts[pts["dataset"] == "fish_crcp"], "fish_crcp").set_index("sample_id")
    counts = counts.reindex(p.index).fillna(0)
    train = (p["label_split"] == "train").to_numpy()
    top = (counts[train] > 0).mean(axis=0).sort_values(ascending=False).index[:n_species]
    out = p[["dataset", "lat", "lon", "time_utc", "label_split"]].copy()
    for sp in top:
        key = sp.replace(" ", "_")
        out[f"present_{key}"] = (counts[sp] > 0).astype(int).to_numpy()
        out[f"log1p_{key}"] = np.log1p(counts[sp].to_numpy())
    out.index.name = "sample_id"
    return out.reset_index(), {"dataset": "fish_crcp", "samples": int(len(out)), "train": int(train.sum()),
                               "test": int((~train).sum()), "species": list(top)}


def maxn(cfg, root: Path, pts: pd.DataFrame):
    m = pd.read_csv(root / "multimodal_granite/fish/gf_maxn_top10_granite_manifest_nmf_ecology.csv").drop_duplicates("fish_event_id")
    meta_cols = {"tile_path", "tile_date", "tile_lat", "tile_lon", "fish_event_id", "fish_time", "fish_lat", "fish_lon",
                 "distance_to_shore_km", "log10_distance_to_shore_km", "ocean_depth_m", "log10_ocean_depth_m",
                 "match_day_gap", "match_distance_km", "LAB", "REFERENCE"}
    species = [c for c in m.columns if c not in meta_cols and "_" in c and c[0].isupper()][:10]
    m = m.set_index(m["fish_event_id"].astype(str))
    p = with_split(cfg, pts[pts["dataset"] == "fish_maxn"], "fish_maxn").set_index("sample_id")
    out = p[["dataset", "lat", "lon", "time_utc", "label_split"]].copy()
    for sp in species:
        v = pd.to_numeric(m.reindex(out.index)[sp], errors="coerce")
        out[f"maxn_{sp}"] = v
    out.index.name = "sample_id"
    return out.reset_index(), {"dataset": "fish_maxn", "samples": int(len(out)),
                               "train": int((out["label_split"] == "train").sum()),
                               "test": int((out["label_split"] == "test").sum()), "species": species,
                               "time_precision": "month"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/eval_targets")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 15, 20])
    ap.add_argument("--genera", type=int, default=20)
    ap.add_argument("--pcs", type=int, default=10)
    ap.add_argument("--species", type=int, default=20)
    args = ap.parse_args()
    out = args.out or args.root / "eval_targets"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO / "config" / "pilot.yaml")
    pts = points(REPO)
    info = {"created": datetime.now(timezone.utc).isoformat(), "label_partitions": cfg.raw["training"]["partitions"]["labels"],
            "nmf_settings": NMF_KW, "prevalence_filter_train": PREVALENCE, "datasets": {}}
    for marker in ("16S", "18S"):
        df, inf, comps = edna(cfg, args.root, pts, marker, args.k, args.genera, args.pcs)
        df.to_csv(out / f"edna_{marker}.csv.gz", index=False)
        for k, c in comps.items():
            c.to_csv(out / f"edna_{marker}_nmf_k{k}_components.csv.gz")
        info["datasets"][f"edna_{marker}"] = inf
        print(f"edna_{marker}: {inf['samples']} samples ({inf['train']} train / {inf['test']} test), "
              f"{inf['asvs_kept_train_prevalence']} ASVs kept")
    for name, fn in (("sfer", lambda: sfer(cfg, args.root, pts)), ("fish_crcp", lambda: crcp(cfg, args.root, pts, args.species)),
                     ("fish_maxn", lambda: maxn(cfg, args.root, pts))):
        try:
            df, inf = fn()
        except FileNotFoundError as exc:
            print(f"{name}: skipped ({exc})")
            continue
        df.to_csv(out / f"{name}.csv.gz", index=False)
        info["datasets"][name] = inf
        print(f"{name}: {inf['samples']} samples ({inf['train']} train / {inf['test']} test)")
    (out / "fit_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
