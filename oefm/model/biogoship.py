"""Bio-GO-SHIP metagenome benchmark: frozen embeddings vs Granite, same samples, targets and protocol.

Instructions: multimodal_seusmbol/instructions_agent_biogoship.md. The bundle (benchmark_export:
samples, tiles, targets, Granite reference, benchmark_eval.py) is unzipped to
<data_root>/processed/regional/eval/benchmark_export (config/biogoship.yaml, data_biogoship/).

Steps:
  embed     every sample with time and position (1,977), windows causal / retro, PACE visible /
            dropped, every run (evaluate.embed; runs from `runs_root`, normalisation from
            `stats_dir`, i.e. the Florida-trained encoders and their statistics).
  paired    the 1,215 benchmark tiles. One feature vector per tile = the mean of its samples'
            embeddings (the tile's `sample_keys`; samples without a valid observation are
            skipped). Scored with the bundle's own benchmark_eval.run, unchanged: technical removal,
            standardise -> PCA 64 -> first --pcs -> standardise, ridge with leave-one-cruise-out
            penalty, one cruise held out at a time. All feature sets of a window are scored on the
            same tiles (those where every set has features), and Granite is rerun on exactly those
            tiles (`granite_on_<window>`). Also --pcs 0 for E1 (all standardised dimensions).
  pace      paired tiles with PACE in the window (I09N-2025, parts of A13.5-2024 and I08S-2024): E1
            with PACE visible vs dropped, and Granite on the same tiles. Leave-one-cruise-out over the
            2-3 cruises: indicative only.
  extended  all samples, grouped into tiles by the new model's input: same cruise, same 1.2 km cell
            (zone grid slot), same UTC day and same sequencing platform. Targets = mean of the
            samples' targets; log10_ko_reads = tile mean of log10(ko_reads). Platform is a
            three-level factor (NovaSeq 6000, NovaSeq X, HiSeq 4000: the pre-2016 cruises add HiSeq).
            Same protocol (copy of benchmark_eval.run with a categorical platform term), 16 cruises.
  report    per target: explained share of the non-technical and of the reproducible non-technical
            variance, the difference to Granite on the same tiles, per-column R^2 medians, the
            number of KOs ahead of Granite by >= 0.1 R^2 and in >= 9 of 13 cruises, and per-cruise
            level / pattern / SD for C and all KOs (I09N-2025, I07N, P02-2022, GOMECC4).

    python -m oefm.model.biogoship --runs stage1_A stage1_B stage1_C --tag stage1 [--steps embed paired pace extended report]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TARGETS = ["clr_all", "C", "D", "A", "Dp", "B", "De"]
FOCUS_CRUISES = ("I09N-2025", "I07N", "P02-2022", "GOMECC4")
# instructions section 4: Granite main model, % of non-technical variance (reproduction check)
GRANITE_EXPECTED = {"clr_all": 17.60, "C": 53.80, "D": 13.96, "A": 13.71, "Dp": 16.36, "B": 10.33, "De": 18.89}


def wanted(name: str, window: str, sets_other: tuple[str, ...]) -> bool:
    """Every set in the causal window; in other windows only raw and the listed embedding levels."""
    return window == "causal" or name == "raw" or name.rsplit("/", 1)[-1] in sets_other
PLATFORM_LEVELS = ("NovaSeq 6000", "NovaSeq X", "HiSeq 4000")


def bundle_dir(cfg) -> Path:
    from ..targets import eval_dir

    return eval_dir(cfg) / "benchmark_export"


def load_bench(bundle: Path):
    """The bundle's benchmark_eval.py as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("benchmark_eval", bundle / "benchmark_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


def sample_points(bundle: Path):
    import pandas as pd

    s = pd.read_csv(bundle / "samples.csv")
    t = pd.to_datetime(s["collection_time_utc"], utc=True, errors="coerce")
    s = s[t.notna() & s["lat"].notna() & s["lon"].notna()].copy()
    s["time_utc"] = pd.to_datetime(s["collection_time_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    s["key"] = s["lat"].round(5).astype(str) + "|" + s["lon"].round(5).astype(str) + "|" + s["time_utc"]
    return s.reset_index(drop=True)


# ----------------------------------------------------------------------------- features
def sample_features(out_dir: Path, window: str, pace: str) -> dict:
    """{feature set: frame indexed by point key}, plus 'n_pace' / 'n_obs' series."""
    import pandas as pd

    f = out_dir / f"embeddings_{window}_{pace}.npz"
    if not f.exists():
        return {}
    e = dict(np.load(f, allow_pickle=False))
    keys = e["key"].astype(str)
    out = {"raw": pd.DataFrame(e["raw"], index=keys)}
    for k, v in e.items():
        if "__" in k:
            out[k.replace("__", "/")] = pd.DataFrame(v, index=keys)
    out["_n_pace"] = pd.Series(e["n_pace"], index=keys)
    out["_n_obs"] = pd.Series(e["n_obs"], index=keys)
    return out


def tile_features(samples, tile_members: dict[str, list[str]], feats) -> "pd.DataFrame":  # noqa: F821
    """Mean of the members' sample features per tile (NaN if no member has all of them)."""
    import pandas as pd

    key_of = dict(zip(samples["sample_key"], samples["key"]))
    rows, idx = [], []
    for tk, members in tile_members.items():
        ks = [key_of[m] for m in members if m in key_of and key_of[m] in feats.index]
        sub = feats.loc[ks].to_numpy(float) if ks else np.zeros((0, feats.shape[1]))
        sub = sub[np.isfinite(sub).all(1)]
        rows.append(sub.mean(0) if len(sub) else np.full(feats.shape[1], np.nan))
        idx.append(tk)
    return pd.DataFrame(np.array(rows), index=idx)


def _tile_any(samples, tile_members, series) -> "pd.Series":  # noqa: F821
    import pandas as pd

    key_of = dict(zip(samples["sample_key"], samples["key"]))
    return pd.Series({tk: int(sum(series.get(key_of.get(m, ""), 0) for m in ms)) for tk, ms in tile_members.items()})


# ----------------------------------------------------------------------------- paired benchmark
def paired(cfg, out_dir: Path, windows, pcs: int = 32, sets_other=("e1",), log=print) -> None:
    import pandas as pd

    bundle = bundle_dir(cfg)
    be = load_bench(bundle)
    samples = sample_points(bundle)
    tiles = pd.read_csv(bundle / "tiles.csv", index_col="tile_key")
    members = {tk: str(v).split(";") for tk, v in tiles["sample_keys"].items()}
    res_all = []
    # step 1 of the instructions: the bundle must reproduce Granite on all 1,215 tiles
    chk = be.run(bundle, be.load_features(bundle / "granite" / "tile_embeddings.npy", tiles), "granite_check",
                 TARGETS, 32, set(tiles.index))
    got = chk.set_index("target")["explained_of_nontechnical"].mul(100)
    diff = {t: round(float(got[t]) - v, 2) for t, v in GRANITE_EXPECTED.items()}
    log(f"granite reproduction (% of non-technical; expected {GRANITE_EXPECTED}): "
        f"{got.round(2).to_dict()}; differences {diff} (B and De may differ by <= 0.03)")
    chk["window"], chk["pcs"], chk["set"], chk["feature_set"] = "-", 32, "granite_check", "granite"
    res_all.append(chk)
    for window in windows:
        feats = sample_features(out_dir, window, "visible")
        if not feats:
            log(f"biogoship paired {window}: no embeddings_{window}_visible.npz (run --steps embed)")
            continue
        tf = {name: tile_features(samples, members, f).reindex(tiles.index)
              for name, f in feats.items() if not name.startswith("_") and wanted(name, window, sets_other)}
        keep = set(tiles.index)
        for name, f in tf.items():
            keep &= set(f.index[f.notna().all(1)])
        log(f"biogoship paired {window}: {len(keep):,} of {len(tiles):,} benchmark tiles have every feature set "
            f"({len(tf)} sets); dropped by cruise: "
            f"{tiles.loc[sorted(set(tiles.index) - keep), 'cruise'].value_counts().to_dict()}")
        runs = [(f"granite_on_{window}", be.load_features(bundle / "granite" / "tile_embeddings.npy", tiles), 32)]
        runs += [(f"{name.replace('/', '_')}_{window}", f, pcs) for name, f in tf.items()]
        runs += [(f"{name.replace('/', '_')}_{window}_pcs0", f, 0) for name, f in tf.items() if name.endswith("/e1")]
        for rname, f, p in runs:
            log(f"benchmark {rname} (pcs {p})")
            part = be.run(bundle, f, rname, TARGETS, p, keep)
            part["window"], part["pcs"], part["set"] = window, p, "paired"
            part["feature_set"] = rname.removesuffix(f"_{window}").removesuffix(f"_{window}_pcs0")
            res_all.append(part)
    if res_all:
        pd.concat(res_all).to_csv(out_dir / "biogoship_paired.csv", index=False)
        log(f"-> {out_dir / 'biogoship_paired.csv'}")


def pace_subset(cfg, out_dir: Path, windows, pcs: int = 32, log=print) -> None:
    import pandas as pd

    bundle = bundle_dir(cfg)
    be = load_bench(bundle)
    samples = sample_points(bundle)
    tiles = pd.read_csv(bundle / "tiles.csv", index_col="tile_key")
    members = {tk: str(v).split(";") for tk, v in tiles["sample_keys"].items()}
    res = []
    for window in windows:
        vis, drop = sample_features(out_dir, window, "visible"), sample_features(out_dir, window, "drop")
        if not vis or not drop:
            continue
        npace = _tile_any(samples, members, vis["_n_pace"]).reindex(tiles.index).fillna(0)
        sets = {f"{n}|{p}": tile_features(samples, members, d[n]).reindex(tiles.index)
                for n in vis if n.endswith("/e1") for p, d in (("visible", vis), ("drop", drop))}
        keep = set(npace.index[npace > 0])
        for f in sets.values():
            keep &= set(f.index[f.notna().all(1)])
        by = tiles.loc[sorted(keep), "cruise"].value_counts().to_dict()
        if len(by) < 2 or len(keep) < 20:
            log(f"biogoship pace {window}: {len(keep)} tiles with PACE in {by}; too few, skipped")
            continue
        log(f"biogoship pace {window}: {len(keep)} benchmark tiles with PACE in the window, by cruise {by}")
        runs = [(f"pace_granite_{window}", be.load_features(bundle / "granite" / "tile_embeddings.npy", tiles), 32)]
        runs += [(f"pace_{k.replace('/', '_').replace('|', '_')}_{window}", f, pcs) for k, f in sets.items()]
        for rname, f, p in runs:
            part = be.run(bundle, f, rname, TARGETS, p, keep)
            part["window"], part["set"] = window, "pace_era"
            part["feature_set"] = rname
            res.append(part)
    if res:
        pd.concat(res).to_csv(out_dir / "biogoship_pace.csv", index=False)
        log(f"-> {out_dir / 'biogoship_pace.csv'}")


# ----------------------------------------------------------------------------- extended set
def extended_tiles(cfg, bundle: Path):
    """(tiles frame, {tile: sample keys}) grouping all samples by the new model's input."""
    import pandas as pd

    from ..grid import grids

    s = sample_points(bundle)
    gl = grids(cfg)
    slot = []
    for la, lo in zip(s["lat"].to_numpy(float), s["lon"].to_numpy(float)):
        lab = "none"
        for g in gl:
            x, y = g.proj.forward(np.array([lo]), np.array([la]))
            k = int(g.slots(x, y)[0])
            if k >= 0:
                lab = f"{g.zone}:{k}"
                break
        slot.append(lab)
    s["slot"] = slot
    s["day"] = s["time_utc"].str[:10]
    s["platform_level"] = s["platform"].where(s["platform"].isin(PLATFORM_LEVELS), "NovaSeq X")
    s["log10_ko_reads"] = np.log10(s["ko_reads"].astype(float))
    g = s.groupby(["cruise", "slot", "day", "platform_level"], sort=True)
    rows, members = [], {}
    for i, ((cruise, sl, day, plat), d) in enumerate(g):
        tk = f"x{i:05d}"
        members[tk] = list(d["sample_key"])
        rows.append({"tile_key": tk, "cruise": cruise, "platform_level": plat, "k_samples": len(d),
                     "sample_keys": ";".join(d["sample_key"]), "log10_ko_reads": float(d["log10_ko_reads"].mean()),
                     "slot": sl, "day": day})
    return pd.DataFrame(rows).set_index("tile_key"), members


def technical3(y, plat_codes, logtot):
    """benchmark_eval.technical with a categorical platform (one indicator per non-reference level)."""
    from sklearn.preprocessing import SplineTransformer

    levels = sorted(set(plat_codes))
    ind = [(plat_codes == lv).astype(float)[:, None] for lv in levels[1:]]
    X = np.hstack([np.ones((len(y), 1)), *ind, SplineTransformer(n_knots=8, degree=3).fit_transform(logtot[:, None])])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    return y - (fit - y.mean(0)), fit - fit.mean(0)


def run_extended(be, tiles, targets_by: dict, feats, name: str, pcs: int, keep: set, out_dir: Path):
    """benchmark_eval.run's protocol on the extended tiles (categorical platform; within-tile noise from
    the extended tiles' own same-tile pairs)."""
    import pandas as pd

    parts = []
    od = out_dir / f"ext_{name}"
    od.mkdir(exist_ok=True)
    for t, Yall in targets_by.items():
        Y = Yall.loc[[k for k in Yall.index if k in keep]]
        tt = tiles.loc[Y.index]
        x = feats.loc[Y.index].to_numpy(float)
        y = Y.to_numpy(float)
        cr = tt["cruise"].to_numpy()
        ya, tech = technical3(y, tt["platform_level"].to_numpy(), tt["log10_ko_reads"].to_numpy())
        pred = np.full_like(ya, np.nan)
        for c in np.unique(cr):
            te, tr = np.where(cr == c)[0], np.where(cr != c)[0]
            a, b = be.features_fold(x, tr, te, pcs)
            pred[te] = be.ridge_loco_alpha(a, ya[tr], b, cr[tr])
        np.savez_compressed(od / f"predictions_{t}.npz", predicted=pred.astype(np.float32), observed=ya.astype(np.float32),
                            tile_key=Y.index.to_numpy(), columns=np.array(Y.columns), cruise=cr)
        yac, res = ya - ya.mean(0), ya - pred
        r2 = 1 - (res ** 2).sum(0) / (yac ** 2).sum(0)
        tot = ((y - y.mean(0)) ** 2).sum()
        p = {"target": t, "features": name, "tiles": len(y), "columns": y.shape[1], "technical": (tech ** 2).sum() / tot,
             "explained": ((yac ** 2).sum() - (res ** 2).sum()) / tot, "median_column_r2": float(np.median(r2))}
        p["explained_of_nontechnical"] = p["explained"] / (1 - p["technical"])
        parts.append(p)
    return pd.DataFrame(parts)


def extended(cfg, out_dir: Path, windows, pcs: int = 32, sets_other=("e1",), log=print) -> None:
    import pandas as pd

    bundle = bundle_dir(cfg)
    be = load_bench(bundle)
    tiles, members = extended_tiles(cfg, bundle)
    tiles.to_csv(out_dir / "biogoship_extended_tiles.csv")
    samples = sample_points(bundle)
    targets_by = {}
    for t in TARGETS:
        st = pd.read_csv(bundle / "targets_sample" / f"{t}.csv.gz", index_col="sample_key")
        rows = {tk: st.loc[[m for m in ms if m in st.index]].mean(0) for tk, ms in members.items()
                if any(m in st.index for m in ms)}
        targets_by[t] = pd.DataFrame(rows).T
    log(f"biogoship extended: {len(samples):,} samples -> {len(tiles):,} tiles in {tiles['cruise'].nunique()} cruises; "
        f"platform levels {tiles['platform_level'].value_counts().to_dict()}")
    res = []
    for window in windows:
        feats = sample_features(out_dir, window, "visible")
        if not feats:
            continue
        tf = {n: tile_features(samples, members, f).reindex(tiles.index) for n, f in feats.items()
              if not n.startswith("_") and wanted(n, window, sets_other)}
        keep = set(tiles.index)
        for f in tf.values():
            keep &= set(f.index[f.notna().all(1)])
        log(f"biogoship extended {window}: {len(keep):,} tiles with every feature set, by cruise "
            f"{tiles.loc[sorted(keep), 'cruise'].value_counts().to_dict()}")
        for n, f in tf.items():
            part = run_extended(be, tiles, targets_by, f, f"{n.replace('/', '_')}_{window}", pcs, keep, out_dir)
            part["window"], part["set"], part["feature_set"] = window, "extended", n
            res.append(part)
            log(f"  {n:14s} " + "  ".join(f"{r.target} {100 * r.explained_of_nontechnical:.2f}" for r in part.itertuples()))
    if res:
        pd.concat(res).to_csv(out_dir / "biogoship_extended.csv", index=False)
        log(f"-> {out_dir / 'biogoship_extended.csv'}")


# ----------------------------------------------------------------------------- report
def ko_comparison(bundle: Path, new: str, ref: str, target: str = "clr_all", margin: float = 0.1, min_cruises: int = 9):
    """Per KO: R^2 of `new` and `ref` (same tiles) and the cruises where new has the lower squared
    error. Returns (frame, n ahead by >= margin, n ahead by >= margin and in >= min_cruises cruises)."""
    import pandas as pd

    a = np.load(bundle / f"eval_{new}" / f"predictions_{target}.npz", allow_pickle=True)
    b = np.load(bundle / f"eval_{ref}" / f"predictions_{target}.npz", allow_pickle=True)
    assert (a["tile_key"] == b["tile_key"]).all()
    o = a["observed"]
    oc = o - o.mean(0)
    r2a = 1 - ((o - a["predicted"]) ** 2).sum(0) / (oc ** 2).sum(0)
    r2b = 1 - ((o - b["predicted"]) ** 2).sum(0) / (oc ** 2).sum(0)
    cr = a["cruise"]
    wins = np.zeros(o.shape[1], int)
    for c in np.unique(cr):
        m = cr == c
        wins += (((o[m] - a["predicted"][m]) ** 2).sum(0) < ((o[m] - b["predicted"][m]) ** 2).sum(0)).astype(int)
    df = pd.DataFrame({"column": a["columns"], "r2_new": r2a, "r2_ref": r2b, "cruises_better": wins,
                       "n_cruises": len(np.unique(cr))})
    ahead = (df["r2_new"] - df["r2_ref"]) >= margin
    return df, int(ahead.sum()), int((ahead & (df["cruises_better"] >= min_cruises)).sum())


def report(cfg, out_dir: Path, log=print) -> None:
    import pandas as pd

    bundle = bundle_dir(cfg)
    f = out_dir / "biogoship_paired.csv"
    if not f.exists():
        log("biogoship report: no biogoship_paired.csv")
        return
    p = pd.read_csv(f)
    for window, d in p.groupby("window", sort=False):
        if window == "-":
            continue
        ref = d[d["features"] == f"granite_on_{window}"].set_index("target")
        tab = d.pivot_table(index="features", columns="target", values="explained_of_nontechnical", sort=False)[TARGETS]
        rep = d.pivot_table(index="features", columns="target", values="explained_of_reproducible", sort=False)[TARGETS]
        log(f"Bio-GO-SHIP paired benchmark, window {window}, {int(ref['tiles'].iloc[0]):,} tiles: explained % of "
            f"non-technical variance (Granite rerun on the same tiles first)")
        log(tab.mul(100).round(2).to_string())
        log("  % of reproducible non-technical variance:")
        log(rep.mul(100).round(2).to_string())
        log("  difference to Granite on the same tiles (percentage points):")
        log(tab.sub(tab.loc[f"granite_on_{window}"], axis=1).mul(100).round(2).to_string())
        med = d.pivot_table(index="features", columns="target", values="median_column_r2", sort=False)[TARGETS]
        log("  median per-column R^2:")
        log(med.round(4).to_string())
        rows = []
        for name in d["features"].unique():
            if name.startswith("granite"):
                continue
            try:
                _, n1, n9 = ko_comparison(bundle, name, f"granite_on_{window}")
            except (FileNotFoundError, AssertionError):
                continue
            rows.append({"features": name, "KOs ahead >= 0.1": n1, "and in >= 9 cruises": n9})
        if rows:
            log("  all-KO comparison with Granite (3,933 KOs):")
            log(pd.DataFrame(rows).to_string(index=False))
        pcs = []
        for name in d["features"].unique():
            pc = bundle / f"eval_{name}" / "per_cruise.csv"
            if pc.exists():
                c = pd.read_csv(pc)
                pcs.append(c[c["target"].isin(["C", "clr_all"])])
        if pcs:
            c = pd.concat(pcs)
            c = c[c["cruise"].isin(FOCUS_CRUISES)]
            log("  per cruise (C and all KOs): level bias / pattern r / within-cruise SD")
            c["cell"] = (c["bias_mean"].round(2).astype(str) + " / " + c["r_median"].round(2).astype(str)
                         + " / " + c["sd_obs_median"].round(2).astype(str))
            log(c.pivot_table(index=["target", "cruise"], columns="features", values="cell", aggfunc="first").to_string())
    for name, label in (("biogoship_pace.csv", "PACE-era subset"), ("biogoship_extended.csv", "extended set")):
        g = out_dir / name
        if g.exists():
            d = pd.read_csv(g)
            log(f"Bio-GO-SHIP {label}: explained % of non-technical variance")
            log(d.pivot_table(index=["window", "feature_set"], columns="target", values="explained_of_nontechnical",
                              sort=False)[TARGETS].mul(100).round(2).to_string())


def main(argv=None) -> None:
    from ..config import load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["stage1_A", "stage1_B", "stage1_C"])
    ap.add_argument("--weights", default="final")
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--steps", nargs="+", default=["embed", "paired", "pace", "extended", "report"],
                    choices=["embed", "paired", "pace", "extended", "report"])
    ap.add_argument("--windows", nargs="+", default=["causal", "retro"], choices=["causal", "retro"])
    ap.add_argument("--pcs", type=int, default=32)
    ap.add_argument("--sets-other", nargs="+", default=["e1"],
                    help="embedding levels scored in windows other than causal (causal: all)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config()
    out_dir = cfg.data_root / "runs" / f"eval_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = bundle_dir(cfg)
    if not (bundle / "benchmark_eval.py").exists():
        raise SystemExit(f"bundle not found: unzip benchmark_export.zip to {bundle}")
    if "embed" in args.steps:
        import torch   # GPU environment only; the scoring steps need numpy, pandas and scikit-learn

        from .evaluate import embed, load_run

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models, tcfg = [], None
        for r in args.runs:
            m, info = load_run(cfg, r, args.weights, device)
            models.append((m, info))
            tcfg = tcfg or m.train_config
            print(f"loaded {info['run']}: arm {info['arm']}, step {info['step']:,}, {info['weights']}", flush=True)
        (out_dir / "models.json").write_text(json.dumps([i for _, i in models], indent=1))
        pts = sample_points(bundle)[["key", "lat", "lon", "time_utc"]].drop_duplicates("key").reset_index(drop=True)
        embed(cfg, models, out_dir, tcfg, args.windows, ["visible", "drop"], args.batch, args.workers, args.limit,
              points=pts)
    if "paired" in args.steps:
        paired(cfg, out_dir, args.windows, args.pcs, tuple(args.sets_other))
    if "pace" in args.steps:
        pace_subset(cfg, out_dir, args.windows, args.pcs)
    if "extended" in args.steps:
        extended(cfg, out_dir, args.windows, args.pcs, tuple(args.sets_other))
    if "report" in args.steps:
        report(cfg, out_dir)


if __name__ == "__main__":
    main()
