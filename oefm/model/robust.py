"""Stage 1 robustness checks (step 2.10b), on the embeddings of `evaluate --steps embed`.

    python -m oefm.model.evaluate --runs ... --steps robust amplitude

Settings: PACE visible, eDNA with the sequencing covariates (as the headline table of `probes`).

rotate     Rotate the test year: for every label year with >= 30 samples, probes are fitted on all
           other years and scored on that year (alpha by grouped CV inside the training years, as in
           `probes`). The official split is the rotation that holds out 2024 (eDNA, CRCP, MaxN) or
           2025 (SFER). Imagery has no label hold-out, so any year can be the test year. Caveats:
           training years after the test year are used, and the eDNA NMF/PCA bases were fitted on
           2018-2023 labels (shared by every feature set, so comparisons stay fair).
           -> robust_rotate.csv (per test year and target family: median R^2 over targets with
           test spread >= 0.25 x training spread, mean AUC for presence)
           -> robust_rotate_summary.csv (mean and SD over years; per feature set, the mean
           difference to the reference feature set and the number of years it is better)
bootstrap  The official split with paired bootstrap intervals: test samples resampled within
           their cross-validation group (cruise / site / month), the same resample for every
           feature set, n_boot times. 95% intervals of each family's median R^2 (mean AUC), of the
           difference to the reference feature set, and the share of resamples where the feature
           set beats the reference. The 2024 eDNA test year has only 4 cruises, so resampling whole
           cruises would be degenerate: these intervals cover sampling noise within the cruises,
           and `rotate` covers year-to-year (cruise-to-cruise) variation.
           -> robust_bootstrap.csv
pace_era   PACE visible vs hidden where the probe can learn from PACE: in `probes` no eDNA
           training sample has PACE (all are before 2024), so the probe cannot use PACE-specific
           directions. Here only samples with >= 1 PACE observation in the 3 x 3 cells (this
           window) are used, in both training and test, once with PACE-visible and once with
           PACE-dropped embeddings. If both label splits have enough such samples (SFER: 2024
           training, 2025 test), the official split is used; otherwise (eDNA: 2024 only) grouped
           cross-validation among them (<= 5 folds by group: leave-one-cruise-out for eDNA). Paired bootstrap of the
           difference visible - dropped (within groups).
           -> robust_pace_era.csv
retention  Does the embedding keep what its input had? Ridge from each feature set to quantities
           of the raw reflectance of the same observations (raw MODIS-band Rrs averaged over the
           window): each band (asinh), brightness (log10 mean Rrs 412-555 nm) and the OC3M
           blue/green ratio log10(max(Rrs443, Rrs488) / Rrs547), the inputs of empirical
           chlorophyll. 5-fold cross-validation grouped by month over all evaluation samples
           (no labels involved). -> robust_retention.csv

amplitude  (needs the models) The tokenizer test of the design document on trained weights, and
           the same at the E1 level: synthetic uniform scenes of one MODIS-band spectrum scaled by
           0.5, 1 and 2 must give distinguishable outputs, and a linear probe must recover log
           amplitude (and the log blue/green ratio) with R^2 >= 0.95. -> amplitude_check.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from .evaluate import (_metrics, fit_predict, probe_inputs, ridge_probe, sd_ratio, target_family,
                       usable_targets)

MIN_PERIOD = 30          # labelled samples for a test year
SD_MIN = 0.25            # test spread / training spread below which R^2 is left out of the median


def _period(times, unit: str) -> np.ndarray:
    import pandas as pd

    t = pd.to_datetime(pd.Series(times), utc=True)
    if unit == "quarter":   # synthetic test data (one year)
        return (t.dt.year.astype(str) + "Q" + t.dt.quarter.astype(str)).to_numpy()
    return t.dt.year.astype(str).to_numpy()


def _setting(inp: dict) -> tuple[str, np.ndarray | None]:
    """Headline setting: the sequencing covariates where they exist."""
    return ("seq", inp["C"]) if inp["C"] is not None else ("none", None)


def _features(inp: dict, pace: str = "visible") -> dict[str, np.ndarray]:
    return {name: X for (name, p), X in inp["fsets"].items() if p == pace}


def _family_scores(Y, P, keep, tcols) -> dict[str, dict]:
    """Per family: median R^2 over continuous targets in `keep`, mean AUC over presence targets."""
    fam: dict[str, dict] = {}
    for j, c in enumerate(tcols):
        binary = c.startswith("present_")
        if not keep[j] and not binary:
            continue
        m = _metrics(Y[:, j], P[:, j], binary)
        d = fam.setdefault(target_family(c), {"r2": [], "auc": []})
        if binary:
            d["auc"].append(m.get("test_auc", np.nan))
        else:
            d["r2"].append(m.get("test_r2", np.nan))
    out = {}
    for f, d in fam.items():
        out[f] = {"r2_median": float(np.nanmedian(d["r2"])) if np.isfinite(d["r2"]).any() else np.nan,
                  "auc_mean": float(np.nanmean(d["auc"])) if np.isfinite(d["auc"]).any() else np.nan,
                  "n_targets": len(d["r2"]) + len(d["auc"])}
    return out


# ----------------------------------------------------------------------------- bootstrap helpers
def _strata_resamples(groups: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n] indices: each group's samples resampled with replacement within the group."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    out = np.empty((n_boot, n), int)
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        out[:, ix] = ix[rng.integers(0, len(ix), (n_boot, len(ix)))]
    return out


def _boot_r2(y: np.ndarray, p: np.ndarray, I: np.ndarray) -> np.ndarray:
    """R^2 of one target on each resample (NaN labels skipped)."""
    yb, pb = y[I], p[I]
    ok = np.isfinite(yb)
    cnt = ok.sum(1)
    ybz = np.where(ok, yb, 0.0)
    mean = ybz.sum(1) / np.maximum(cnt, 1)
    ss = (np.where(ok, yb - mean[:, None], 0.0) ** 2).sum(1)
    sr = (np.where(ok, yb - pb, 0.0) ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ss > 0, 1 - sr / ss, np.nan)


def _boot_auc(y: np.ndarray, p: np.ndarray, I: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    ok = np.isfinite(y)
    yb, pb = y[I], p[I]
    res = np.full(len(I), np.nan)
    if not ok.all():
        return res
    r = rankdata(pb, axis=1)
    n1 = (yb == 1).sum(1)
    n0 = yb.shape[1] - n1
    with np.errstate(divide="ignore", invalid="ignore"):
        res = ((r * (yb == 1)).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.where((n1 > 0) & (n0 > 0), res, np.nan)


def _boot_family(Y, P, keep, tcols, I) -> dict[str, np.ndarray]:
    """Per family, [n_boot] median R^2 (or mean AUC for presence) on the resamples."""
    acc: dict[str, list] = {}
    for j, c in enumerate(tcols):
        binary = c.startswith("present_")
        if not keep[j] and not binary:
            continue
        f = target_family(c)
        acc.setdefault((f, binary), []).append(_boot_auc(Y[:, j], P[:, j], I) if binary else _boot_r2(Y[:, j], P[:, j], I))
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for (f, binary), v in acc.items():
            v = np.stack(v, 1)
            out[f] = np.nanmean(v, 1) if binary else np.nanmedian(v, 1)
    return out


def _interval_rows(base: dict, point: dict, boots: dict, ref: str, metric_of) -> list[dict]:
    """point/boots: features -> family -> value / [n_boot]."""
    rows = []
    for name in point:
        for f, val in point[name].items():
            b = boots[name].get(f)
            rec = {**base, "features": name, "family": f, "metric": metric_of(f), "value": val,
                   "ci_lo": np.nanpercentile(b, 2.5) if b is not None else np.nan,
                   "ci_hi": np.nanpercentile(b, 97.5) if b is not None else np.nan}
            if ref in point and f in point[ref] and b is not None:
                d = b - boots[ref][f]
                rec.update(ref=ref, delta=val - point[ref][f], delta_lo=np.nanpercentile(d, 2.5),
                           delta_hi=np.nanpercentile(d, 97.5), p_better=float(np.nanmean(d > 0)))
            rows.append(rec)
    return rows


# ----------------------------------------------------------------------------- steps
def robustness(cfg, out_dir: Path, windows, steps=("rotate", "bootstrap", "pace_era", "retention"),
               ref: str = "stage1_A/e1", n_boot: int = 1000, period: str = "year", min_period: int = MIN_PERIOD,
               log=print) -> None:
    import pandas as pd

    rot, boot, era = [], [], []
    for window in windows:
        for inp in probe_inputs(cfg, out_dir, window, ["visible", "drop"], "on", log):
            ds, df, Y0, groups, common = inp["ds"], inp["df"], inp["Y"], inp["groups"], inp["common"]
            cov, C = _setting(inp)
            feats = _features(inp)
            per = _period(df["time_utc"], period)
            off_test = (df["label_split"] == "test").to_numpy()
            off_periods = set(per[off_test & common])
            # --- rotate: every test year; bootstrap: the official split (label_split)
            folds = []
            if "rotate" in steps:
                folds += [(y, common & (per != y), common & (per == y)) for y in sorted(set(per[common]))]
            if "bootstrap" in steps:
                folds.append(("official", common & ~off_test, common & off_test))
            for y, tr, te in folds:
                if te.sum() < min_period or tr.sum() < 20:
                    continue
                ok_t = usable_targets(Y0, tr)
                tcols = [c for c, o in zip(inp["tcols"], ok_t) if o]
                Y = Y0[:, ok_t]
                keep = sd_ratio(Y, tr, te) >= SD_MIN
                preds = {name: fit_predict(name, X, Y, tr, groups, C)[0] for name, X in feats.items()}
                scores = {n: _family_scores(Y[te], P[te], keep, tcols) for n, P in preds.items()}
                if y != "official":
                    for name, fam in scores.items():
                        for f, sc in fam.items():
                            rot.append({"dataset": ds, "window": window, "covariates": cov, "features": name,
                                        "test_period": y, "official_test_period": y in off_periods,
                                        "n_train": int(tr.sum()), "n_test": int(te.sum()),
                                        "n_groups_test": len(set(groups[te])), "family": f, **sc})
                    continue
                I = _strata_resamples(groups[te], n_boot, seed=20261123)
                point = {n: {f: (sc["auc_mean"] if np.isnan(sc["r2_median"]) else sc["r2_median"]) for f, sc in fam.items()}
                         for n, fam in scores.items()}
                boots = {n: _boot_family(Y[te], P[te], keep, tcols, I) for n, P in preds.items()}
                binfam = {target_family(c) for c in tcols if c.startswith("present_")}
                boot += _interval_rows({"dataset": ds, "window": window, "covariates": cov,
                                        "test_periods": "+".join(sorted(off_periods)), "n_test": int(te.sum()),
                                        "n_groups_test": len(set(groups[te])), "n_boot": n_boot},
                                       point, boots, ref, lambda f: "auc_mean" if f in binfam else "r2_median")
            log(f"robust {ds}/{window}: test periods {sorted(set(per[common]))} (official {sorted(off_periods)})")
            # --- PACE era: visible vs dropped where the probe trains on PACE-era samples too
            if "pace_era" in steps and "drop" in inp["n_pace"]:
                era += _pace_era(inp, window, cov, C, n_boot, log)
    if rot:
        r = pd.DataFrame(rot)
        r.to_csv(out_dir / "robust_rotate.csv", index=False)
        _rotate_summary(r, ref, out_dir, log)
    if boot:
        b = pd.DataFrame(boot)
        b.to_csv(out_dir / "robust_bootstrap.csv", index=False)
        _print_boot(b, ref, log)
    if era:
        e = pd.DataFrame(era)
        e.to_csv(out_dir / "robust_pace_era.csv", index=False)
        _print_era(e, log)
    if "retention" in steps:
        retention(out_dir, windows, log=log)


def _pace_era(inp, window, cov, C, n_boot, log) -> list[dict]:
    ds, df, Y0, groups, common = inp["ds"], inp["df"], inp["Y"], inp["groups"], inp["common"]
    P_ok = common & (inp["n_pace"]["visible"] > 0)
    off_tr = (df["label_split"] == "train").to_numpy()
    if (P_ok & off_tr).sum() >= 50 and (P_ok & ~off_tr).sum() >= 30:
        mode, folds = "split", [(P_ok & off_tr, P_ok & ~off_tr)]
    else:
        gs = sorted(set(groups[P_ok]))
        if len(gs) < 3 or P_ok.sum() < 30:
            log(f"robust pace_era {ds}/{window}: {P_ok.sum()} samples with PACE in {len(gs)} groups; skipped")
            return []
        from sklearn.model_selection import GroupKFold

        # grouped K-fold (<= 5 folds): leave-one-cruise-out for eDNA (3 cruises); CRCP groups are sites
        idx = np.where(P_ok)[0]
        mode, folds = "grouped_cv", []
        for _, va in GroupKFold(n_splits=min(5, len(gs))).split(idx, groups=groups[idx]):
            te = np.zeros(len(df), bool)
            te[idx[va]] = True
            folds.append((P_ok & ~te, te))
    te_all = np.zeros(len(df), bool)
    for _, te in folds:
        te_all |= te
    ok_t = usable_targets(Y0, P_ok & ~te_all if mode == "split" else P_ok)
    tcols = [c for c, o in zip(inp["tcols"], ok_t) if o]
    Y = Y0[:, ok_t]
    names = sorted({n for (n, p) in inp["fsets"] if n not in ("meta", "seq")})
    preds = {}
    for pace in ("visible", "drop"):
        for name in names:
            X = inp["fsets"][(name, pace)]
            P = np.full(Y.shape, np.nan)
            for tr, te in folds:
                if tr.sum() < 20:
                    continue
                P[te] = fit_predict(name, X, Y, tr, groups, C)[0][te]
            preds[(name, pace)] = P
    tr_ref = folds[0][0] if mode == "split" else P_ok
    keep = sd_ratio(Y, tr_ref, te_all) >= SD_MIN if mode == "split" else np.ones(Y.shape[1], bool)
    I = _strata_resamples(groups[te_all], n_boot, seed=20261124)
    rows = []
    for name in names:
        sv = _family_scores(Y[te_all], preds[(name, "visible")][te_all], keep, tcols)
        sd = _family_scores(Y[te_all], preds[(name, "drop")][te_all], keep, tcols)
        bv = _boot_family(Y[te_all], preds[(name, "visible")][te_all], keep, tcols, I)
        bd = _boot_family(Y[te_all], preds[(name, "drop")][te_all], keep, tcols, I)
        for f in sv:
            metric = "r2_median" if np.isfinite(sv[f]["r2_median"]) else "auc_mean"
            d = bv[f] - bd[f] if f in bv and f in bd else np.array([np.nan])
            rows.append({"dataset": ds, "window": window, "covariates": cov, "mode": mode, "features": name,
                         "family": f, "metric": metric, "n_test": int(te_all.sum()),
                         "n_groups": len(set(groups[te_all])), "visible": sv[f][metric], "drop": sd[f][metric],
                         "delta": sv[f][metric] - sd[f][metric], "delta_lo": np.nanpercentile(d, 2.5),
                         "delta_hi": np.nanpercentile(d, 97.5), "p_visible_better": float(np.nanmean(d > 0))})
    log(f"robust pace_era {ds}/{window}: {mode}, {te_all.sum()} scored samples with PACE in "
        f"{len(set(groups[te_all]))} groups, {len(tcols)} targets")
    return rows


def _rotate_summary(r, ref, out_dir, log) -> None:
    import pandas as pd

    r = r.copy()
    r["value"] = r["r2_median"].where(r["r2_median"].notna(), r["auc_mean"])
    keys = ["dataset", "family", "window", "covariates"]
    g = r.groupby(keys + ["features"])["value"]
    s = pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n_periods": g.size(), "min": g.min(), "max": g.max()}).reset_index()
    refv = r[r["features"] == ref][keys + ["test_period", "value"]].rename(columns={"value": "ref_value"})
    if len(refv):
        m = r.merge(refv, on=keys + ["test_period"], how="left")
        m["d"] = m["value"] - m["ref_value"]
        gd = m.groupby(keys + ["features"])
        s = s.merge(pd.DataFrame({"delta_vs_ref_mean": gd["d"].mean(), "periods_better_than_ref": gd["d"].apply(lambda x: int((x > 0).sum())),
                                  "periods_compared": gd["d"].apply(lambda x: int(x.notna().sum()))}).reset_index(),
                    on=keys + ["features"], how="left")
        s["ref"] = ref
    s.to_csv(out_dir / "robust_rotate_summary.csv", index=False)
    log(f"Test-year rotation: mean over test years of the family median R^2 (mean AUC for presence); "
        f"[years better than {ref} / years]. Per year: robust_rotate.csv")
    s["cell"] = s["mean"].round(3).astype(str) + (
        " [" + s["periods_better_than_ref"].astype("Int64").astype(str) + "/" + s["periods_compared"].astype("Int64").astype(str) + "]"
        if "periods_better_than_ref" in s else "")
    log(s.pivot_table(index=["dataset", "family", "window"], columns="features", values="cell", aggfunc="first").to_string())


def _print_boot(b, ref, log) -> None:
    b = b.copy()
    b["cell"] = b["value"].round(3).astype(str) + " (" + b["ci_lo"].round(2).astype(str) + "," + b["ci_hi"].round(2).astype(str) + ")"
    log("Official split, paired bootstrap within groups: value (95% interval). robust_bootstrap.csv also has "
        f"the difference to {ref} and its interval")
    log(b.pivot_table(index=["dataset", "family", "window"], columns="features", values="cell", aggfunc="first").to_string())


def _print_era(e, log) -> None:
    e = e.copy()
    e["cell"] = e["delta"].round(3).astype(str) + " (" + e["delta_lo"].round(2).astype(str) + "," + e["delta_hi"].round(2).astype(str) + ")"
    log("PACE era, visible - dropped (95% interval), probes trained and scored on samples with PACE:")
    log(e.pivot_table(index=["dataset", "family", "window", "mode"], columns="features", values="cell", aggfunc="first").to_string())


# ----------------------------------------------------------------------------- retention
def optical_quantities(raw: np.ndarray) -> dict[str, np.ndarray]:
    """raw: [n, 10] Rrs (sr^-1) at MODIS_REF_WL (412 443 469 488 531 547 555 645 667 678)."""
    from .data import MODIS_REF_WL

    wl = list(MODIS_REF_WL)
    q = {f"band_{int(w)}": np.arcsinh(raw[:, i] / 5e-4) for i, w in enumerate(wl)}
    with np.errstate(divide="ignore", invalid="ignore"):
        vis = raw[:, [wl.index(w) for w in (412.0, 443.0, 469.0, 488.0, 531.0, 547.0, 555.0)]]
        q["log_brightness"] = np.log10(np.nanmean(np.where(vis > 0, vis, np.nan), 1))
        bg = np.fmax(raw[:, wl.index(443.0)], raw[:, wl.index(488.0)]) / raw[:, wl.index(547.0)]
        q["log_blue_green_oc3m"] = np.log10(np.where(bg > 0, bg, np.nan))
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in q.items()}


def retention(out_dir: Path, windows, n_folds: int = 5, log=print) -> None:
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    rows = []
    for window in windows:
        f = out_dir / f"embeddings_{window}_visible.npz"
        if not f.exists():
            continue
        e = dict(np.load(f, allow_pickle=False))
        month = np.array([k.split("|")[2][:7] for k in e["key"]])
        t = pd.to_datetime(pd.Series([k.split("|")[2] for k in e["key"]]), utc=True)
        doy = 2 * np.pi * t.dt.day_of_year.to_numpy() / 365.25
        lat = np.array([float(k.split("|")[0]) for k in e["key"]])
        lon = np.array([float(k.split("|")[1]) for k in e["key"]])
        feats = {"meta": np.c_[lat, lon, np.sin(doy), np.cos(doy)]}
        feats.update({k.replace("__", "/"): v for k, v in e.items() if "__" in k})
        Q = optical_quantities(e["raw"].astype(float))
        ok = np.all([np.isfinite(v).all(1) for k, v in feats.items() if k != "meta"], axis=0)
        ok &= np.all([np.isfinite(v) for v in Q.values()], axis=0)   # every quantity (joint fit)
        if ok.sum() < 50 or len(set(month[ok])) < n_folds:
            log(f"retention {window}: {ok.sum()} samples; skipped")
            continue
        names = list(Q)
        Ym = np.c_[tuple(Q[k][ok] for k in names)]
        g = month[ok]
        folds = list(GroupKFold(n_splits=n_folds).split(Ym, groups=g))
        for name, X in feats.items():
            X = X[ok].astype(float)
            P = np.full(Ym.shape, np.nan)
            for tr, va in folds:   # alpha per quantity by grouped CV inside the training folds
                P[va] = ridge_probe(X[tr], Ym[tr], g[tr], X[va])[0]
            for j, q in enumerate(names):
                y, p = Ym[:, j], P[:, j]
                ss = ((y - y.mean()) ** 2).sum()
                rows.append({"window": window, "features": name, "quantity": q, "n": len(y),
                             "cv_r2": float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan})
        log(f"retention {window}: {ok.sum()} samples, {len(set(g))} months, {len(feats)} feature sets")
    if rows:
        r = pd.DataFrame(rows)
        r.to_csv(out_dir / "robust_retention.csv", index=False)
        show = r[r["quantity"].isin(["log_brightness", "log_blue_green_oc3m", "band_443", "band_547"])]
        log("Retention of input reflectance quantities (cross-validated R^2, grouped by month). All bands: "
            "robust_retention.csv")
        log(show.pivot_table(index=["window", "quantity"], columns="features", values="cv_r2").round(3).to_string())


# ----------------------------------------------------------------------------- amplitude
MODIS_SHAPE = np.array([0.0080, 0.0072, 0.0066, 0.0060, 0.0042, 0.0036, 0.0033, 0.0004, 0.0003, 0.0003])


def _probe_r2(X: np.ndarray, y: np.ndarray, n_train: int) -> float:
    Xs = (X - X[:n_train].mean(0)) / np.where(X[:n_train].std(0) > 1e-8, X[:n_train].std(0), 1.0)
    Xc = np.c_[Xs, np.ones(len(Xs))]
    a = Xc[:n_train]
    w = np.linalg.solve(a.T @ a + 1e-2 * np.eye(Xc.shape[1]), a.T @ y[:n_train])
    p = Xc[n_train:] @ w
    yt = y[n_train:]
    return float(1 - ((p - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())


@torch.no_grad()
def amplitude_check(model, norm, n: int = 300, seed: int = 0, device=None) -> list[dict]:
    """Synthetic uniform 32 x 32 scenes of one MODIS-Aqua spectrum (3 observations at dt -1, 0, 1).
    Token level (spatial-stage input, cell 0) and E1 level (mean over cells). Returns rows of
    (level, metric, value)."""
    from ..spectral import nominal_fwhm
    from .data import MODIS_REF_WL, collate, to_device

    device = device or next(model.parameters()).device
    wl = MODIS_REF_WL.astype(float)
    rng = np.random.default_rng(seed)

    def obs(spec, dt):
        raw = np.repeat(np.repeat(spec[:, None, None], 32, 1), 32, 2)
        y, bm = norm.transform("rrs", wl, raw)
        o = {"product": "modisa_oc", "family": "MODIS", "view": "rrs", "olci": False, "wl": wl,
             "fwhm": nominal_fwhm("MODIS", wl), "values": np.nan_to_num(y).astype(np.float16), "band_mask": bm,
             "quality": np.zeros((9, 32, 32), np.uint8), "cell_valid": bm.any(0).ravel(),
             "raw_mean": np.full(1024, float(spec.mean()), np.float32), "dt_days": float(dt), "is_pace": False}
        o.update(ctx_cells=o["cell_valid"].copy(), tgt_cells=np.zeros(1024, bool), band_mask_ctx=bm)
        return o

    def run(specs):
        tok_out, e1_out = [], []
        for i in range(0, len(specs), 16):
            items = [{"obs": [obs(s, dt) for dt in (-1, 0, 1)], "mode": "eval"} for s in specs[i:i + 16]]
            b = to_device(collate(items), device)
            g = b["groups"][0]
            tok = model.tok(g, g["band_mask_ctx"])                  # [n_obs, 1024, D]
            tok_out.append(tok[g["slot"] == 1][:, 0].float().cpu().numpy())
            e1, m = model.embed_e1(b)
            e1_out.append((e1 * m[..., None]).sum(1).div(m.sum(1, keepdim=True).clamp(min=1)).float().cpu().numpy())
        return np.concatenate(tok_out), np.concatenate(e1_out)

    rows = []
    t1, e1 = run([MODIS_SHAPE, 0.5 * MODIS_SHAPE, 2.0 * MODIS_SHAPE])
    for level, z in (("token", t1), ("e1", e1)):
        for k, scale in ((1, 0.5), (2, 2.0)):
            rows.append({"level": level, "metric": f"rel_distance_x{scale}",
                         "value": float(np.linalg.norm(z[0] - z[k]) / np.linalg.norm(z[0]))})
    specs, amp, ratio = [], [], []
    for _ in range(n):
        sh = MODIS_SHAPE * np.exp(rng.normal(0, 0.3, len(wl)))
        a = float(np.exp(rng.uniform(np.log(0.25), np.log(4.0))))
        specs.append(a * sh)
        amp.append(np.log(a))
        ratio.append(np.log10(max(sh[1], sh[3]) / sh[5]))
    tk, em = run(specs)
    n_tr = int(0.67 * n)
    for level, z in (("token", tk), ("e1", em)):
        rows.append({"level": level, "metric": "log_amplitude_r2", "value": _probe_r2(z, np.array(amp), n_tr)})
        rows.append({"level": level, "metric": "log_blue_green_r2", "value": _probe_r2(z, np.array(ratio), n_tr)})
    return rows


def amplitude(cfg, models, out_dir: Path, log=print) -> None:
    import pandas as pd

    from ..normstats import stats_dir
    from ..spectral import Normalizer

    norm = Normalizer.load(stats_dir(cfg) / "normstats.json")
    rows = []
    for model, info in models:
        model.float()
        for r in amplitude_check(model, norm):
            rows.append({"run": info["run"], "arm": info["arm"], "step": info["step"], **r,
                         # design-document test: amplitude only; the blue/green probe is reported, not judged
                         "pass": (None if "blue_green" in r["metric"] else
                                  bool(r["value"] >= 0.95) if r["metric"].endswith("_r2") else bool(r["value"] > 0.05))})
    r = pd.DataFrame(rows)
    r.to_csv(out_dir / "amplitude_check.csv", index=False)
    log("Amplitude check on trained weights (design document: log-amplitude R^2 >= 0.95, relative distance "
        "> 0.05; blue/green R^2 reported only):")
    r["cell"] = r["value"].round(3).astype(str) + np.where(r["pass"] == False, " FAIL", "")  # noqa: E712
    log(r.pivot_table(index=["level", "metric"], columns="run", values="cell", aggfunc="first").to_string())
