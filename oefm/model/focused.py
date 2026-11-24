"""Focused downstream tests (step 2.10c): what PACE can see, and what fish gain over habitat.

    python -m oefm.model.evaluate --runs ... --steps focused [--focused phyto fish]
        --windows causal retro retro1 retro3 seasonal

Both use the embeddings of `--steps embed` (frozen encoders) and E1 only.

phyto   PACE-era phytoplankton test. Targets: the CLR phytoplankton groups (phyto_*; 18S diatoms,
        Dinophyceae, haptophytes, prasinophytes, other Chlorophyta; 16S Prochlorococcus,
        Synechococcus, plastids; scripts/make_eval_targets.py), not whole-community NMF. Samples:
        eDNA with >= 1 usable PACE observation in the 3 x 3 cells of this window (all are 2024) and,
        when the MOANA matchups exist, a MOANA value in the matching window (+/- 1 d for retro1,
        +/- 3 d otherwise) so that every feature set is scored on the same samples. Leave-one-
        cruise-out (grouped K-fold, <= 5 folds), ridge probes with the sequencing covariates
        (unpenalised), alpha by grouped CV inside the training cruises. Feature sets: seq only,
        meta, raw (MODIS-band Rrs), MOANA (log10 Prochlorococcus, Synechococcus, picoeukaryotes:
        NASA's community product from the same OCI spectra; oefm/moana.py), and every run's E1
        with PACE visible and dropped. Paired bootstrap within cruises of E1 visible - dropped and
        of E1 visible - MOANA. -> focused_phyto.csv (per target), focused_phyto_summary.csv
fish    Added value over habitat. Habitat covariates (hab_*: CRCP survey depth and habitat type;
        MaxN GEBCO depth and distance to shore) enter every model unpenalised; the baseline is
        habitat alone ('habitat'), and each feature set adds a penalised block (meta, raw, every
        run's E1). Counts get a Poisson ridge GLM (CRCP counts per survey unit = expm1 of the
        log1p targets; MaxN counts), alpha by grouped CV (site / month) on held-out Poisson
        deviance; features standardised and reduced to <= 50 principal components fitted on the
        training labels (same for every feature set). Windows: causal (30 d) and seasonal (mean of
        three causal 30 d windows, 0-90 d before the survey). Metrics per species: D^2 (deviance
        explained vs the test mean, as R^2: penalises a year-level shift), D^2 after rescaling the
        predictions by one factor to the test total ('d2_level_free': ranks and relative
        contrasts only; uses one number from the test labels, a diagnostic, not a skill score)
        and Spearman. Official split (2024 test) with paired bootstrap of the difference to
        'habitat', and test-year rotation. -> focused_fish.csv (per species, split),
        focused_fish_summary.csv

Caveats (printed): eDNA PACE-era samples are 3 cruises in 2024 (~100 per marker), so phyto
intervals are wide; CRCP sites are optically shallow (bottom reflectance in Rrs).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..targets import eval_dir
from .evaluate import _standardise, fit_predict, probe_inputs
from .robust import _boot_r2, _strata_resamples

PACE_START = "2024-03-05"
N_PC = 50
POISSON_ALPHAS = np.logspace(-1, 4, 8)


# ----------------------------------------------------------------------------- phyto
def _moana(cfg, df, window: str):
    import pandas as pd

    f = eval_dir(cfg) / "moana_matchups.csv.gz"
    if not f.exists():
        return None
    m = pd.read_csv(f).drop_duplicates("key").set_index("key")
    w = 1 if window == "retro1" else 3
    cols = [f"moana_{g}_pm{w}" for g in ("pro", "syn", "pico")]
    if not all(c in m.columns for c in cols):
        return None
    return m.reindex(df["key"])[cols].to_numpy(float)


def phyto(cfg, out_dir: Path, windows, n_boot: int = 1000, log=print) -> None:
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    rows, summ = [], []
    for window in windows:
        for inp in probe_inputs(cfg, out_dir, window, ["visible", "drop"], "on", log):
            ds, df = inp["ds"], inp["df"]
            if not ds.startswith("edna"):
                continue
            tsel = [j for j, c in enumerate(inp["tcols"]) if c.startswith("phyto_")]
            if not tsel:
                log(f"focused phyto {ds}/{window}: no phyto_* targets (rebuild targets: make_eval_targets.py, evalsets)")
                continue
            tcols = [inp["tcols"][j] for j in tsel]
            Y = inp["Y"][:, tsel]
            C, groups = inp["C"], inp["groups"]
            era = (pd.to_datetime(df["time_utc"], utc=True) >= pd.Timestamp(PACE_START, tz="UTC")).to_numpy()
            ok = inp["common"] & era & (inp["n_pace"]["visible"] > 0) & np.isfinite(Y).all(1)
            feats = {("seq", "-"): np.zeros((len(df), 0)), ("meta", "-"): inp["fsets"][("meta", "visible")],
                     ("raw", "visible"): inp["fsets"][("raw", "visible")], ("raw", "drop"): inp["fsets"][("raw", "drop")]}
            M = _moana(cfg, df, window)
            if M is not None:
                ok &= np.isfinite(M).all(1)
                feats[("moana", "-")] = M
            for (name, pace), X in inp["fsets"].items():
                if name.endswith("/e1"):
                    feats[(name, pace)] = X
            gs = sorted(set(groups[ok]))
            if ok.sum() < 30 or len(gs) < 3:
                log(f"focused phyto {ds}/{window}: {ok.sum()} PACE-era samples in {len(gs)} cruises; skipped")
                continue
            idx = np.where(ok)[0]
            folds = []
            for _, va in GroupKFold(n_splits=min(5, len(gs))).split(idx, groups=groups[idx]):
                te = np.zeros(len(df), bool)
                te[idx[va]] = True
                folds.append((ok & ~te, te))
            preds = {}
            for key, X in feats.items():
                P = np.full(Y.shape, np.nan)
                for tr, te in folds:
                    P[te] = fit_predict(key[0], X, Y, tr, groups, C)[0][te]
                preds[key] = P[ok]
            Yo, I = Y[ok], _strata_resamples(groups[ok], n_boot, seed=20261125)
            boots = {k: np.stack([_boot_r2(Yo[:, j], P[:, j], I) for j in range(len(tcols))], 1) for k, P in preds.items()}
            base = {"dataset": ds, "window": window, "n": int(ok.sum()), "n_cruises": len(gs),
                    "moana": M is not None, "covariates": "seq" if C is not None else "none"}
            for (name, pace), P in preds.items():
                for j, c in enumerate(tcols):
                    y, p = Yo[:, j], P[:, j]
                    ss = ((y - y.mean()) ** 2).sum()
                    rows.append({**base, "features": name, "pace": pace, "target": c,
                                 "r2": float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan,
                                 "spearman": float(spearmanr(y, p)[0]) if np.std(p) > 0 else np.nan,
                                 "r2_lo": np.nanpercentile(boots[(name, pace)][:, j], 2.5),
                                 "r2_hi": np.nanpercentile(boots[(name, pace)][:, j], 97.5)})
                med = np.nanmedian(boots[(name, pace)], 1)
                rec = {**base, "features": name, "pace": pace,
                       "r2_median": float(np.nanmedian([r["r2"] for r in rows[-len(tcols):]])),
                       "ci_lo": np.nanpercentile(med, 2.5), "ci_hi": np.nanpercentile(med, 97.5)}
                for other, label in (((name, "drop"), "vs_drop"), (("moana", "-"), "vs_moana")):
                    if pace == "visible" and name.endswith("/e1") and other in boots and other != (name, pace):
                        d = med - np.nanmedian(boots[other], 1)
                        rec[f"delta_{label}"] = np.nan   # point value filled below (the other set may come later)
                        rec[f"delta_{label}_lo"], rec[f"delta_{label}_hi"] = np.nanpercentile(d, [2.5, 97.5])
                        rec[f"p_better_{label}"] = float(np.nanmean(d > 0))
                summ.append(rec)
            log(f"focused phyto {ds}/{window}: {ok.sum()} PACE-era samples, {len(gs)} cruises, "
                f"{len(tcols)} targets, MOANA {'yes' if M is not None else 'not available (run targets.sbatch STEPS=moana)'}")
    if not rows:
        return
    pd.DataFrame(rows).to_csv(out_dir / "focused_phyto.csv", index=False)
    s = pd.DataFrame(summ)
    med = s.set_index(["dataset", "window", "features", "pace"])["r2_median"]
    for label, ref in (("vs_drop", None), ("vs_moana", ("moana", "-"))):
        col = f"delta_{label}"
        if col not in s:
            continue
        for i in s.index[s[col + "_lo"].notna()]:
            r = s.loc[i]
            other = (r["features"], "drop") if ref is None else ref
            s.loc[i, col] = r["r2_median"] - med.get((r["dataset"], r["window"], *other), np.nan)
    s.to_csv(out_dir / "focused_phyto_summary.csv", index=False)
    s["cell"] = s["r2_median"].round(3).astype(str) + " [" + s["ci_lo"].round(2).astype(str) + ", " + s["ci_hi"].round(2).astype(str) + "]"
    log("PACE-era phytoplankton groups (leave-one-cruise-out): median R^2 over groups [95% bootstrap within cruises]")
    log(s.pivot_table(index=["dataset", "window", "n"], columns=["features", "pace"], values="cell", aggfunc="first").T.to_string())
    d = s[s["features"].str.endswith("/e1") & (s["pace"] == "visible")]
    cols = [c for c in ("delta_vs_drop", "delta_vs_drop_lo", "delta_vs_drop_hi", "p_better_vs_drop",
                        "delta_vs_moana", "delta_vs_moana_lo", "delta_vs_moana_hi", "p_better_vs_moana") if c in d]
    log("E1 PACE visible minus dropped, and minus MOANA:")
    log(d[["dataset", "window", "features"] + cols].round(3).to_string(index=False))
    log(f"-> {out_dir / 'focused_phyto.csv'}, {out_dir / 'focused_phyto_summary.csv'}")


# ----------------------------------------------------------------------------- Poisson ridge
def poisson_fit(X, C, y, alpha: float, iters: int = 50):
    """Poisson GLM, log link: minimise sum(mu - y eta) + alpha/2 |w|^2, eta = b0 + C c + X w; the
    intercept and the covariates C are unpenalised. Newton with step halving. Returns beta over
    [1, C, X]."""
    Z = np.c_[np.ones(len(y)), C, X]
    k = 1 + C.shape[1]
    pen = np.r_[np.zeros(k), np.ones(X.shape[1])] * alpha
    b = np.zeros(Z.shape[1])
    b[0] = np.log(max(y.mean(), 1e-3))

    def obj(b):
        eta = np.clip(Z @ b, -30, 20)
        return np.exp(eta).sum() - y @ eta + 0.5 * (pen * b * b).sum()

    f = obj(b)
    for _ in range(iters):
        mu = np.exp(np.clip(Z @ b, -30, 20))
        g = Z.T @ (mu - y) + pen * b
        H = (Z * mu[:, None]).T @ Z + np.diag(pen + 1e-8)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while t > 1e-4:
            nb = b - t * step
            nf = obj(nb)
            if nf <= f:
                break
            t /= 2
        if t <= 1e-4:
            break
        b, done = nb, f - nf < 1e-9 * (1 + abs(f))
        f = nf
        if done:
            break
    return b


def poisson_predict(X, C, b):
    return np.exp(np.clip(np.c_[np.ones(len(X)), C, X] @ b, -30, 20))


def poisson_deviance(y, mu):
    mu = np.maximum(mu, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * (t - (y - mu))


def _pca(Xtr, Xte, n: int = N_PC):
    Xtr, Xte = _standardise(Xtr, Xte)
    if Xtr.shape[1] <= n:
        return Xtr, Xte
    _, _, vt = np.linalg.svd(Xtr - Xtr.mean(0), full_matrices=False)
    V = vt[:n].T
    a, b = Xtr @ V, Xte @ V
    sd = a.std(0)
    sd[sd < 1e-8] = 1
    return a / sd, b / sd


def poisson_probe(Xtr, Ctr, ytr, gtr, Xte, Cte, n_folds: int = 5):
    """alpha by grouped CV on held-out deviance; returns test means and the alpha."""
    from sklearn.model_selection import GroupKFold

    ng = len(set(gtr))
    best = POISSON_ALPHAS[len(POISSON_ALPHAS) // 2]
    if Xtr.shape[1] and ng >= 2:
        score = np.zeros(len(POISSON_ALPHAS))
        for a_ix, b_ix in GroupKFold(n_splits=min(n_folds, ng)).split(Xtr, groups=gtr):
            Xa, Xb = _standardise(Xtr[a_ix], Xtr[b_ix])
            Ca, Cb = _standardise(Ctr[a_ix], Ctr[b_ix])
            for i, al in enumerate(POISSON_ALPHAS):
                score[i] += poisson_deviance(ytr[b_ix], poisson_predict(Xb, Cb, poisson_fit(Xa, Ca, ytr[a_ix], al))).sum()
        best = POISSON_ALPHAS[int(np.argmin(score))]
    Xa, Xb = _standardise(Xtr, Xte)
    Ca, Cb = _standardise(Ctr, Cte)
    return poisson_predict(Xb, Cb, poisson_fit(Xa, Ca, ytr, best)), best


def _fish_counts(df, ds):
    if ds == "fish_crcp":
        cols = [c for c in df.columns if c.startswith("log1p_")]
        return [c[len("log1p_"):] for c in cols], np.rint(np.expm1(df[cols].to_numpy(float)))
    cols = [c for c in df.columns if c.startswith("maxn_")]
    return [c[len("maxn_"):] for c in cols], df[cols].to_numpy(float)


def _d2(y, mu, I=None):
    """Deviance explained vs the (resampled) test mean; and after rescaling mu to the test total."""
    if I is None:
        I = np.arange(len(y))[None, :]
    yb, mb = y[I], mu[I]
    null = yb.mean(1, keepdims=True)
    dn = poisson_deviance(yb, np.broadcast_to(null, yb.shape)).sum(1)
    dm = poisson_deviance(yb, mb).sum(1)
    scale = yb.sum(1, keepdims=True) / np.maximum(mb.sum(1, keepdims=True), 1e-12)
    dr = poisson_deviance(yb, mb * scale).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dn > 0, 1 - dm / dn, np.nan), np.where(dn > 0, 1 - dr / dn, np.nan)


def fish(cfg, out_dir: Path, windows, n_boot: int = 1000, rotate: bool = True, period: str = "year",
         min_period: int = 30, log=print) -> None:
    import pandas as pd
    from scipy.stats import spearmanr

    from .robust import _period

    rows, summ = [], []
    for window in windows:
        for inp in probe_inputs(cfg, out_dir, window, ["visible"], "off", log):
            ds, df, groups = inp["ds"], inp["df"], inp["groups"]
            if not ds.startswith("fish"):
                continue
            hcols = [c for c in df.columns if c.startswith("hab_")]
            if not hcols:
                log(f"focused fish {ds}/{window}: no hab_* covariates (rebuild targets: make_eval_targets.py, evalsets)")
                continue
            H = df[hcols].to_numpy(float)
            species, Ycnt = _fish_counts(df, ds)
            feats = {"habitat": np.zeros((len(df), 0)), "meta": inp["fsets"][("meta", "visible")],
                     "raw": inp["fsets"][("raw", "visible")]}
            feats.update({n: X for (n, p), X in inp["fsets"].items() if n.endswith("/e1")})
            ok = inp["common"] & np.isfinite(H).all(1)
            per = _period(df["time_utc"], period)
            off_te = (df["label_split"] == "test").to_numpy()
            splits = [("official", ok & ~off_te, ok & off_te)]
            if rotate:  # every other year with enough samples (the official test year is 'official')
                off_years = set(per[ok & off_te])
                splits += [(y, ok & (per != y), ok & (per == y)) for y in sorted(set(per[ok]) - off_years)]
            for split, tr, te in splits:
                if tr.sum() < 50 or te.sum() < min_period:
                    continue
                I = _strata_resamples(groups[te], n_boot, seed=20261126) if split == "official" else None
                preds = {}
                for name, X in feats.items():
                    X = X.astype(float)
                    if name == "raw":
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            mu = np.nanmean(X[tr], 0)
                        X = np.where(np.isfinite(X), X, np.where(np.isfinite(mu), mu, 0.0))
                    Xa, Xb = _pca(X[tr], X[te]) if X.shape[1] else (X[tr], X[te])
                    P = np.full((te.sum(), len(species)), np.nan)
                    for j in range(len(species)):
                        y = Ycnt[tr, j]
                        fin = np.isfinite(y)
                        if (y[fin] > 0).sum() < 10:
                            continue
                        P[:, j] = poisson_probe(Xa[fin], H[tr][fin], y[fin], groups[tr][fin], Xb, H[te])[0]
                    preds[name] = P
                Yt = Ycnt[te]
                boot_d2 = {}
                for name, P in preds.items():
                    bd = []
                    for j, sp in enumerate(species):
                        y, mu = Yt[:, j], P[:, j]
                        if not np.isfinite(mu).all() or not np.isfinite(y).all() or y.sum() == 0:
                            continue
                        d2, d2r = _d2(y, mu)
                        rows.append({"dataset": ds, "window": window, "split": split, "features": name,
                                     "species": sp, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                                     "d2": float(d2[0]), "d2_level_free": float(d2r[0]),
                                     "spearman": float(spearmanr(y, mu)[0]) if np.std(mu) > 0 and np.std(y) > 0 else np.nan,
                                     "test_mean": float(y.mean()), "pred_mean": float(mu.mean())})
                        if I is not None:
                            bd.append(_d2(y, mu, I)[0])
                    if bd:
                        boot_d2[name] = np.nanmedian(np.stack(bd, 1), 1)
                r = pd.DataFrame([x for x in rows if x["dataset"] == ds and x["window"] == window and x["split"] == split])
                if r.empty:
                    continue
                med = r.groupby("features")[["d2", "d2_level_free", "spearman"]].median()
                for name in med.index:
                    rec = {"dataset": ds, "window": window, "split": split, "features": name, "n_test": int(te.sum()),
                           "n_species": int((r["features"] == name).sum()), **med.loc[name].to_dict()}
                    rec["delta_d2_vs_habitat"] = rec["d2"] - med.loc["habitat", "d2"] if "habitat" in med.index else np.nan
                    rec["delta_spearman_vs_habitat"] = rec["spearman"] - med.loc["habitat", "spearman"] if "habitat" in med.index else np.nan
                    if I is not None and name in boot_d2 and "habitat" in boot_d2:
                        d = boot_d2[name] - boot_d2["habitat"]
                        rec["delta_d2_lo"], rec["delta_d2_hi"] = np.nanpercentile(d, [2.5, 97.5])
                        rec["p_better_than_habitat"] = float(np.nanmean(d > 0))
                    summ.append(rec)
                log(f"focused fish {ds}/{window}/{split}: train {tr.sum()}, test {te.sum()}, {len(species)} species")
    if not rows:
        return
    pd.DataFrame(rows).to_csv(out_dir / "focused_fish.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(out_dir / "focused_fish_summary.csv", index=False)
    off = s[s["split"] == "official"]
    log("Fish, official split, Poisson ridge with habitat covariates: median over species "
        "(D^2 | D^2 level-free | Spearman); delta D^2 vs habitat [95% bootstrap]")
    cols = [c for c in ("d2", "d2_level_free", "spearman", "delta_d2_vs_habitat", "delta_d2_lo", "delta_d2_hi",
                        "p_better_than_habitat") if c in off]
    log(off[["dataset", "window", "features"] + cols].round(3).to_string(index=False))
    rot = s[s["split"] != "official"]
    if len(rot):
        log("Test-year rotation: mean over held-out years of the median D^2 and Spearman")
        log(rot.groupby(["dataset", "window", "features"])[["d2", "d2_level_free", "spearman", "delta_d2_vs_habitat"]]
            .mean().round(3).to_string())
    log("Caveat: CRCP sites are optically shallow (reef bottom reflectance contaminates Rrs).")
    log(f"-> {out_dir / 'focused_fish.csv'}, {out_dir / 'focused_fish_summary.csv'}")


def focused(cfg, out_dir: Path, windows, parts=("phyto", "fish"), n_boot: int = 1000, log=print) -> None:
    if "phyto" in parts:
        phyto(cfg, out_dir, [w for w in windows if w != "seasonal"], n_boot, log)
    if "fish" in parts:
        fish(cfg, out_dir, [w for w in windows if w in ("causal", "seasonal")], n_boot, log=log)
