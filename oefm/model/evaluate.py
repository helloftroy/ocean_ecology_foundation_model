"""Stage 1 evaluation (step 2.10): frozen embeddings, linear probes, PACE spectral retention.

    python -m oefm.model.evaluate --runs stage1_A stage1_B stage1_C [--steps embed probes pace]

Everything is frozen: no backbone fine-tuning (design document, "Downstream protocol"). Outputs go
to data/runs/eval_<tag>/.

embed   Embeddings at every evaluation sample (eval/matched/*.csv.gz; 16S and 18S share samples, so
        each place and time is encoded once). Sequences come from all imagery around the sample
        (split 'all': the label hold-out concerns labels, not imagery), length as in training,
        two windows: causal (30 d up to the sample) and retro (+/- 15 d), and two PACE settings:
        visible and dropped (PACE on/off). Per run: E1 (valid-weighted mean of temporal-stage tokens),
        E2 (predictor query at dt = 0 with MODIS-Aqua bands; arms B and C), E3 (E1 + spread, 2D),
        each averaged over the valid cells of the 3 x 3 cells around the sample (the coverage rule of
        2.4b). Baselines from the same sequences: 'meta' (lat, lon, day of year) and 'raw' (raw Rrs
        of every observation interpolated to the MODIS bands, averaged, asinh-scaled).
        -> embeddings_<window>_<pace>.npz

probes  Ridge probes, identical for every feature set: standardised features, ridge penalty chosen
        per target by grouped 5-fold cross-validation on the training labels (groups: cruise for
        eDNA and SFER, site for CRCP, month for MaxN), refitted on all training labels, scored on the
        held-out labels (eDNA, CRCP, MaxN: 2024; SFER: 2025). R^2 and Spearman for continuous targets,
        AUC for presence (linear-probability ridge). Every feature set of one dataset and window is
        scored on the same samples (those where all of them exist). The PACE on/off comparison is
        repeated on the test samples with >= 1 usable PACE observation.
        -> probes.csv (per target), probes_summary.csv (mean per target family)

pace    PACE spectral retention: anchors = PACE observations (retro window), the anchor visible in
        the context ('visible': can the embedding give back the spectrum it saw?) or every PACE
        observation removed ('hidden': what the heritage sensors and time imply). One ridge decoder
        per run and setting, E1 -> the normalised PACE spectrum (all bands), fitted on training-split
        cells, scored on April-October 2026 (test_temporal) and the geographic test blocks
        (test_geo). Per wavelength RMSE (normalised units) and R^2. Baselines: training mean
        spectrum, and the heritage-band ridge of arm C (pace_ridge.npz) applied to the same PACE
        observation. -> pace_retention.csv (per wavelength), pace_retention_summary.csv
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..loader import TileSequences
from ..normstats import stats_dir
from ..spectral import Normalizer
from ..targets import eval_dir
from .data import MODIS_REF_WL, TrainData, collate, to_device
from .net import Model, effective_rank, embed_e3

META_COLS = ["sample_id", "dataset", "lat", "lon", "time_utc", "label_split"]
NUTRIENTS = ("po4", "no3", "si", "chl")


# ----------------------------------------------------------------------------- models
def load_run(cfg, run: str, weights: str, device) -> tuple[Model, dict]:
    d = cfg.data_root / "runs" / run
    cands = {"final": d / "weights_final.pt", "checkpoint": d / "checkpoint.pt"}
    p = cands[weights] if weights in cands else Path(weights)
    if weights == "final" and not p.exists():
        p = cands["checkpoint"]
    st = torch.load(p, map_location="cpu", weights_only=False)
    m = Model(st["config"])
    m.load_state_dict(st["model"])
    m.eval().to(device)
    m.train_config = st["config"]
    return m, {"run": run, "arm": st["config"]["arm"], "step": int(st.get("step", -1)), "weights": str(p)}


def _obs_maker(cfg, with_ridge: bool = False):
    """TrainData's per-observation preparation (normalisation, masks, quality, amplitude) without
    building a training index."""
    te = TrainData.__new__(TrainData)
    te.cfg = cfg
    te.norm = Normalizer.load(stats_dir(cfg) / "normstats.json")
    rp = stats_dir(cfg) / "pace_ridge.npz"
    te.ridge = dict(np.load(rp)) if (with_ridge and rp.exists()) else None
    return te


def _as_context(x: dict) -> dict:
    x["ctx_cells"] = x["cell_valid"].copy()
    x["tgt_cells"] = np.zeros(1024, bool)
    x["band_mask_ctx"] = x["band_mask"]
    return x


def _nb_cells(cell: tuple[int, int], n: int = 32) -> np.ndarray:
    r, c = cell
    return np.array([i * n + j for i in range(max(0, r - 1), min(n, r + 2)) for j in range(max(0, c - 1), min(n, c + 2))])


def _raw_at(raw12: np.ndarray, wl: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """Raw Rrs of one observation at the 3 x 3 cells, interpolated to the MODIS bands (NaN outside
    the observation's band range)."""
    flat = raw12.reshape(len(wl), -1)[:, nb]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spec = np.nanmean(flat, axis=1)
    ok = np.isfinite(spec)
    out = np.full(len(MODIS_REF_WL), np.nan)
    if ok.sum() >= 2:
        w, v = wl[ok], spec[ok]
        inside = (MODIS_REF_WL >= w.min() - 15) & (MODIS_REF_WL <= w.max() + 15)
        out[inside] = np.interp(MODIS_REF_WL[inside], w, v)
    return out


# ----------------------------------------------------------------------------- embed
class PointSet(torch.utils.data.Dataset):
    def __init__(self, cfg, points, window: str, pace: str, length: int, sensors) -> None:
        self.pts = points.reset_index(drop=True)
        self.seq = TileSequences(cfg, split="all", mode=window, length=length, anchors="pace", sensors=sensors,
                                 pace="drop" if pace == "drop" else "visible", selection="recent",
                                 include_300m=True, physics=False)
        self.te = _obs_maker(cfg)

    def __len__(self) -> int:
        return len(self.pts)

    def __getitem__(self, k: int) -> dict:
        r = self.pts.iloc[k]
        s = self.seq.sequence_for_point(float(r["lat"]), float(r["lon"]), str(r["time_utc"]))
        item = {"k": k, "obs": [], "mode": "eval", "raw": np.full(len(MODIS_REF_WL), np.nan), "nb": None,
                "n_obs": 0, "n_pace": 0}
        if s is None:
            return item
        nb = _nb_cells(s["cell"])
        raws, obs = [], []
        for o in s["obs"]:
            x = self.te._obs(o, s["anchor_time"], None)
            if not x["cell_valid"].any():
                continue
            if x["view"] == "rrs" and x["cell_valid"][nb].any():
                raws.append(_raw_at(x["_raw12"], x["wl"], nb))
            x.pop("_raw12", None)
            obs.append(_as_context(x))
        if raws:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                item["raw"] = np.nanmean(np.stack(raws), axis=0)
        item.update(obs=obs, nb=nb if obs else None, n_obs=sum(bool(o["cell_valid"][nb].any()) for o in obs),
                    n_pace=sum(bool(o["is_pace"] and o["cell_valid"][nb].any()) for o in obs))
        return item


def _collate_points(items: list[dict]) -> tuple[dict | None, list[dict]]:
    keep = [it for it in items if it["obs"]]
    meta = [{k: it[k] for k in ("k", "raw", "nb", "n_obs", "n_pace")} for it in items]
    return (collate(keep) if keep else None), meta


@torch.no_grad()
def _point_embeddings(model: Model, batch: dict, nbs: list[np.ndarray]) -> dict[str, np.ndarray]:
    """E1, E2 (B/C), E3 averaged over the valid cells among each sample's 3 x 3 cells."""
    z, gmask, _ = model.encode_context(batch)
    e1, m = model.embed_e1(batch, z, gmask)
    out = {"e1": e1, "e3": embed_e3(e1, z, gmask)}
    if model.arm in ("B", "C"):
        out["e2"] = model.embed_e2(batch).float()
    res = {}
    for name, e in out.items():
        rows = []
        for b, nb in enumerate(nbs):
            idx = torch.as_tensor(nb, device=e.device)
            # E2 exists in every cell; E1/E3 only where a context observation is valid
            w = torch.ones(len(nb), device=e.device) if name == "e2" else m[b, idx].float()
            rows.append((e[b, idx].float() * w[:, None]).sum(0) / w.sum() if w.sum() > 0
                        else torch.full((e.shape[-1],), float("nan"), device=e.device))
        res[name] = torch.stack(rows).cpu().numpy().astype(np.float32)
    return res


def load_points(cfg, limit: int = 0):
    import pandas as pd

    frames = []
    for f in sorted((eval_dir(cfg) / "matched").glob("*.csv.gz")):
        t = pd.read_csv(f, dtype={"sample_id": str}, usecols=lambda c: c in META_COLS)
        frames.append(t)
    if not frames:
        raise SystemExit(f"No matched evaluation sets in {eval_dir(cfg) / 'matched'} (run targets.sbatch, step 2.7)")
    pts = pd.concat(frames, ignore_index=True)
    pts["key"] = pts["lat"].round(5).astype(str) + "|" + pts["lon"].round(5).astype(str) + "|" + pts["time_utc"].astype(str)
    uniq = pts.drop_duplicates("key")[["key", "lat", "lon", "time_utc"]].reset_index(drop=True)
    if limit:
        uniq = uniq.iloc[:limit]
    return uniq


def embed(cfg, models: list[tuple[Model, dict]], out_dir: Path, tcfg: dict, windows, paces, batch: int = 16,
          workers: int = 8, limit: int = 0, log=print) -> None:
    pts = load_points(cfg, limit)
    device = next(models[0][0].parameters()).device
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for window in windows:
        for pace in paces:
            t0 = time.time()
            ds = PointSet(cfg, pts, window, pace, int(tcfg["data"]["length"]), tcfg["data"].get("sensors"))
            dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, collate_fn=_collate_points)
            n = len(pts)
            feats: dict[str, np.ndarray] = {}
            raw = np.full((n, len(MODIS_REF_WL)), np.nan, np.float32)
            n_obs = np.zeros(n, np.int16)
            n_pace = np.zeros(n, np.int16)
            for bt, meta in dl:
                for mm in meta:
                    raw[mm["k"]], n_obs[mm["k"]], n_pace[mm["k"]] = mm["raw"], mm["n_obs"], mm["n_pace"]
                if bt is None:
                    continue
                # collate keeps items with observations, in order; those are exactly the ones with nb set
                has = [mm for mm in meta if mm["nb"] is not None]
                bt = to_device(bt, device)
                for model, info in models:
                    with torch.autocast(**amp):
                        res = _point_embeddings(model, bt, [h["nb"] for h in has])
                    for name, arr in res.items():
                        key = f"{info['run']}__{name}"
                        if key not in feats:
                            feats[key] = np.full((n, arr.shape[1]), np.nan, np.float32)
                        feats[key][[h["k"] for h in has]] = arr
            dest = out_dir / f"embeddings_{window}_{pace}.npz"
            np.savez_compressed(dest, key=pts["key"].to_numpy().astype(str), raw=raw, n_obs=n_obs, n_pace=n_pace, **feats)
            msg = [f"embed {window}/{pace}: {n:,} samples, {(n_obs > 0).sum():,} with a valid observation "
                   f"in the 3x3 cells ({(n_pace > 0).sum():,} with PACE), {time.time() - t0:.0f} s"]
            for k, v in feats.items():
                ok = np.isfinite(v).all(1)
                if ok.sum() > 10:
                    msg.append(f"  {k}: dim {v.shape[1]}, effective rank {effective_rank(torch.from_numpy(v[ok])):.1f}")
            log("\n".join(msg))


# ----------------------------------------------------------------------------- probes
def group_of(dataset: str, sample_id: str, time_utc: str) -> str:
    """Cross-validation groups: cruise (eDNA, SFER), site (CRCP), month (MaxN and fallback)."""
    if dataset.startswith(("edna", "sfer")):
        m = re.search(r"(WS\d{5}|[A-Z]{1,3}\d{4,5})", sample_id)
        if m:
            return m.group(1)
    if dataset == "fish_crcp":
        return sample_id.split("_")[0]
    return str(time_utc)[:7]


def target_columns(df) -> list[str]:
    skip = set(META_COLS) | {"tile_id", "usable_any", "year", "window", "full_window_in_pace_era", "key"}
    return [c for c in df.columns if c not in skip and not c.startswith(("causal_", "sameday_"))
            and np.issubdtype(df[c].dtype, np.number)]


def transform_target(name: str, y: np.ndarray) -> np.ndarray:
    if name in NUTRIENTS:
        return np.log10(np.clip(y, 0, None) + 0.01)
    if name.startswith("maxn_"):
        return np.log1p(np.clip(y, 0, None))
    return y.astype(float)


def target_family(name: str) -> str:
    if name in NUTRIENTS:
        return name
    m = re.match(r"(nmf_k\d+|pc|present|log1p|maxn|genus|g)_", name)
    return m.group(1) if m else name.rsplit("_", 1)[0]


def _standardise(xtr, xte):
    mu, sd = xtr.mean(0), xtr.std(0)
    sd[sd < 1e-8] = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def _ridge_fit(X, Y, alpha):
    """Closed-form multi-output ridge with intercept on standardised X (dual form when n < d)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    n, d = Xc.shape
    if n < d:
        K = Xc @ Xc.T
        A = np.linalg.solve(K + alpha * np.eye(n), Yc)
        W = Xc.T @ A
    else:
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ Yc)
    return W, ym - xm @ W


ALPHAS = np.logspace(-2, 5, 15)


def ridge_probe(Xtr, Ytr, groups, Xte, n_folds: int = 5):
    """Per-target alpha by grouped CV (R^2 on held-out folds); returns test predictions, alphas, CV R^2."""
    from sklearn.model_selection import GroupKFold

    Xtr, Xte = _standardise(Xtr, Xte)
    ng = len(set(groups))
    folds = list(GroupKFold(n_splits=min(n_folds, ng)).split(Xtr, groups=groups)) if ng >= 2 else []
    score = np.zeros((len(ALPHAS), Ytr.shape[1]))
    for tr, va in folds:
        Xa, Xv = _standardise(Xtr[tr], Xtr[va])
        for i, a in enumerate(ALPHAS):
            W, b = _ridge_fit(Xa, Ytr[tr], a)
            P = Xv @ W + b
            ss = ((Ytr[va] - Ytr[va].mean(0)) ** 2).sum(0)
            score[i] += 1 - ((Ytr[va] - P) ** 2).sum(0) / np.where(ss > 0, ss, np.nan)
    best = np.nanargmax(np.where(np.isfinite(score), score, -np.inf), axis=0) if folds else np.full(Ytr.shape[1], 7)
    pred = np.zeros((Xte.shape[0], Ytr.shape[1]))
    for i in np.unique(best):
        cols = np.where(best == i)[0]
        W, b = _ridge_fit(Xtr, Ytr[:, cols], ALPHAS[i])
        pred[:, cols] = Xte @ W + b
    cv = score[best, np.arange(Ytr.shape[1])] / max(1, len(folds))
    return pred, ALPHAS[best], cv


def _metrics(y, p, binary: bool) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    out = {}
    if binary:
        out["test_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    else:
        ss = ((y - y.mean()) ** 2).sum()
        out["test_r2"] = float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan
        out["test_spearman"] = float(spearmanr(y, p).statistic) if np.std(y) > 0 and np.std(p) > 0 else np.nan
    return out


def probes(cfg, out_dir: Path, windows, paces, log=print) -> None:
    import pandas as pd

    rows = []
    for window in windows:
        emb = {}
        for pace in paces:
            f = out_dir / f"embeddings_{window}_{pace}.npz"
            if f.exists():
                emb[pace] = dict(np.load(f, allow_pickle=False))
        if not emb:
            log(f"probes: no embeddings for window {window}; run --steps embed first")
            continue
        key_index = {k: i for i, k in enumerate(next(iter(emb.values()))["key"])}
        for f in sorted((eval_dir(cfg) / "matched").glob("*.csv.gz")):
            df = pd.read_csv(f, dtype={"sample_id": str})
            ds = df["dataset"].iloc[0]
            df["key"] = df["lat"].round(5).astype(str) + "|" + df["lon"].round(5).astype(str) + "|" + df["time_utc"].astype(str)
            df = df[df["key"].isin(key_index)].reset_index(drop=True)
            ix = df["key"].map(key_index).to_numpy()
            t = pd.to_datetime(df["time_utc"], utc=True)
            doy = 2 * np.pi * t.dt.day_of_year.to_numpy() / 365.25
            meta = np.c_[df["lat"], df["lon"], np.sin(doy), np.cos(doy)]
            fsets: dict[tuple[str, str], np.ndarray] = {}
            for pace, e in emb.items():
                fsets[("meta", pace)] = meta
                fsets[("raw", pace)] = np.arcsinh(e["raw"][ix] / 5e-4)
                for k, v in e.items():
                    if "__" in k:
                        fsets[(k.replace("__", "/"), pace)] = v[ix]
            # raw: a band missing for a sample (sensor range) -> fill with the training mean of that band
            common = np.ones(len(df), bool)
            for (name, pace), X in fsets.items():
                if name == "raw":
                    common &= np.isfinite(X).any(1)
                elif name != "meta":
                    common &= np.isfinite(X).all(1)
            tcols = target_columns(df)
            tr = (df["label_split"] == "train").to_numpy() & common
            te = (df["label_split"] == "test").to_numpy() & common
            if tr.sum() < 20 or te.sum() < 5:
                log(f"probes {ds}/{window}: too few samples with features (train {tr.sum()}, test {te.sum()}); skipped")
                continue
            Y = np.column_stack([transform_target(c, df[c].to_numpy(float)) for c in tcols])
            ok_t = np.isfinite(Y[tr]).all(0) & (np.nanstd(Y[tr], axis=0) > 0)
            tcols = [c for c, o in zip(tcols, ok_t) if o]
            Y = Y[:, ok_t]
            groups = np.array([group_of(ds, s, tt) for s, tt in zip(df["sample_id"], df["time_utc"])])
            pace_ok = (df.get("causal_PACE", pd.Series(0, index=df.index)).fillna(0).to_numpy() >= 1) & te
            for (name, pace), X in fsets.items():
                X = X.astype(float)
                if name == "raw":
                    mu = np.nanmean(X[tr], axis=0)
                    mu = np.where(np.isfinite(mu), mu, 0.0)
                    X = np.where(np.isfinite(X), X, mu)
                yt = Y[tr]
                fin = np.isfinite(yt).all(1)
                pred, alphas, cv = ridge_probe(X[tr][fin], yt[fin], groups[tr][fin], X)
                for j, c in enumerate(tcols):
                    binary = c.startswith("present_")
                    rec = {"dataset": ds, "window": window, "pace": pace, "features": name, "target": c,
                           "family": target_family(c), "n_train": int(fin.sum()), "n_test": int(te.sum()),
                           "alpha": float(alphas[j]), "cv_r2": float(cv[j]),
                           **_metrics(Y[te, j], pred[te, j], binary)}
                    if pace_ok.sum() >= 10:
                        sub = _metrics(Y[pace_ok, j], pred[pace_ok, j], binary)
                        rec.update({f"{k}_pace_usable": v for k, v in sub.items()})
                        rec["n_test_pace_usable"] = int(pace_ok.sum())
                    rows.append(rec)
            log(f"probes {ds}/{window}: {len(tcols)} targets, {len(fsets)} feature sets, train {tr.sum()}, "
                f"test {te.sum()} ({pace_ok.sum()} with PACE)")
    if not rows:
        return
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "probes.csv", index=False)
    val = [c for c in res.columns if c.startswith("test_")]
    summ = res.groupby(["dataset", "family", "window", "pace", "features"], as_index=False)[val].mean()
    summ.to_csv(out_dir / "probes_summary.csv", index=False)
    with_r2 = summ.dropna(subset=["test_r2"]) if "test_r2" in summ else summ.iloc[:0]
    if len(with_r2):
        piv = with_r2.pivot_table(index=["dataset", "family", "window"], columns=["features", "pace"], values="test_r2")
        log("Mean test R^2 per target family (rows) and feature set / PACE setting (columns):")
        log(piv.round(3).to_string())
    log(f"probes -> {out_dir / 'probes.csv'}, {out_dir / 'probes_summary.csv'}")


# ----------------------------------------------------------------------------- PACE retention
def _pace_cells(cfg, split: str, n_anchors: int, cells_per: int, length: int, sensors, seed: int, log=print):
    """Sequences anchored on PACE observations; per anchor up to `cells_per` cells where the anchor
    spectrum is complete. Yields (obs_visible, obs_hidden, cells, Y [cells, P], heritage [cells, P] | None, wl)."""
    seq = TileSequences(cfg, split=split, mode="retro", length=length, anchors="pace", sensors=sensors,
                        selection="recent", include_300m=True, physics=False, seed=seed)
    te = _obs_maker(cfg, with_ridge=True)
    rng = np.random.default_rng(seed)
    if len(seq) == 0:
        log(f"pace {split}: no PACE anchors")
        return
    take = rng.choice(len(seq), min(n_anchors, len(seq)), replace=False)
    for k in take:
        s = seq[int(k)]
        anchor = next((o for o in s["obs"] if o.get("is_anchor")), None)
        if anchor is None:
            continue
        obs, a_x = [], None
        for o in s["obs"]:
            x = te._obs(o, s["anchor_time"], None)
            if o is anchor:
                a_x = x
                her = te._residual(o, x["wl"], x["values"].astype("f4"), x["band_mask"]) if te.ridge is not None else {}
            x.pop("_raw12", None)
            if x["cell_valid"].any():
                obs.append(_as_context(x))
        if a_x is None:
            continue
        P = len(a_x["wl"])
        full = a_x["band_mask"].reshape(P, -1).all(0)
        cells = np.where(full)[0]
        if cells.size == 0:
            continue
        cells = rng.choice(cells, min(cells_per, cells.size), replace=False)
        Y = a_x["values"].astype("f4").reshape(P, -1)[:, cells].T
        H = None
        if her:
            resid = her["pace_resid"].astype("f4")[:, cells].T
            ok = her["pace_resid_mask"][:, cells].T.all(1)
            H = np.where(ok[:, None], Y - resid, np.nan)
        hidden = [o for o in obs if not o["is_pace"]]
        yield obs, hidden, cells, Y, H, a_x["wl"]


@torch.no_grad()
def _cell_e1(models, obs: list[dict], cells: np.ndarray, device) -> dict[str, np.ndarray]:
    out = {}
    if not obs:
        return {info["run"]: np.full((len(cells), m.d), np.nan, np.float32) for m, info in models}
    bt = to_device(collate([{"obs": obs, "mode": "eval"}]), device)
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for m, info in models:
        with torch.autocast(**amp):
            z, gmask, _ = m.encode_context(bt)
            e, valid = m.embed_e1(bt, z, gmask)
        e = e[0, torch.as_tensor(cells, device=e.device)].float()
        v = valid[0, torch.as_tensor(cells, device=e.device)]
        e[~v] = float("nan")
        out[info["run"]] = e.cpu().numpy()
    return out


def pace_retention(cfg, models, out_dir: Path, tcfg: dict, n_train: int = 2000, n_test: int = 800, cells_per: int = 32,
                   log=print) -> None:
    import pandas as pd

    device = next(models[0][0].parameters()).device
    length, sensors = int(tcfg["data"]["length"]), tcfg["data"].get("sensors")
    data = {}
    wl_ref = None
    for split, n in (("train", n_train), ("test_temporal", n_test), ("test_geo", n_test)):
        t0 = time.time()
        acc = {"Y": [], "H": [], "anchor": [], **{(r["run"], v): [] for _, r in models for v in ("visible", "hidden")}}
        for a, (obs, hidden, cells, Y, H, wl) in enumerate(_pace_cells(cfg, split, n, cells_per, length, sensors,
                                                                       seed=20261118 + len(split), log=log)):
            if wl_ref is None:
                wl_ref = wl
            if len(wl) != len(wl_ref) or not np.allclose(wl, wl_ref):
                continue
            ev, eh = _cell_e1(models, obs, cells, device), _cell_e1(models, hidden, cells, device)
            acc["Y"].append(Y)
            acc["H"].append(H if H is not None else np.full_like(Y, np.nan))
            acc["anchor"].append(np.full(len(cells), a))
            for _, r in models:
                acc[(r["run"], "visible")].append(ev[r["run"]])
                acc[(r["run"], "hidden")].append(eh[r["run"]])
        if not acc["Y"]:
            continue
        data[split] = {k: np.concatenate(v) for k, v in acc.items()}
        log(f"pace {split}: {len(acc['Y'])} anchors, {len(data[split]['Y']):,} cells, {time.time() - t0:.0f} s")
    if "train" not in data or len(data) < 2:
        log("pace: need training and at least one test split with PACE; skipped")
        return
    tr = data["train"]
    rows = []

    def score(name, setting, split, Yt, P):
        ok = np.isfinite(P).all(1) & np.isfinite(Yt).all(1)
        if ok.sum() < 10:
            return
        err = Yt[ok] - P[ok]
        ss = ((Yt[ok] - Yt[ok].mean(0)) ** 2).sum(0)
        for j, w in enumerate(wl_ref):
            rows.append({"model": name, "setting": setting, "split": split, "wavelength": float(w), "n_cells": int(ok.sum()),
                         "rmse": float(np.sqrt((err[:, j] ** 2).mean())),
                         "r2": float(1 - (err[:, j] ** 2).sum() / ss[j]) if ss[j] > 0 else np.nan})

    mean_spec = tr["Y"].mean(0)
    for split, d in data.items():
        if split == "train":
            continue
        score("train_mean", "-", split, d["Y"], np.broadcast_to(mean_spec, d["Y"].shape))
        score("heritage_ridge", "-", split, d["Y"], d["H"])
    # decoder: identical ridge for every run and setting; alpha chosen on 20% of training anchors
    rng = np.random.default_rng(0)
    anchors = np.unique(tr["anchor"])
    val_a = rng.choice(anchors, max(1, len(anchors) // 5), replace=False)
    is_val = np.isin(tr["anchor"], val_a)
    for _, info in models:
        for setting in ("visible", "hidden"):
            X = tr[(info["run"], setting)]
            ok = np.isfinite(X).all(1)
            if ok.sum() < 50:
                continue
            Xa, Ya = X[ok & ~is_val], tr["Y"][ok & ~is_val]
            Xv, Yv = X[ok & is_val], tr["Y"][ok & is_val]
            mu, sd = Xa.mean(0), Xa.std(0)
            sd[sd < 1e-8] = 1
            best, best_e = ALPHAS[7], np.inf
            for a in ALPHAS:
                W, b = _ridge_fit((Xa - mu) / sd, Ya, a)
                e = (((Xv - mu) / sd @ W + b - Yv) ** 2).mean()
                if e < best_e:
                    best, best_e = a, e
            Xall, Yall = X[ok], tr["Y"][ok]
            mu, sd = Xall.mean(0), Xall.std(0)
            sd[sd < 1e-8] = 1
            W, b = _ridge_fit((Xall - mu) / sd, Yall, best)
            for split, d in data.items():
                if split == "train":
                    continue
                Xt = d[(info["run"], setting)]
                score(info["run"], setting, split, d["Y"], (Xt - mu) / sd @ W + b)
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "pace_retention.csv", index=False)

    def band(w):
        return "UV <400" if w < 400 else "blue 400-500" if w < 500 else "green 500-600" if w < 600 else "red >=600"

    res["band"] = res["wavelength"].map(band)
    summ = res.groupby(["split", "model", "setting", "band"], as_index=False)[["rmse", "r2"]].mean()
    allb = res.groupby(["split", "model", "setting"], as_index=False)[["rmse", "r2"]].mean().assign(band="all")
    summ = pd.concat([allb, summ], ignore_index=True)
    summ.to_csv(out_dir / "pace_retention_summary.csv", index=False)
    log("PACE spectral retention (mean over wavelengths; RMSE in normalised units, lower is better):")
    log(allb.pivot_table(index=["model", "setting"], columns="split", values=["rmse", "r2"]).round(3).to_string())
    log(f"pace -> {out_dir / 'pace_retention.csv'}, {out_dir / 'pace_retention_summary.csv'}")


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["stage1_A", "stage1_B", "stage1_C"])
    ap.add_argument("--weights", default="final", help="final (falls back to checkpoint) | checkpoint | path")
    ap.add_argument("--tag", default="stage1")
    ap.add_argument("--steps", nargs="+", default=["embed", "probes", "pace"], choices=["embed", "probes", "pace"])
    ap.add_argument("--windows", nargs="+", default=["causal", "retro"])
    ap.add_argument("--pace", nargs="+", default=["visible", "drop"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N evaluation samples only (quick check)")
    ap.add_argument("--pace-anchors", type=int, nargs=2, default=[2000, 800], metavar=("TRAIN", "TEST"))
    args = ap.parse_args(argv)

    cfg = load_config()
    out_dir = cfg.data_root / "runs" / f"eval_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, tcfg = [], None
    if {"embed", "pace"} & set(args.steps):
        for r in args.runs:
            m, info = load_run(cfg, r, args.weights, device)
            models.append((m, info))
            tcfg = tcfg or m.train_config
            print(f"loaded {info['run']}: arm {info['arm']}, step {info['step']:,}, {info['weights']}", flush=True)
        (out_dir / "models.json").write_text(json.dumps([i for _, i in models], indent=1))
    if "embed" in args.steps:
        embed(cfg, models, out_dir, tcfg, args.windows, args.pace, args.batch, args.workers, args.limit)
    if "probes" in args.steps:
        probes(cfg, out_dir, args.windows, args.pace)
    if "pace" in args.steps:
        pace_retention(cfg, models, out_dir, tcfg, *args.pace_anchors)


if __name__ == "__main__":
    main()
