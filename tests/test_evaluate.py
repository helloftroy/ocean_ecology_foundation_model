"""End-to-end check of oefm.model.evaluate on a small synthetic data root: tile stores, index,
normalisation statistics, matched evaluation sets and two tiny saved models (arms A and B)."""

import os
import tempfile
import unittest

import numpy as np

try:
    import pandas as pd
    import torch
    import zarr  # noqa: F401
except ImportError:
    torch = None


def _flat():
    return {"coef": [0.0], "center": 500.0, "scale": 100.0}


def build_root(root):
    """Synthetic data root: VIIRS every 2 days 2025-01..2025-12 and 2026-04..2026-07, PACE every
    3 days from 2025-03 on, on 8 tiles (training tiles and geographic test tiles)."""
    import json

    import zarr
    from numcodecs import Blosc

    from oefm.config import load_config
    from oefm.grid import TileGrid, write_static
    from oefm.loader import geographic_labels
    from oefm.normstats import stats_dir
    from oefm.targets import eval_dir
    from oefm.tiling import tiles_root

    os.environ["OEFM_DATA_ROOT"] = str(root)
    cfg = load_config()
    grid = TileGrid(cfg)
    troot = tiles_root(cfg)
    write_static(cfg, grid, troot / "grid", log=lambda *a: None)
    geo = geographic_labels(cfg, grid)
    tiles = [t for t in grid.ids if geo.get(t) == "train"][:5] + [t for t in grid.ids if geo.get(t) == "geo_test"][:3]
    rng = np.random.default_rng(0)
    comp = Blosc(cname="zstd", clevel=1)
    prods = {"viirsn_oc": ("VIIRS", np.array([410, 443, 486, 551, 671.0]), 2),
             "pace_oci_aop": ("PACE", np.arange(350.0, 720.0, 10.0), 3)}
    rows = []
    days = list(pd.date_range("2025-01-01T18:00Z", "2025-12-31T23:00Z", freq="1D")) + \
        list(pd.date_range("2026-04-01T18:00Z", "2026-07-31T23:00Z", freq="1D"))
    for prod, (fam, wl, step) in prods.items():
        for di, day in enumerate(days):
            if di % step or (prod == "pace_oci_aop" and day < pd.Timestamp("2025-03-05", tz="UTC")):
                continue
            n = len(tiles)
            shape = (n, len(wl), 32, 32)
            base = 0.006 * np.exp(-((wl - 450) / 120) ** 2)[None, :, None, None] * (1 + 0.3 * np.sin(di / 20))
            refl = (base * np.exp(rng.normal(0, 0.1, (n, 1, 32, 32)))).astype("f4")
            wv = (rng.random((n, 32, 32)) > 0.2).astype("f4")
            refl[np.broadcast_to(wv[:, None] == 0, shape)] = np.nan
            path = troot / prod / "2025" / f"{prod}_{di}.zarr.zip"
            path.parent.mkdir(parents=True, exist_ok=True)
            store = zarr.ZipStore(str(path), mode="w")
            g = zarr.group(store=store)
            for name, a in {"reflectance": refl, "w_valid": wv, "w_strict": wv, "senz": np.full((n, 32, 32), 20, "f4"),
                            "flags_or": np.zeros((n, 32, 32), "u4"), "wavelength": wl.astype("f4")}.items():
                g.create_dataset(name, data=a, chunks=(1,) + a.shape[1:] if a.ndim > 1 else a.shape, compressor=comp)
            g.attrs.update({"flag_meanings": ["LAND", "HISATZEN"], "flag_masks": [1, 2], "tile_ids": tiles})
            store.close()
            for j, t in enumerate(tiles):
                rows.append({"product": prod, "family": fam, "granule_key": f"{prod}{di}", "filename": "x",
                             "output_path": str(path.relative_to(root)), "row": j, "tile_id": t,
                             "time_utc": (day + pd.Timedelta(hours=1 if fam == "PACE" else 0)).isoformat(),
                             "valid_fraction": float((wv[j] > 0).mean())})
    (troot / "index").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(troot / "index" / "tile_observations.parquet")
    sd = stats_dir(cfg)
    sd.mkdir(parents=True, exist_ok=True)
    view = {"s": {"coef": [float(np.log(1e-3))], "center": 500.0, "scale": 100.0, "log": True},
            "mean": {**_flat(), "coef": [1.0]}, "std": {**_flat(), "coef": [1.0]}}
    (sd / "normstats.json").write_text(json.dumps({"rrs": view, "surface": view}))
    # evaluation points at tile centres: SFER-like, train 2025-01..06, test 2025-07..12 (labels only)
    ed = eval_dir(cfg) / "matched"
    ed.mkdir(parents=True, exist_ok=True)
    centres = grid.table().set_index("tile_id")
    pts = []
    for i in range(120):
        t = tiles[i % 5]
        lat, lon = float(centres.loc[t, "lat_centre"]), float(centres.loc[t, "lon_centre"])
        day = pd.Timestamp("2025-03-20T15:00Z") + pd.Timedelta(days=int(i * 2.2))
        pts.append({"sample_id": f"WS25{100 + i // 10:03d}_{i}", "dataset": "sfer", "lat": lat, "lon": lon,
                    "time_utc": day.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "label_split": "train" if day < pd.Timestamp("2025-08-01", tz="UTC") else "test",
                    "po4": abs(rng.normal(0.05, 0.02)), "no3": abs(rng.normal(0.1, 0.05)), "si": 2.0 + np.sin(i / 7),
                    "chl": np.nan if i % 7 == 3 else 0.3 + 0.1 * np.sin(i / 9) + abs(rng.normal(0, 0.02)),
                    "causal_PACE": 1, "tile_id": t})  # Chl missing for some samples, as in the real SFER table
    pd.DataFrame(pts).to_csv(ed / "sfer.csv.gz", index=False)
    # eDNA-like set on the same points: sequencing platform switches mid-training (as Illumina ->
    # Element in the real data; every test sample on Element), with a platform offset in the target
    ed_rows = []
    for i, p in enumerate(pts):
        elem = float(i >= 40)
        ed_rows.append({**{k: p[k] for k in ("lat", "lon", "time_utc", "label_split", "causal_PACE", "tile_id")},
                        "sample_id": p["sample_id"], "dataset": "edna_16S",
                        "seq_platform": "ELEMENT" if elem else "ILLUMINA", "seq_runs": f"run{int(elem)}",
                        "seq_element_frac": elem, "seq_n_replicates": 3.0,
                        "seq_log10_input_reads": 5.5 + rng.normal(0, 0.1) if elem else 4.1 + rng.normal(0, 0.1),
                        "seq_log10_table_reads": 5.0, "nmf_k5_1": 0.5 * elem + 0.1 * np.sin(i / 9) + rng.normal(0, 0.02),
                        "pc_1": np.cos(i / 11) + rng.normal(0, 0.05),
                        # nearly absent in the test period (like 18S NMF component 10 in 2024)
                        "pc_2": (np.cos(i / 5) if p["label_split"] == "train" else 0.001 * np.cos(i / 5)),
                        "phyto_synechococcus": np.sin(i / 6) + rng.normal(0, 0.1),
                        "phyto_chloroplast": np.cos(i / 8) + rng.normal(0, 0.1)})
    edf = pd.DataFrame(ed_rows)
    edf.to_csv(ed / "edna_16S.csv.gz", index=False)
    # MOANA matchups (oefm/moana.py) for every eDNA sample, keyed like the embeddings
    key = edf["lat"].round(5).astype(str) + "|" + edf["lon"].round(5).astype(str) + "|" + edf["time_utc"].astype(str)
    mo = edf[["dataset", "sample_id", "lat", "lon", "time_utc"]].assign(key=key)
    for w in (1, 3):
        for g, col in (("pro", "phyto_chloroplast"), ("syn", "phyto_synechococcus"), ("pico", "phyto_chloroplast")):
            mo[f"moana_{g}_pm{w}"] = 4 + 0.3 * edf[col] + rng.normal(0, 0.2, len(edf))
        mo[f"moana_ndays_pm{w}"] = 2
    mo.to_csv(eval_dir(cfg) / "moana_matchups.csv.gz", index=False)
    # CRCP-like reef fish: survey units on 15 sites, habitat covariates, counts tied to depth
    fr = []
    for i in range(150):
        t = tiles[i % 5]
        day = pd.Timestamp("2025-03-25T14:00Z") + pd.Timedelta(days=int(i * 1.7))
        depth = 0.6 + 0.8 * ((i * 7) % 15) / 15
        row = {"sample_id": f"S{i % 15:02d}_{day:%Y-%m-%d}_{i}", "dataset": "fish_crcp",
               "lat": float(centres.loc[t, "lat_centre"]), "lon": float(centres.loc[t, "lon_centre"]),
               "time_utc": day.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "label_split": "train" if day < pd.Timestamp("2025-08-15", tz="UTC") else "test",
               "causal_PACE": 1, "tile_id": t, "hab_log10_depth_m": depth, "hab_type_1": float(i % 3 == 0)}
        for k, (a, b) in enumerate(((1.5, 0.8), (2.0, -2.5))):
            cnt = rng.poisson(np.exp(a + b * depth))
            row[f"present_sp{k}"], row[f"log1p_sp{k}"] = int(cnt > 0), float(np.log1p(cnt))
        fr.append(row)
    pd.DataFrame(fr).to_csv(ed / "fish_crcp.csv.gz", index=False)
    return cfg


@unittest.skipIf(torch is None, "PyTorch / processing dependencies not installed")
class TestRidge(unittest.TestCase):
    def test_unpenalised_covariates(self):
        from oefm.model.evaluate import _ridge_fit, _ridge_predict

        rng = np.random.default_rng(1)
        n = 200
        X, C = rng.normal(size=(n, 30)), rng.normal(size=(n, 2))
        Y = (C @ np.array([2.0, -1.0]) + 0.3 * X[:, 0] + 1.5)[:, None]
        # heavy penalty: the X part shrinks to ~0, the covariate part is still fitted exactly
        W, G = _ridge_fit(X, Y, 1e8, C)
        self.assertLess(np.abs(W).max(), 1e-4)
        np.testing.assert_allclose(G[:, 0], [1.5, 2.0, -1.0], atol=0.1)
        # no penalty: identical to ordinary least squares on [1, C, X]
        W0, G0 = _ridge_fit(X, Y, 1e-10, C)
        D = np.c_[np.ones(n), C, X]
        np.testing.assert_allclose(_ridge_predict(X, C, W0, G0), D @ np.linalg.lstsq(D, Y, rcond=None)[0], atol=1e-6)
        # zero feature columns = covariates only; dual form (n < d) agrees with the primal form
        Wz, Gz = _ridge_fit(np.zeros((n, 0)), Y, 1.0, C)
        self.assertEqual(Wz.shape, (0, 1))
        Xw = rng.normal(size=(20, 50))
        Wd, Gd = _ridge_fit(Xw, Y[:20], 3.0, C[:20])
        R = np.c_[np.ones(20), C[:20]]
        Xr = Xw - R @ np.linalg.lstsq(R, Xw, rcond=None)[0]
        Yr = Y[:20] - R @ np.linalg.lstsq(R, Y[:20], rcond=None)[0]
        np.testing.assert_allclose(Wd, np.linalg.solve(Xr.T @ Xr + 3.0 * np.eye(50), Xr.T @ Yr), atol=1e-8)


@unittest.skipIf(torch is None, "PyTorch / processing dependencies not installed")
class TestPoisson(unittest.TestCase):
    def test_matches_sklearn_and_unpenalised_covariates(self):
        from sklearn.linear_model import PoissonRegressor

        from oefm.model.focused import _d2, poisson_fit, poisson_predict

        rng = np.random.default_rng(2)
        n = 400
        X, C = rng.normal(size=(n, 5)), rng.normal(size=(n, 1))
        y = rng.poisson(np.exp(0.5 + 0.4 * X[:, 0] + 0.8 * C[:, 0])).astype(float)
        # no covariates: sum-form penalty alpha = n x sklearn's mean-form alpha
        b = poisson_fit(X, np.zeros((n, 0)), y, alpha=n * 0.01)
        sk = PoissonRegressor(alpha=0.01, max_iter=1000, tol=1e-10).fit(X, y)
        np.testing.assert_allclose(b[1:], sk.coef_, atol=1e-4)
        self.assertAlmostEqual(b[0], sk.intercept_, places=4)
        # heavy penalty: X shrinks to ~0, the covariate keeps its full effect
        b = poisson_fit(X, C, y, alpha=1e8)
        self.assertLess(np.abs(b[2:]).max(), 1e-4)
        self.assertAlmostEqual(b[1], 0.8, delta=0.1)
        mu = poisson_predict(X, C, poisson_fit(X, C, y, 1.0))
        d2, d2r = _d2(y, mu)
        self.assertGreater(d2[0], 0.3)
        # a pure level shift: D^2 drops, the level-free D^2 does not
        d2s, d2rs = _d2(y, mu * 3)
        self.assertLess(d2s[0], d2[0] - 0.2)
        self.assertAlmostEqual(d2rs[0], d2r[0], places=6)


@unittest.skipIf(torch is None, "PyTorch / processing dependencies not installed")
class TestBootstrap(unittest.TestCase):
    def test_vectorised_r2_and_auc(self):
        from sklearn.metrics import r2_score, roc_auc_score

        from oefm.model.robust import _boot_auc, _boot_r2, _strata_resamples

        rng = np.random.default_rng(0)
        g = np.repeat(["a", "b", "c"], 20)
        I = _strata_resamples(g, 30, seed=1)
        self.assertTrue(all((g[I[k]] == g).all() for k in range(30)))      # resampled within groups
        y, p = rng.normal(size=60), rng.normal(size=60)
        y[5] = np.nan
        r = _boot_r2(y, p, I)
        k = 3
        ok = np.isfinite(y[I[k]])
        self.assertAlmostEqual(r[k], r2_score(y[I[k]][ok], p[I[k]][ok]))
        yb = (rng.random(60) > 0.5).astype(float)
        a = _boot_auc(yb, p, I)
        self.assertAlmostEqual(a[k], roc_auc_score(yb[I[k]], p[I[k]]))


class TestEvaluate(unittest.TestCase):
    def test_end_to_end(self):
        from test_model import TINY

        from oefm.model import evaluate as ev
        from oefm.model.net import Model

        old = os.environ.get("OEFM_DATA_ROOT")
        with tempfile.TemporaryDirectory() as d:
            try:
                cfg = build_root(__import__("pathlib").Path(d))
                cfgs = {}
                for arm in ("A", "B"):
                    torch.manual_seed(0)
                    c = {**TINY, "arm": arm, "data": {"length": 8, "sensors": ["viirsn_oc", "pace_oci_aop"]}}
                    run = cfg.data_root / "runs" / f"stage1_{arm}"
                    run.mkdir(parents=True)
                    torch.save({"model": Model(c).state_dict(), "step": 5, "config": c}, run / "weights_final.pt")
                    cfgs[arm] = c
                logs = []
                out = cfg.data_root / "runs" / "eval_t"
                out.mkdir(parents=True)
                models = [ev.load_run(cfg, r, "final", torch.device("cpu")) for r in ("stage1_A", "stage1_B")]
                tc = models[0][0].train_config
                ev.embed(cfg, models, out, tc, ["causal", "retro", "retro3", "seasonal"], ["visible", "drop"], batch=4,
                         workers=0, log=logs.append)
                e = np.load(out / "embeddings_retro_visible.npz")
                self.assertEqual(e["stage1_A__e1"].shape[1], 32)
                self.assertEqual(e["stage1_B__e3"].shape[1], 64)
                self.assertIn("stage1_B__e2", e)
                self.assertNotIn("stage1_A__e2", e)
                self.assertGreater(np.isfinite(e["stage1_B__e1"]).all(1).mean(), 0.9)
                self.assertGreater((e["n_pace"] > 0).mean(), 0.5)
                dr = np.load(out / "embeddings_retro_drop.npz")
                self.assertEqual(int((dr["n_pace"] > 0).sum()), 0)
                # window spans: +/- 3 d stays within 3 d; seasonal covers fish only and reaches back > 30 d
                r3 = np.load(out / "embeddings_retro3_visible.npz")
                self.assertLessEqual(np.nanmax(r3["span_days"]), 3.0 + 1e-6)
                self.assertGreater(np.nanmax(e["span_days"]), 3.0)
                se = np.load(out / "embeddings_seasonal_visible.npz")
                self.assertEqual(len(se["key"]), 150)
                self.assertGreater(np.nanmax(se["span_days"]), 30)
                self.assertGreater(np.isfinite(se["stage1_A__e1"]).all(1).mean(), 0.9)
                ev.probes(cfg, out, ["causal", "retro"], ["visible", "drop"], log=logs.append)
                p = pd.read_csv(out / "probes.csv")
                self.assertEqual(set(p.loc[p["dataset"] == "sfer", "target"]), {"po4", "no3", "si", "chl"})
                ed = p[p["dataset"] == "edna_16S"]
                # seq_* are covariates, not targets
                self.assertEqual(set(ed["target"]), {"nmf_k5_1", "pc_1", "pc_2", "phyto_synechococcus", "phyto_chloroplast"})
                self.assertFalse(p["target"].str.startswith("hab_").any())        # habitat = covariates
                chl = p[(p["dataset"] == "sfer") & (p["target"] == "chl")]
                self.assertTrue((chl["n_train"] < p.loc[(p["dataset"] == "sfer") & (p["target"] == "po4"), "n_train"].max()).all())
                self.assertTrue((chl["n_test_target"] < chl["n_test"]).all())
                self.assertTrue((ed.loc[ed["target"] == "pc_2", "test_sd_ratio"] < 0.25).all())
                sm = pd.read_csv(out / "probes_summary.csv")
                self.assertIn("test_r2_median", sm)
                self.assertTrue((sm.loc[(sm["dataset"] == "edna_16S") & (sm["family"] == "pc"), "n_targets"] == 1).all())
                self.assertEqual(set(ed["covariates"]), {"seq", "none"})
                self.assertEqual(set(ed.loc[ed["features"] == "seq", "covariates"]), {"seq"})
                self.assertEqual(set(p.loc[p["dataset"] == "sfer", "covariates"]), {"none"})
                self.assertTrue({"meta", "raw", "stage1_A/e1", "stage1_B/e2", "stage1_B/e3"} <= set(p["features"]))
                cont = ~p["target"].str.startswith("present_")
                self.assertTrue(p.loc[cont, "test_r2"].notna().all())
                self.assertTrue(p.loc[~cont, "test_auc"].notna().all())
                # same samples for every feature set within a dataset and window
                self.assertEqual(p.groupby(["dataset", "window"])["n_test"].nunique().max(), 1)
                # external features (e.g. Granite): eDNA only, half the samples -> eDNA scored on that half,
                # SFER untouched; embeddings read from the first evaluation (--embeddings-from)
                ek = np.load(out / "embeddings_causal_visible.npz")["key"]
                m16 = pd.read_csv(sorted((ev.eval_dir(cfg) / "matched").glob("*16S*.csv.gz"))[0], dtype={"sample_id": str})
                k16 = (m16["lat"].round(5).astype(str) + "|" + m16["lon"].round(5).astype(str) + "|"
                       + m16["time_utc"].astype(str)).to_numpy()
                k16 = np.array([k for k in k16 if k in set(ek)])
                Xe = np.random.default_rng(0).normal(size=(len(k16), 6)).astype("f4")
                Xe[::2] = np.nan
                np.savez(out / "ext.npz", key=k16, datasets=np.array(["edna_16S"]), **{"ext/strict": Xe})
                out2 = cfg.data_root / "runs" / "eval_t2"
                out2.mkdir()
                try:
                    ev.EXTRA_FEATURES[:] = [out / "ext.npz"]
                    ev.EMB_DIR = out
                    ev.probes(cfg, out2, ["causal"], ["visible"], log=logs.append)
                finally:
                    ev.EXTRA_FEATURES[:] = []
                    ev.EMB_DIR = None
                p2 = pd.read_csv(out2 / "probes.csv")
                e2 = p2[p2["dataset"] == "edna_16S"]
                self.assertIn("ext/strict", set(e2["features"]))
                self.assertEqual(e2["n_test"].nunique(), 1)
                n1 = p[(p["dataset"] == "edna_16S") & (p["window"] == "causal")]["n_test"].iloc[0]
                self.assertLess(e2["n_test"].iloc[0], n1)
                self.assertNotIn("ext/strict", set(p2.loc[p2["dataset"] == "sfer", "features"]))
                # robustness (test periods = quarters in this one-year synthetic set)
                from oefm.model import robust as rb

                rb.robustness(cfg, out, ["causal", "retro"], period="quarter", min_period=10, n_boot=50, log=logs.append)
                rot = pd.read_csv(out / "robust_rotate.csv")
                self.assertGreaterEqual(rot.groupby(["dataset", "window"])["test_period"].nunique().min(), 3)
                self.assertGreaterEqual(pd.read_csv(out / "robust_rotate_summary.csv")["n_periods"].max(), 3)
                bs = pd.read_csv(out / "robust_bootstrap.csv")
                sm_v = sm[(sm["pace"] == "visible") & ((sm["covariates"] == "seq") | (sm["dataset"] == "sfer"))]
                mm = bs[bs["metric"] == "r2_median"].merge(sm_v, on=["dataset", "window", "family", "features"])
                self.assertGreater(len(mm), 10)
                np.testing.assert_allclose(mm["value"], mm["test_r2_median"])   # official split = probes
                self.assertTrue((bs["ci_lo"] <= bs["ci_hi"]).all())
                self.assertTrue((bs.loc[bs["features"] == "stage1_A/e1", "delta"] == 0).all())
                era = pd.read_csv(out / "robust_pace_era.csv")
                self.assertTrue({"visible", "drop", "delta", "delta_lo"} <= set(era.columns))
                self.assertFalse({"meta", "seq"} & set(era["features"]))
                ret = pd.read_csv(out / "robust_retention.csv")
                self.assertTrue({"log_brightness", "log_blue_green_oc3m", "band_443"} <= set(ret["quantity"]))
                rb.amplitude(cfg, models, out, log=logs.append)
                amp = pd.read_csv(out / "amplitude_check.csv")
                self.assertEqual(set(amp["level"]), {"token", "e1"})
                self.assertTrue(amp.loc[amp["metric"] == "log_amplitude_r2", "pass"].astype(bool).all())
                # focused: PACE-era phytoplankton (with MOANA) and fish over habitat
                from oefm.model import focused as fo

                fo.phyto(cfg, out, ["retro", "retro3"], n_boot=50, log=logs.append)
                ph = pd.read_csv(out / "focused_phyto_summary.csv")
                self.assertTrue({"moana", "seq", "stage1_A/e1", "stage1_B/e1"} <= set(ph["features"]))
                self.assertEqual(set(ph["window"]), {"retro", "retro3"})
                self.assertEqual(ph.groupby("window")["n"].nunique().max(), 1)     # same samples per window
                self.assertTrue({"delta_vs_drop", "delta_vs_moana"} <= set(ph.columns))
                e1v = ph[ph["features"].str.endswith("/e1") & (ph["pace"] == "visible")]
                self.assertTrue(e1v[["delta_vs_drop", "delta_vs_moana"]].notna().all().all())
                self.assertGreater(ph.loc[ph["features"] == "moana", "r2_median"].min(), 0.2)
                fo.fish(cfg, out, ["causal", "seasonal"], n_boot=50, period="quarter", min_period=10, log=logs.append)
                fs = pd.read_csv(out / "focused_fish_summary.csv")
                off = fs[fs["split"] == "official"]
                self.assertTrue({"habitat", "meta", "raw", "stage1_A/e1"} <= set(off["features"]))
                self.assertGreater(off.loc[off["features"] == "habitat", "d2"].min(), 0.1)  # counts follow depth
                self.assertTrue((off.loc[off["features"] == "habitat", "delta_d2_vs_habitat"] == 0).all())
                self.assertGreater((fs["split"] != "official").sum(), 0)
                ev.pace_retention(cfg, models, out, tc, n_train=30, n_test=10, cells_per=8, log=logs.append)
                r = pd.read_csv(out / "pace_retention_summary.csv")
                self.assertTrue({"train_mean", "stage1_A", "stage1_B"} <= set(r["model"]))
                self.assertTrue({"test_temporal", "test_geo"} <= set(r["split"]))
                self.assertTrue(np.isfinite(r["rmse"]).all())
            finally:
                if old is None:
                    os.environ.pop("OEFM_DATA_ROOT", None)
                else:
                    os.environ["OEFM_DATA_ROOT"] = old


if __name__ == "__main__":
    unittest.main()
