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
                    "chl": 0.3 + 0.1 * np.sin(i / 9) + abs(rng.normal(0, 0.02)), "causal_PACE": 1, "tile_id": t})
    pd.DataFrame(pts).to_csv(ed / "sfer.csv.gz", index=False)
    return cfg


@unittest.skipIf(torch is None, "PyTorch / processing dependencies not installed")
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
                ev.embed(cfg, models, out, tc, ["causal", "retro"], ["visible", "drop"], batch=4, workers=0, log=logs.append)
                e = np.load(out / "embeddings_retro_visible.npz")
                self.assertEqual(e["stage1_A__e1"].shape[1], 32)
                self.assertEqual(e["stage1_B__e3"].shape[1], 64)
                self.assertIn("stage1_B__e2", e)
                self.assertNotIn("stage1_A__e2", e)
                self.assertGreater(np.isfinite(e["stage1_B__e1"]).all(1).mean(), 0.9)
                self.assertGreater((e["n_pace"] > 0).mean(), 0.5)
                dr = np.load(out / "embeddings_retro_drop.npz")
                self.assertEqual(int((dr["n_pace"] > 0).sum()), 0)
                ev.probes(cfg, out, ["causal", "retro"], ["visible", "drop"], log=logs.append)
                p = pd.read_csv(out / "probes.csv")
                self.assertEqual(set(p["target"]), {"po4", "no3", "si", "chl"})
                self.assertTrue({"meta", "raw", "stage1_A/e1", "stage1_B/e2", "stage1_B/e3"} <= set(p["features"]))
                self.assertTrue(p["test_r2"].notna().all())
                # same samples for every feature set within a dataset and window
                self.assertEqual(p.groupby(["window"])["n_test"].nunique().max(), 1)
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
