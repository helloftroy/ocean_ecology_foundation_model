"""Partition and leakage checks for the loader on a synthetic tile index (no data files)."""

import unittest

import numpy as np

try:
    import pandas as pd

    from oefm.config import load_config
    from oefm.grid import TileGrid
    from oefm.loader import TileSequences, apply_masking, geographic_labels, label_split, temporal_labels
except ImportError:
    TileSequences = None


def synthetic_index(grid, days_step=3):
    """Every tile observed by VIIRS every `days_step` days 2023-12-01 .. 2026-10-31 and by PACE
    from 2024-03-05, all fully valid."""
    rows = []
    times = pd.date_range("2023-12-01T18:00Z", "2026-10-31T18:00Z", freq=f"{days_step}D")
    for tid in grid.ids[::9]:
        for t in times:
            rows.append({"product": "viirsn_oc", "family": "VIIRS", "tile_id": tid, "time_utc": t.isoformat(),
                         "valid_fraction": 0.8, "output_path": "x", "row": 0, "granule_key": f"v{t}"})
            if t >= pd.Timestamp("2024-03-05", tz="UTC"):
                rows.append({"product": "pace_oci_aop", "family": "PACE", "tile_id": tid,
                             "time_utc": (t + pd.Timedelta(hours=1)).isoformat(), "valid_fraction": 0.5,
                             "output_path": "x", "row": 0, "granule_key": f"p{t}"})
    return pd.DataFrame(rows)


@unittest.skipIf(TileSequences is None, "processing dependencies not installed")
class TestPartitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.grid = TileGrid(cls.cfg)
        cls.idx = synthetic_index(cls.grid)

    def ds(self, split, **kw):
        return TileSequences(self.cfg, split=split, index=self.idx.copy(), physics=False, **kw)

    def test_temporal_labels(self):
        t = pd.to_datetime(["2025-12-31T23:00Z", "2026-01-01T00:00Z", "2026-03-15T00:00Z", "2026-10-01T23:00Z",
                            "2026-10-02T00:00Z"], utc=True).astype("int64").to_numpy() / 1e9
        self.assertEqual(list(temporal_labels(self.cfg, t)), ["train", "val", "gap", "test", "none"])

    def test_no_leakage_between_splits(self):
        tr, va, te, tg = (self.ds(s, mode=m) for s, m in
                          (("train", "retro"), ("val", "retro"), ("test_temporal", "retro"), ("test_geo", "retro")))
        last_train = tr.obs["t"].max()
        self.assertLess(last_train, pd.Timestamp("2026-01-01", tz="UTC").timestamp())
        self.assertGreaterEqual(va.obs["t"].min(), pd.Timestamp("2026-01-01", tz="UTC").timestamp())
        self.assertGreaterEqual(te.obs["t"].min(), pd.Timestamp("2026-04-01", tz="UTC").timestamp())
        # retro windows of the last train anchors reach past 2026-01-01 but contain no val observation
        a = tr.anchors[-1]
        win = tr._window(tr._by_tile[tr.obs.at[a, "tile_id"]], tr._times[a])
        self.assertTrue((tr._times[win] < pd.Timestamp("2026-01-01", tz="UTC").timestamp()).all())
        self.assertFalse(set(tr.obs["tile_id"]) & set(tg.obs["tile_id"]))
        geo = geographic_labels(self.cfg, self.grid)
        self.assertTrue(all(geo[t] == "train" for t in tr.obs["tile_id"].unique()))
        self.assertTrue(all(geo[t] == "geo_test" for t in tg.obs["tile_id"].unique()))
        # 2024 PACE (unlabeled imagery) is in pretraining
        self.assertGreater(((tr.obs["family"] == "PACE") & (tr.obs["t"] < pd.Timestamp("2025-01-01", tz="UTC").timestamp())).sum(), 0)

    def test_pace_retention_anchors_and_modes(self):
        te = self.ds("test_temporal", anchors="pace")
        self.assertTrue((te.obs.loc[te.anchors, "family"] == "PACE").all())
        dropped = self.ds("train", pace="drop")
        self.assertFalse((dropped.obs["family"] == "PACE").any())
        no_pace_sensors = self.ds("train", sensors=["viirsn_oc"])
        self.assertEqual(set(no_pace_sensors.obs["product"]), {"viirsn_oc"})

    def test_selection_keeps_anchor_and_length(self):
        tr = self.ds("train", length=8, mode="causal")
        a = int(tr.anchors[len(tr) // 2])
        win = tr._window(tr._by_tile[tr.obs.at[a, "tile_id"]], tr._times[a])
        sel = tr._select(win, a)
        self.assertLessEqual(sel.size, 8)
        self.assertIn(a, sel)
        self.assertTrue((tr._times[sel] <= tr._times[a]).all())  # causal: nothing after the anchor

    def test_label_split(self):
        self.assertEqual(label_split(self.cfg, "edna_16S", "2023-12-31T23:00:00Z"), "train")
        self.assertEqual(label_split(self.cfg, "edna_16S", "2024-03-07T12:00:00Z"), "test")
        self.assertEqual(label_split(self.cfg, "sfer", "2024-11-12T12:00:00Z"), "train")
        self.assertEqual(label_split(self.cfg, "sfer", "2025-01-25T12:00:00Z"), "test")

    def test_masking_keeps_visible_observations(self):
        rng = np.random.default_rng(0)
        m = {**self.cfg.raw["training"]["masking"], "drop_time_prob": 1.0, "drop_sensor_prob": 1.0}
        for _ in range(50):
            obs = [{"family": f, "is_pace": f == "PACE", "hidden": False, "valid": np.ones((32, 32), bool),
                    "patch_hidden": np.zeros((32, 32), bool)} for f in ("PACE", "VIIRS", "VIIRS", "MODIS")]
            s = {"obs": obs, "physics": {"hidden": False}}
            apply_masking(s, m, rng)
            self.assertGreaterEqual(sum(not o["hidden"] for o in obs), 2)


if __name__ == "__main__":
    unittest.main()
