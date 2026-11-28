"""processing.exclude_windows: imagery skipped in training, evaluation embeddings and matchup counts."""

import unittest

import numpy as np
import pandas as pd

from oefm.config import load_config
from oefm.exclusions import drop_excluded, excluded, tile_centres, windows


def ts(s):
    return pd.Timestamp(s).timestamp()


class TestExclusions(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_eclipse_window_in_config(self):
        w = windows(self.cfg)
        self.assertTrue(any(x["t0"] == ts("2024-04-08T15:30:00Z") and x["t1"] == ts("2024-04-08T21:30:00Z") for x in w))

    def test_time_and_box(self):
        t = np.array([ts("2024-04-08T15:00:00Z"), ts("2024-04-08T18:13:26Z"), ts("2024-04-08T21:29:00Z"),
                      ts("2024-04-08T22:00:00Z"), ts("2024-04-08T18:13:26Z")])
        lon = np.array([-82.0, -82.0, -82.0, -82.0, 10.0])      # last: Europe/Africa, outside the box
        lat = np.array([26.0, 26.0, 26.0, 26.0, 5.0])
        self.assertEqual(excluded(self.cfg, t, lon, lat).tolist(), [False, True, True, False, False])
        # unknown location: time only (conservative)
        self.assertEqual(excluded(self.cfg, t).tolist(), [False, True, True, False, True])

    def test_drop_excluded_on_tile_frame(self):
        from oefm.grid import TileGrid

        g = TileGrid(self.cfg)
        cen = tile_centres(g)
        tid = g.ids[0]
        lon, lat = cen[tid]
        self.assertTrue(-120 <= lon <= -50 and 0 <= lat <= 60)
        df = pd.DataFrame({"tile_id": [tid, tid, "unknown_tile"],
                           "t": [ts("2024-04-08T18:00:00Z"), ts("2024-04-09T18:00:00Z"), ts("2024-04-08T18:00:00Z")]})
        out = drop_excluded(self.cfg, df, [g])
        self.assertEqual(out["t"].tolist(), [ts("2024-04-09T18:00:00Z")])


if __name__ == "__main__":
    unittest.main()
