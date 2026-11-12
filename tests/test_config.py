import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from oefm.config import Region, Window, load_config, merge_windows
from oefm.manifest import Manifest


class RegionTests(unittest.TestCase):
    def test_area_one_degree_at_equator(self):
        r = Region("t", "t", 0, 0, 1, 1)
        self.assertAlmostEqual(r.area_km2(), 12364, delta=15)  # ~111.2 km x 111.2 km

    def test_contains_arrays(self):
        r = Region("t", "t", -82, 24, -80, 26)
        lon = np.array([-81.0, -79.0, np.nan])
        lat = np.array([25.0, 25.0, 25.0])
        self.assertEqual(r.contains(lon, lat).tolist(), [True, False, False])


class WindowTests(unittest.TestCase):
    def test_merge_touching_and_overlapping(self):
        w = merge_windows([
            Window(date(2020, 1, 1), date(2020, 1, 10), ("a",)),
            Window(date(2020, 1, 11), date(2020, 1, 20), ("b",)),   # touches a
            Window(date(2020, 1, 15), date(2020, 2, 1), ("c",)),    # overlaps b
            Window(date(2020, 3, 1), date(2020, 3, 2), ("d",)),
        ])
        self.assertEqual([(x.start, x.end) for x in w],
                         [(date(2020, 1, 1), date(2020, 2, 1)), (date(2020, 3, 1), date(2020, 3, 2))])
        self.assertEqual(w[0].names, ("a", "b", "c"))

    def test_pilot_config(self):
        cfg = load_config(today=date(2026, 11, 11))
        sw = cfg.satellite_windows()
        # The core window runs to today; an eDNA cruise window may extend it back into Dec 2023.
        self.assertLessEqual(sw[-1].start, date(2024, 1, 1))
        self.assertIn("core", sw[-1].names)
        self.assertEqual(sw[-1].end, date(2026, 11, 11))
        # Events from time.event_files are loaded (generated eDNA cruise windows).
        self.assertTrue(any(e["name"].startswith("edna_") for e in cfg.events))
        self.assertTrue(any(n.startswith("edna_") for w in sw for n in w.names))
        # every historical year has at least three windows' worth of days
        for y in range(2018, 2024):
            self.assertGreaterEqual(sum(w.days for w in sw if w.start.year == y), 135)
        pw = cfg.physical_windows()
        self.assertEqual(pw[-1].end, sw[-1].end)
        self.assertLessEqual(pw[0].start, date(2017, 11, 3))
        d = cfg.domain()
        for r in cfg.regions.values():
            self.assertTrue(d.west <= r.west and d.east >= r.east and d.south <= r.south and d.north >= r.north)


class ManifestTests(unittest.TestCase):
    def test_upsert_keeps_download_state_and_merges_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = Manifest(Path(tmp) / "m.sqlite")
            row = {"granule_key": "p:f.nc", "product": "p", "filename": "f.nc", "size_bytes": 10,
                   "regions": "a", "windows": "w1", "local_path": "raw/p/f.nc"}
            self.assertEqual(m.upsert_inventory([row]), (1, 0))
            m.update("p:f.nc", status="verified", sha256="abc")
            self.assertEqual(m.upsert_inventory([{**row, "regions": "b"}]), (0, 1))
            r = m.rows()[0]
            self.assertEqual(r["status"], "verified")
            self.assertEqual(r["sha256"], "abc")
            self.assertEqual(r["regions"], "a,b")
            m.close()


if __name__ == "__main__":
    unittest.main()
