"""Tile grid, regridding and QC-policy checks on synthetic swaths (no data files needed)."""

import math
import unittest

import numpy as np

try:
    from oefm import tiling
    from oefm.config import load_config
    from oefm.grid import Laea, TileGrid, tile_id
    from oefm.qcpolicy import policy, red_angle_weight
except ImportError:  # netCDF4 / zarr missing
    tiling = None


@unittest.skipIf(tiling is None, "processing dependencies not installed")
class TestGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.grid = TileGrid(cls.cfg)

    def test_projection_round_trip_and_equal_area(self):
        p = Laea(26.05, -85.4)
        lon = np.array([-92.0, -85.4, -78.8, -80.0])
        lat = np.array([22.8, 26.05, 29.3, 24.5])
        x, y = p.forward(lon, lat)
        lo, la = p.inverse(x, y)
        np.testing.assert_allclose(lo, lon, atol=1e-9)
        np.testing.assert_allclose(la, lat, atol=1e-9)
        # a 1 x 1 degree box: projected area equals the spherical area
        t = np.linspace(0, 1, 400)
        bl = np.r_[-90 + t, np.full(400, -89.0), -89 - t, np.full(400, -90.0)]
        bb = np.r_[np.full(400, 25.0), 25 + t, np.full(400, 26.0), 26 - t]
        bx, by = p.forward(bl, bb)
        area = 0.5 * abs(np.dot(bx, np.roll(by, 1)) - np.dot(by, np.roll(bx, 1)))
        sph = p.r ** 2 * math.radians(1) * (math.sin(math.radians(26)) - math.sin(math.radians(25)))
        self.assertAlmostEqual(area / sph, 1.0, places=4)

    def test_tile_ids_fixed(self):
        g = self.grid
        self.assertEqual(g.tile_m, 38400.0)
        self.assertEqual(g.olci_tc, 128)
        self.assertEqual(tile_id(5, -7), "X+05Y-07")
        self.assertEqual(len(set(g.ids)), g.n_tiles)
        # every pilot region is covered: its centre falls in a selected tile
        for r in self.cfg.regions.values():
            x, y = g.proj.forward((r.west + r.east) / 2, (r.south + r.north) / 2)
            self.assertGreaterEqual(int(g.slots(np.array([x]), np.array([y]))[0]), 0)

    def test_slots_cell_order(self):
        g = self.grid
        k = 0
        ix, iy = int(g.ix[k]), int(g.iy[k])
        # centre of cell (row 2, col 5) of tile k
        x = np.array([(ix * g.tc + 5 + 0.5) * g.cell_m])
        y = np.array([(iy * g.tc + 2 + 0.5) * g.cell_m])
        self.assertEqual(int(g.slots(x, y)[0]), k * g.tc * g.tc + 2 * g.tc + 5)
        self.assertEqual(int(g.slots(np.array([1e8]), np.array([0.0]))[0]), -1)


@unittest.skipIf(tiling is None, "processing dependencies not installed")
class TestRegridding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()
        cls.grid = TileGrid(cls.cfg)

    def _swath(self, spacing_scan, spacing_track, nr=120, nc=160, angle_deg=12.0):
        """A rotated regular swath of pixel centres around the centre of tile 0."""
        g = self.grid
        cx, cy = (g.ix[40] + 0.5) * g.tile_m, (g.iy[40] + 0.5) * g.tile_m
        a = math.radians(angle_deg)
        c, r = np.meshgrid(np.arange(nc) - nc / 2, np.arange(nr) - nr / 2)
        x = cx + c * spacing_scan * math.cos(a) - r * spacing_track * math.sin(a)
        y = cy + c * spacing_scan * math.sin(a) + r * spacing_track * math.cos(a)
        return x, y

    def _regrid(self, x, y, vals, cell_m=None):
        g = self.grid
        cell_m = cell_m or g.cell_m
        trx, try_ = tiling._track_vectors(x, y)
        pix, slot, w = tiling.footprint_pairs(g, x, y, trx, try_, cell_m)
        n = int(round(g.tile_m / cell_m))
        acc = tiling._Acc(g.n_tiles, n, vals.shape[1], aux=True)
        npx = x.size
        valid = np.ones(npx, bool)
        z = np.zeros(npx)
        acc.add(pix, slot, w, valid, valid.astype("f8"), np.zeros(npx, "u4"), vals, z, z)
        return acc, w

    def test_large_pixels_fill_cells_and_conserve_weight(self):
        # 4 km x 1.2 km pixels (PACE swath edge): every cell under the swath interior gets data
        x, y = self._swath(4000.0, 1200.0, nr=60, nc=40)
        vals = np.full((x.size, 2), 0.01, "f4")
        acc, w = self._regrid(x, y, vals)
        self.assertAlmostEqual(float(w.sum()), x.size, delta=1e-3 * x.size)  # each pixel carries weight 1
        refl = np.where(acc.wb[0] > 0, acc.sum[0] / np.maximum(acc.wb[0], 1e-12), np.nan)
        self.assertTrue(np.allclose(refl[np.isfinite(refl)], 0.01, rtol=1e-5))
        # no holes: interior cells (well inside the swath footprint) all have data
        g = self.grid
        cx, cy = (g.ix[40] + 0.5) * g.tile_m, (g.iy[40] + 0.5) * g.tile_m
        xs, ys = np.meshgrid(cx + np.arange(-12, 12) * 1200.0, cy + np.arange(-12, 12) * 1200.0)
        sl = g.slots(xs.ravel(), ys.ravel())
        self.assertTrue((acc.w_valid[sl] > 0).all())

    def test_linear_field_preserved(self):
        x, y = self._swath(750.0, 750.0)
        f = (1e-6 * (x - x.mean()) + 0.005).ravel()
        acc, _ = self._regrid(x, y, f[:, None].astype("f4"))
        ok = acc.wb[0] > 0.9  # fully covered cells
        refl = acc.sum[0][ok] / acc.wb[0][ok]
        g = self.grid
        slots = np.where(ok)[0]
        t, rem = np.divmod(slots, g.tc * g.tc)
        j, i = np.divmod(rem, g.tc)
        xc = (g.ix[t] * g.tc + i + 0.5) * g.cell_m
        expect = 1e-6 * (xc - x.mean()) + 0.005
        # the field changes by 1.2e-3 across one cell; footprint sub-sampling may shift a cell's
        # mean by a few percent of that, with no bias
        err = refl - expect
        self.assertLess(float(np.median(np.abs(err))), 0.05 * 1.2e-3)
        self.assertLess(abs(float(np.mean(err))), 0.005 * 1.2e-3)

    def test_dequantise(self):
        true = -80.0 + 0.0027 * np.arange(60)
        stepped = np.repeat(true[::3], 3)[None, :]  # each value repeated over 3 columns
        out = tiling.dequantise(stepped)[0]
        self.assertLess(float(np.abs(np.diff(out)).min()), 0.0027 * 1.01)
        self.assertGreater(float(np.abs(np.diff(out)).min()), 0.0027 * 0.99)
        plain = true[None, :]
        np.testing.assert_array_equal(tiling.dequantise(plain), plain)


@unittest.skipIf(tiling is None, "processing dependencies not installed")
class TestPolicy(unittest.TestCase):
    def test_masks_per_sensor(self):
        cfg = load_config()
        self.assertNotIn("HISATZEN", policy(cfg, "pace_oci_aop").training_flags)
        self.assertNotIn("HISATZEN", policy(cfg, "viirsn_oc").training_flags)
        self.assertIn("HISATZEN", policy(cfg, "modisa_oc").training_flags)
        self.assertIn("HISATZEN", policy(cfg, "viirsn_oc").strict_flags)
        self.assertEqual(policy(cfg, "pace_oci_sfrefl").training_mask, "sfrefl")
        for p in cfg.satellite_products:
            fl = policy(cfg, p).training_flags
            for hard in ("ATMFAIL", "NAVFAIL", "HILT", "LAND"):
                self.assertIn(hard, fl, f"{p} must reject {hard}")

    def test_red_angle_weight(self):
        cfg = load_config()
        w = red_angle_weight(cfg, "viirsn_oc", np.array([443, 671, 671, 671]), np.array([65, 10, 50, 65]))
        self.assertEqual(float(w[0]), 1.0)           # blue: always 1
        self.assertEqual(float(w[1]), 1.0)           # red at low angle: within tolerance
        self.assertAlmostEqual(float(w[2]), 0.10 / (1 - 0.782), places=4)
        self.assertAlmostEqual(float(w[3]), 0.10 / (1 - 0.636), places=4)
        self.assertTrue((red_angle_weight(cfg, "pace_oci_sfrefl", np.array([865]), np.array([70])) == 1).all())


if __name__ == "__main__":
    unittest.main()
