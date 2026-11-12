import math
import unittest

import numpy as np

try:
    from oefm import zenith
except ImportError:  # netCDF4 etc. missing
    zenith = None


class _Var:
    def __init__(self, a):
        self.a = a
        self.variables = {}

    def __getitem__(self, k):
        return self.a[k]


class _FakeSwath:
    """Equator crossing northward: columns step east from a nadir column at lon 0."""

    def __init__(self, ncol=401, step_deg=0.05, tilt=None):
        lons = (np.arange(ncol) - ncol // 2) * step_deg
        self.lon = np.tile(lons, (3, 1)).astype("f4")
        self.lat = np.zeros_like(self.lon)
        nav = {}
        if tilt is not None:
            nav["tilt"] = np.ma.array(np.full(3, tilt, "f4"))
        self.ds = {"navigation_data": type("N", (), {"variables": nav, "__getitem__": lambda s, k: nav[k]})()}


@unittest.skipIf(zenith is None, "oefm dependencies not installed")
class ZenithGeometryTest(unittest.TestCase):
    def closed_form(self, ground_km, h):
        r = zenith.EARTH_KM
        g = ground_km / r
        return math.degrees(g + math.atan2(r * math.sin(g), r + h - r * math.cos(g)))

    def test_nadir_zero_and_known_distance(self):
        sw = _FakeSwath()
        z = zenith.sensor_zenith(sw, "MODIS", slice(None), slice(None))
        self.assertAlmostEqual(float(z[1, 200]), 0.0, places=3)
        d_km = 100 * 0.05 * math.pi / 180 * zenith.EARTH_KM  # 100 columns east of nadir
        self.assertAlmostEqual(float(z[1, 300]), self.closed_form(d_km, 705.0), places=2)
        self.assertTrue(np.all(np.diff(z[1, 200:]) > 0))

    def test_tilt_raises_nadir_angle(self):
        flat = zenith.sensor_zenith(_FakeSwath(), "PACE", slice(None), slice(None))
        tilted = zenith.sensor_zenith(_FakeSwath(tilt=20.0), "PACE", slice(None), slice(None))
        # At the centre column the view zenith is the tilt geometry alone: > 20 deg at the ground.
        self.assertGreater(float(tilted[1, 200]), 20.0)
        self.assertLess(float(tilted[1, 200]), 24.0)
        self.assertTrue(np.all(tilted[1] >= flat[1] - 1e-4))


if __name__ == "__main__":
    unittest.main()
