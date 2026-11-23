"""MOANA matchups (oefm/moana.py) on tiny synthetic daily files."""

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

try:
    import netCDF4
    import pandas as pd
except ImportError:  # pragma: no cover
    netCDF4 = None


def _write(path: Path, value: float, names=("prococcus_moana", "syncoccus_moana", "picoeuk_moana")):
    lat = np.arange(30.0, 20.0, -0.5)          # north to south, as the mapped files
    lon = np.arange(-85.0, -75.0, 0.5)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f4", ("lat",))[:] = lat
        ds.createVariable("lon", "f4", ("lon",))[:] = lon
        for k, n in enumerate(names):
            v = ds.createVariable(n, "i4", ("lat", "lon"), fill_value=-32767)
            a = np.full((len(lat), len(lon)), int(value * (k + 1)), "i4")
            a[0, 0] = -32767
            v[:] = a


@unittest.skipIf(netCDF4 is None, "netCDF4 not installed")
class TestMoana(unittest.TestCase):
    def test_extract_and_window_means(self):
        from oefm.moana import extract_day, matchups, needed_days

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write(d / "a.nc", 999.0)
            _write(d / "b.nc", 9999.0, names=("prochlorococcus_moana", "synechococcus_moana", "picoeukaryotes_moana"))
            pts = pd.DataFrame({"dataset": ["edna_18S", "edna_18S", "sfer"], "sample_id": ["a", "b", "c"],
                                "lat": [25.0, 25.2, 40.0], "lon": [-80.0, -80.1, -80.0],
                                "time_utc": ["2024-06-02T15:00:00Z"] * 3})
            pts["day"] = pd.to_datetime(pts["time_utc"], utc=True).dt.date
            pts["key"] = pts["sample_id"]
            e = extract_day(d / "a.nc", pts["lat"].to_numpy(), pts["lon"].to_numpy())
            self.assertAlmostEqual(e["pro"][0], 999.0)
            self.assertAlmostEqual(e["syn"][0], 1998.0)
            self.assertTrue(np.isnan(e["pro"][2]))                       # outside the file's domain
            self.assertTrue(np.isfinite(extract_day(d / "b.nc", pts["lat"].to_numpy(), pts["lon"].to_numpy())["pico"][0]))
            self.assertEqual(len(needed_days(pts, 3)), 7)
            # day -1 = 999, day +2 = 9999: +/- 1 d sees one day, +/- 3 d averages both (in log space)
            files = {date(2024, 6, 1): d / "a.nc", date(2024, 6, 4): d / "b.nc"}
            m = matchups(pts, files, 3)
            self.assertAlmostEqual(m.loc[0, "moana_pro_pm1"], np.log10(1000.0))
            self.assertAlmostEqual(m.loc[0, "moana_pro_pm3"], (np.log10(1000.0) + np.log10(10000.0)) / 2)
            self.assertEqual(list(m["moana_ndays_pm3"]), [2, 2, 0])
            self.assertEqual(list(m["moana_ndays_pm1"]), [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
