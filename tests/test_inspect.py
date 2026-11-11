"""Inspection, index and matchup code on synthetic files laid out like OB.DAAC L2 and L4 grids."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

try:
    import netCDF4
except ImportError:  # pragma: no cover
    netCDF4 = None

from oefm.config import load_config

FLAGS = ("ATMFAIL LAND PRODWARN HIGLINT HILT HISATZEN COASTZ SPARE STRAYLIGHT CLDICE COCCOLITH TURBIDW "
         "HISOLZEN SPARE LOWLW CHLFAIL NAVWARN ABSAER SPARE MAXAERITER MODGLINT CHLWARN ATMWARN SPARE "
         "SEAICE NAVFAIL FILTER SPARE BOWTIEDEL HIPOL PRODFAIL SPARE").split()


def make_l2(path: Path, hyperspectral: bool, shift: float = 0.0):
    ny, nx = 120, 100
    lat = np.linspace(23.0, 29.0, ny)[:, None] * np.ones((1, nx))
    lon = np.ones((ny, 1)) * np.linspace(-90.0, -79.0, nx)[None, :]
    with netCDF4.Dataset(path, "w") as ds:
        ds.time_coverage_start = "2025-06-10T17:21:54Z"
        ds.processing_version = "3.2" if hyperspectral else "R2022.0"
        ds.createDimension("number_of_lines", ny)
        ds.createDimension("pixels_per_line", nx)
        nav = ds.createGroup("navigation_data")
        nav.createVariable("latitude", "f4", ("number_of_lines", "pixels_per_line"))[:] = lat
        nav.createVariable("longitude", "f4", ("number_of_lines", "pixels_per_line"))[:] = lon
        geo = ds.createGroup("geophysical_data")
        sbp = ds.createGroup("sensor_band_parameters")
        base = 0.004 + 0.001 * np.sin(lat) + shift
        if hyperspectral:
            wl = np.arange(400, 720, 10.0)
            ds.createDimension("wavelength_3d", wl.size)
            sbp.createVariable("wavelength_3d", "f4", ("wavelength_3d",))[:] = wl
            v = geo.createVariable("Rrs", "i2", ("number_of_lines", "pixels_per_line", "wavelength_3d"),
                                   fill_value=-32767)
            v.scale_factor, v.add_offset, v.units = 2e-6, 0.05, "sr^-1"
            v[:] = base[:, :, None] * np.ones(wl.size)
        else:
            for w in (412, 443, 488, 555, 667):
                v = geo.createVariable(f"Rrs_{w}", "i2", ("number_of_lines", "pixels_per_line"), fill_value=-32767)
                v.scale_factor, v.add_offset, v.units = 2e-6, 0.05, "sr^-1"
                v[:] = base
        f = geo.createVariable("l2_flags", "i4", ("number_of_lines", "pixels_per_line"))
        f.flag_masks = np.array([1 << i for i in range(32)], dtype="i8").astype("i4")
        f.flag_meanings = " ".join(FLAGS)
        bits = np.zeros((ny, nx), "i4")
        bits[:, -10:] |= 1 << FLAGS.index("LAND")
        bits[:20, :] |= 1 << FLAGS.index("CLDICE")
        bits[40:45, :] |= 1 << FLAGS.index("STRAYLIGHT")
        f[:] = bits


def make_grid(path: Path, lon0_360: bool):
    lat = np.arange(20.0, 32.0, 0.25)
    lon = np.arange(260.0, 285.0, 0.25) if lon0_360 else np.arange(-100.0, -75.0, 0.25)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("latitude", lat.size)
        ds.createDimension("longitude", lon.size)
        ds.createVariable("time", "f8", ("time",))[:] = [0]
        ds["time"].units = "hours since 2025-06-10"
        ds.createVariable("latitude", "f4", ("latitude",))[:] = lat
        ds.createVariable("longitude", "f4", ("longitude",))[:] = lon
        v = ds.createVariable("ws", "f4", ("time", "latitude", "longitude"), fill_value=-9999.0)
        v.units = "m s-1"
        v[:] = np.random.default_rng(0).random((1, lat.size, lon.size)) * 10


@unittest.skipIf(netCDF4 is None, "netCDF4 not installed")
class InspectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = load_config(today=date(2026, 11, 11))

    def tearDown(self):
        self.tmp.cleanup()

    def test_l2_hyperspectral_and_multiband(self):
        from oefm import inspect as ins

        for hyper in (True, False):
            p = self.dir / f"x{int(hyper)}.nc"
            make_l2(p, hyper)
            prod = self.cfg.products["pace_oci_aop" if hyper else "modisa_oc"]
            res = ins.inspect_l2(p, prod, self.cfg)
            self.assertEqual(res["variable"], "Rrs")
            self.assertEqual(res["n_bands"], 32 if hyper else 5)
            st = res["regions"]["offshore_gulf"]
            self.assertGreater(st["n_pixels"], 0)
            v = st["mask_valid_fraction"]
            self.assertGreaterEqual(v["minimal"], v["l3_no_straylight"])
            self.assertGreaterEqual(v["l3_no_straylight"], v["l3_default"])
            json.dumps(res)  # serialisable
            pngs = ins.map_l2(p, prod, self.cfg, self.dir / "maps" / p.stem)
            self.assertEqual(len(pngs), 2)
            self.assertTrue(all(x.exists() for x in pngs))
            ins.structure(p)

    def test_index_stats_and_matchup(self):
        from oefm.index import granule_stats, matchup

        a, b = self.dir / "a.nc", self.dir / "b.nc"
        make_l2(a, True)
        make_l2(b, False, shift=0.0005)
        recs = granule_stats((str(self.cfg.path), str(a), {"granule_key": "k"}, "2026-11-11"))
        regions = {r["region"] for r in recs}
        self.assertEqual(regions, set(self.cfg.regions))
        self.assertTrue(all(0 <= r["valid_cover_minimal"] <= 1 for r in recs))
        res = matchup(self.cfg, a, b, self.cfg.regions["offshore_gulf"], self.dir / "ov")
        band555 = next(x for x in res["bands"] if x["target_nm"] == 555)
        self.assertGreater(band555["n_cells"], 100)
        self.assertGreater(band555["median_ratio_b_over_a"], 1.0)
        self.assertTrue(Path(res["figure"]).exists())

    def test_grid_0_360(self):
        from oefm import inspect as ins

        for wrap in (True, False):
            p = self.dir / f"g{int(wrap)}.nc"
            make_grid(p, wrap)
            res = ins.inspect_grid(p, self.cfg.products["ccmp_wind"], self.cfg)
            self.assertEqual(res["lon_convention"], "0-360" if wrap else "-180-180")
            self.assertLess(res["variables"]["ws"]["domain_nan_fraction"], 0.01)
            self.assertEqual(len(ins.map_grid(p, self.cfg.products["ccmp_wind"], self.cfg, self.dir / f"m{int(wrap)}")), 1)


if __name__ == "__main__":
    unittest.main()
