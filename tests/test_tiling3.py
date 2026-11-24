"""catalog_v1 and tiling v3 on synthetic OB.DAAC-like L2 granules (no data files)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import netCDF4
    import zarr

    from oefm import catalog as C
    from oefm import tiling3
    from oefm.config import load_config
except ImportError:  # pragma: no cover
    netCDF4 = None

FLAGS = ("ATMFAIL LAND PRODWARN HIGLINT HILT HISATZEN COASTZ SPARE STRAYLIGHT CLDICE COCCOLITH TURBIDW "
         "HISOLZEN SPARE LOWLW CHLFAIL NAVWARN ABSAER SPARE MAXAERITER MODGLINT CHLWARN ATMWARN SPARE "
         "SEAICE NAVFAIL FILTER SPARE BOWTIEDEL HIPOL PRODFAIL SPARE").split()
BIT = {n: 1 << i for i, n in enumerate(FLAGS) if n != "SPARE"}
NY, NX = 130, 120


def make_l2(path: Path, kind: str = "modis"):
    """~1 km pixels over 29.4-30.6 N, 88.0-86.6 W (crosses the 30 N zone border). Field: Rrs = 0.004 +
    1e-3 * (lat - 30). Clouds (CLDICE, Rrs fill) in the first 15 rows; LAND (fill) in the last 8
    columns; STRAYLIGHT rows 60-64 (values kept); HISATZEN on every other pixel of columns 20-40 with
    Rrs + 0.002 (so cells there are mixed and the clean mean differs). For SFREFL, rhos stays finite
    under CLDICE (as in real files)."""
    lat = np.linspace(29.4, 30.6, NY)[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * np.linspace(-88.0, -86.6, NX)[None, :]
    field = 0.004 + 1e-3 * (lat - 30.0)
    bits = np.zeros((NY, NX), "i8")
    bits[:15, :] |= BIT["CLDICE"]
    bits[:, -8:] |= BIT["LAND"]
    bits[60:65, :] |= BIT["STRAYLIGHT"]
    hz = np.zeros((NY, NX), bool)
    hz[:, 20:41] = (np.add.outer(np.arange(NY), np.arange(21)) % 2) == 0
    bits[hz] |= BIT["HISATZEN"]
    bits[0, 0] |= 1 << 31     # bit 31 set somewhere (negative int32 mask in real files)
    val = field + np.where(hz, 0.002, 0.0)
    fill = (bits & (BIT["CLDICE"] | BIT["LAND"])) != 0
    with netCDF4.Dataset(path, "w") as ds:
        ds.time_coverage_start = "2024-06-10T17:00:00Z"
        ds.time_coverage_end = "2024-06-10T17:05:00Z"
        ds.createDimension("number_of_lines", NY)
        ds.createDimension("pixels_per_line", NX)
        nav = ds.createGroup("navigation_data")
        nav.createVariable("latitude", "f4", ("number_of_lines", "pixels_per_line"))[:] = lat
        nav.createVariable("longitude", "f4", ("number_of_lines", "pixels_per_line"))[:] = lon
        geo = ds.createGroup("geophysical_data")
        sbp = ds.createGroup("sensor_band_parameters")
        if kind in ("pace", "sfrefl"):
            wl = np.arange(400, 720, 40.0)
            ds.createDimension("wavelength_3d", wl.size)
            sbp.createVariable("wavelength_3d", "f4", ("wavelength_3d",))[:] = wl
            if kind == "pace":
                v = geo.createVariable("Rrs", "i2", ("number_of_lines", "pixels_per_line", "wavelength_3d"), fill_value=-32767)
                v.scale_factor, v.add_offset = 2e-6, 0.05
                v[:] = np.ma.masked_array(val[:, :, None] * np.ones(wl.size), np.repeat(fill[:, :, None], wl.size, 2))
                u = geo.createVariable("Rrs_unc", "i2", ("number_of_lines", "pixels_per_line", "wavelength_3d"), fill_value=-32767)
                u.scale_factor, u.add_offset = 5e-8, 0.0015
                u[:] = np.ma.masked_array(np.full((NY, NX, wl.size), 1e-4), np.repeat(fill[:, :, None], wl.size, 2))
                a = geo.createVariable("avw", "f4", ("number_of_lines", "pixels_per_line"), fill_value=-32767.0)
                a[:] = np.ma.masked_array(np.full((NY, NX), 520.0), fill)
            else:
                v = geo.createVariable("rhos", "f4", ("number_of_lines", "pixels_per_line", "wavelength_3d"), fill_value=-32767.0)
                land = (bits & BIT["LAND"]) != 0
                v[:] = np.ma.masked_array((10 * field)[:, :, None] * np.ones(wl.size), np.repeat(land[:, :, None], wl.size, 2))
        else:
            for w in (412, 443, 488, 555, 667):
                v = geo.createVariable(f"Rrs_{w}", "i2", ("number_of_lines", "pixels_per_line"), fill_value=-32767)
                v.scale_factor, v.add_offset = 2e-6, 0.05
                v[:] = np.ma.masked_array(val, fill)
            c = geo.createVariable("chlor_a", "f4", ("number_of_lines", "pixels_per_line"), fill_value=-32767.0)
            c[:] = np.ma.masked_array(np.full((NY, NX), 0.2), fill)
            k = geo.createVariable("Kd_490", "i2", ("number_of_lines", "pixels_per_line"), fill_value=-32767)
            k.scale_factor, k.add_offset = 2e-4, 0.0
            k[:] = np.ma.masked_array(np.full((NY, NX), 0.03), fill)
        f = geo.createVariable("l2_flags", "i4", ("number_of_lines", "pixels_per_line"))
        f.flag_masks = np.array([1 << i for i in range(32)], dtype="i8").astype("i4")
        f.flag_meanings = " ".join(FLAGS)
        f[:] = bits.astype("i8").astype("u4").astype("i4")
    return field


@unittest.skipIf(netCDF4 is None, "processing dependencies not installed")
class TestCatalog(unittest.TestCase):
    def test_zones(self):
        zs = C.zones()
        self.assertEqual(len(zs), 196)
        self.assertEqual(len({z.key for z in zs}), len(zs))
        self.assertEqual(sum(z.key.startswith("Z+00") for z in zs), 24)
        self.assertEqual(list(C.zone_of([29.99, 30.0, 75.0, -75.01], [-87, -87, 0, 0])),
                         ["Z+15_05", "Z+30_05", "ZN", "ZS"])
        json.dumps(C.spec())

    def test_scale_error(self):
        for z in C.zones():
            la, lo = np.meshgrid(np.linspace(max(z.south, -89.5), min(z.north, 89.5), 15),
                                 np.linspace(z.west, z.east, 15))
            la, lo = la.ravel(), lo.ravel()
            x, y = z.proj.forward(lo, la)
            x2, y2 = z.proj.forward(lo, la + 1e-4)
            k = np.hypot(x2 - x, y2 - y) / (np.radians(1e-4) * 6371007.181)
            self.assertLess(np.abs(k - 1).max(), 0.009 if z.is_cap else 0.005, z.key)

    def test_point_uses_own_zone_and_border_tiles_in_both(self):
        tid, r, c = C.point_tile(30.01, -87.0)
        self.assertTrue(tid.startswith("Z+30_05:"))
        tid2, *_ = C.point_tile(29.99, -87.0)
        self.assertTrue(tid2.startswith("Z+15_05:"))
        # the Z+15 tile containing 30.01 N exists as a border member of Z+15 too
        z = C.zone_map()["Z+15_05"]
        x, y = z.proj.forward(np.array([-87.0]), np.array([30.01]))
        ix, iy = int(np.floor(x[0] / C.TILE_M)), int(np.floor(y[0] / C.TILE_M))
        self.assertTrue(C.is_member(z, ix, iy)[0])
        self.assertEqual(C.parse_tile_id(C.tile_id("Z+15_05", ix, iy)), ("Z+15_05", ix, iy))

    def test_solar_angles(self):
        t = np.array([np.datetime64("2024-03-20T12:07:00").astype("datetime64[s]").astype(float)])
        z, a = tiling3.solar_angles(t, np.array([0.0]), np.array([0.0]))
        self.assertLess(z[0], 1.0)
        t = np.array([np.datetime64("2024-06-21T12:00:00").astype("datetime64[s]").astype(float)])
        z, a = tiling3.solar_angles(t, np.array([60.0]), np.array([0.0]))
        self.assertAlmostEqual(z[0], 60 - 23.44, delta=0.5)
        self.assertAlmostEqual(a[0], 180.0, delta=2.0)


@unittest.skipIf(netCDF4 is None, "processing dependencies not installed")
class TestTiling3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.cfg = load_config()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tile(self, kind, product):
        src = self.d / f"{kind}.nc"
        field = make_l2(src, kind)
        row = {"granule_key": f"k_{kind}", "product": product, "filename": src.name, "time_start": "2024-06-10T17:00:00Z",
               "sha256": "x"}
        root = self.d / f"v3_{kind}"
        res = tiling3.tile_granule((str(self.cfg.path), row, str(src), str(root), {"git": "test"}))
        self.assertIsNone(res["error"])
        return res, field

    def open(self, z):
        st = zarr.ZipStore(z["dest"], mode="r")
        return zarr.group(store=st), st

    def test_modis_like(self):
        res, field = self.tile("modis", "modisa_oc")
        zones = {z["zone"]: z for z in res["zones"]}
        self.assertIn("Z+15_05", zones)
        self.assertIn("Z+30_05", zones)
        z = zones["Z+15_05"]
        self.assertEqual(z["status"], "done")
        self.assertGreaterEqual(len(z["overpasses"]), len(z["tiles"]))
        self.assertTrue(all(o["stored"] == (o["valid_fraction"] >= 0.01) for o in z["overpasses"]))
        g, st = self.open(z)
        a = g.attrs.asdict()
        self.assertEqual(a["catalog"], "catalog_v1")
        self.assertEqual(g["refl"].dtype, np.int16)
        refl = g["refl"][:].astype(float)
        ok = refl != -32767
        val = refl * 2e-6 + 0.05
        # values: the field (+0.002 on HISATZEN pixels mixed in); never outside the input range
        self.assertTrue(ok.any())
        self.assertTrue((val[ok] > 0.0033).all() and (val[ok] < 0.0071).all())
        # clean mean only in mixed cells and there it is the field (HISATZEN excluded for MODIS)
        cl = g["refl_clean"][:].astype(float)
        okc = cl != -32767
        self.assertTrue(okc.any())
        self.assertLess(okc.sum(), ok.sum())
        # flags: shares in [0, 255]; STRAYLIGHT present and its cells keep values
        fs = g["flag_share"][:]
        # where the mixing comes from HISATZEN pixels (Rrs + 0.002) the clean mean is lower; STRAYLIGHT
        # (also hard) pixels carry the field itself, so there the two means agree
        hzc = (fs[:, FLAGS.index("HISATZEN")] > 0) & (fs[:, FLAGS.index("STRAYLIGHT")] == 0)
        d = (val - (cl * 2e-6 + 0.05))[:, 0]
        self.assertTrue(((d > 0.0002) | ~okc[:, 0] | ~hzc).all())
        self.assertTrue((okc[:, 0] & hzc).any())
        st_only = (fs[:, FLAGS.index("STRAYLIGHT")] > 0) & (fs[:, FLAGS.index("HISATZEN")] == 0)
        self.assertTrue((np.abs(d)[okc[:, 0] & st_only] < 1e-5).all())
        self.assertGreater(int(fs[:, FLAGS.index("STRAYLIGHT")].max()), 0)
        stray = fs[:, FLAGS.index("STRAYLIGHT")] > 0
        self.assertTrue(ok[:, 0][stray].any())
        self.assertGreater(int(g["flags_or"][:].max()), 1 << 30)               # bit 31 kept
        # extras
        self.assertEqual(g["x_chlor_a"].dtype, np.float32)
        self.assertEqual(g["x_Kd_490"].dtype, np.int16)
        ch = g["x_chlor_a"][:]
        self.assertTrue(np.allclose(ch[np.isfinite(ch)], 0.2, atol=1e-5))
        # angles and time
        sz = g["solz"][:]
        cov = g["w_all"][:] > 0
        self.assertTrue(((sz[cov] * 0.01 > 5) & (sz[cov] * 0.01 < 30)).all())   # 17 UTC in June at 30 N, 87 W
        self.assertTrue((g["senz"][:][cov] != -32767).all())
        self.assertEqual(len(a["tile_ids"]), g["refl"].shape[0])
        st.close()

    def test_point_lookup_is_own_zone_and_in_stored_tiles(self):
        res, _ = self.tile("modis", "modisa_oc")
        stored = {t["tile_id"] for z in res["zones"] for t in z["tiles"]}
        tid, r, c = C.point_tile(30.2, -87.5)
        self.assertTrue(tid.startswith("Z+30_05:"))
        self.assertIn(tid, stored)

    def test_pace_like_with_uncertainty(self):
        res, _ = self.tile("pace", "pace_oci_aop")
        z = [z for z in res["zones"] if z["status"] == "done"][0]
        g, st = self.open(z)
        self.assertIn("refl_unc", g)
        u = g["refl_unc"][:]
        self.assertTrue(np.allclose(u[u != -32767] * 5e-8 + 0.0015, 1e-4, atol=5e-8))
        self.assertEqual(g["refl"].shape[1], 8)
        self.assertEqual(g["x_avw"].dtype, np.float32)
        st.close()

    def test_sfrefl_quantised_and_cloud_land_empty(self):
        res, field = self.tile("sfrefl", "pace_oci_sfrefl")
        z = [z for z in res["zones"] if z["status"] == "done"][0]
        g, st = self.open(z)
        v = g["refl"][:]
        self.assertEqual(v.dtype, np.int16)
        ok = v != -32767
        rh = v[ok] * 2e-5 + 0.6
        self.assertTrue((rh > 0.033).all() and (rh < 0.071).all())
        # cells with only cloud pixels are empty although rhos was finite there
        cl = g["flag_share"][:, FLAGS.index("CLDICE")]
        full_cloud = cl == 255
        self.assertTrue(full_cloud.any())
        self.assertFalse(ok[:, 0][full_cloud].any())
        self.assertEqual(g.attrs["sfrefl_empty_flags"], ["CLDICE", "LAND"])
        st.close()

    def test_run_record_index_and_resume(self):
        src = self.d / "run_modis.nc"
        make_l2(src, "modis")
        with mock.patch.dict(os.environ, {"OEFM_DATA_ROOT": str(self.d / "data")}):
            cfg = load_config()
            (cfg.data_root / "raw").mkdir(parents=True, exist_ok=True)
            os.replace(src, cfg.data_root / "raw" / src.name)
            row = {"granule_key": "g1", "product": "modisa_oc", "filename": src.name, "time_start": "2024-06-10T17:00:00Z",
                   "sha256": "x", "local_path": f"raw/{src.name}"}
            s1 = tiling3.run(cfg, rows=[row], workers=1, log=lambda *a: None)
            self.assertEqual(s1, {"done": 1})
            self.assertEqual(tiling3.run(cfg, rows=[row], workers=1, log=lambda *a: None), {})   # resumable
            out = tiling3.export_index(cfg, log=lambda *a: None)
            import pandas as pd

            t = pd.read_parquet(out["tile_observations"])
            o = pd.read_parquet(out["overpasses"])
            self.assertGreater(len(t), 0)
            self.assertGreaterEqual(len(o), len(t))
            self.assertTrue(set(t["tile_id"]) <= set(o["tile_id"]))
            self.assertTrue((tiling3.v3_root(cfg) / "catalog.json").exists())

    def test_download_select_skips_tiled(self):
        import argparse

        from oefm.download import select

        class Man:
            def rows(self, where, params):
                self.where = where
                return []
        m = Man()
        select(m, ["modisa_oc"], argparse.Namespace(recheck=False, per_product=0, start=None, end=None, region=None,
                                                    max_files=0))
        self.assertIn("'tiled'", m.where)


if __name__ == "__main__":
    unittest.main()
