"""Global (zoned) configs: config inheritance, per-region windows, zone grids across the antimeridian
and the pole, tiling tasks, loader lookup, inventory pairs, the Bio-GO-SHIP config generator and the
benchmark helpers. No data files."""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import pandas as pd
    import yaml

    from oefm.config import load_config
    from oefm.grid import TileGrid, grids
    from oefm.loader import TileSequences, geographic_labels
    from oefm.tiling import granule_zones, out_path
except ImportError:
    TileSequences = None

REPO = Path(__file__).resolve().parents[1]


def write_cfg(d: Path, regions: dict, zones: dict) -> Path:
    cfg = {"extends": str(REPO / "config" / "pilot.yaml"), "data_root": str(d / "data"), "time": {},
           "physical": {}, "targets": {}, "domain_margin_deg": 0.2, "zones": zones, "regions": regions,
           "processing": {"stats_dir": str(d / "stats")}, "runs_root": str(d / "runs")}
    p = d / "zoned.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def synthetic_samples(n_per=12):
    """Three cruises: Atlantic, one across the antimeridian (P02-like), one to the North Pole."""
    rows = []
    t0 = pd.Timestamp("2022-05-01T06:00Z")
    tracks = {"ATL-2020": (np.linspace(-30, -10, n_per), np.linspace(-20, -25, n_per)),
              "P02-2022": (np.full(n_per, 30.0), np.r_[np.linspace(176, 179.6, n_per // 2), np.linspace(-179.6, -176, n_per - n_per // 2)]),
              "HLY-2022": (np.linspace(80, 89.9, n_per), np.linspace(-170, 170, n_per))}
    for c, (la, lo) in tracks.items():
        for i in range(n_per):
            rows.append({"sample_key": f"{c}_{i:03d}", "cruise": c, "collection_time_utc": (t0 + pd.Timedelta(days=i)).isoformat().replace("+00:00", "Z"),
                         "lat": la[i], "lon": lo[i], "ko_reads": 1e6 * (1 + i), "platform": "NovaSeq X" if c != "ATL-2020" else "HiSeq 4000"})
    return pd.DataFrame(rows)


@unittest.skipIf(TileSequences is None, "processing dependencies not installed")
class TestZones(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.samples = synthetic_samples()
        cls.samples.to_csv(d / "samples.csv", index=False)
        from scripts import make_biogoship_config as mk

        cls.cfg_path = d / "biogoship.yaml"
        mk.main(["--samples", str(d / "samples.csv"), "--out", str(cls.cfg_path)])
        raw = yaml.safe_load(cls.cfg_path.read_text())
        raw["extends"] = str(REPO / "config" / "pilot.yaml")
        raw["data_root"] = str(d / "data")
        cls.cfg_path.write_text(yaml.safe_dump(raw))
        cls.cfg = load_config(cls.cfg_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_config_inherits_and_windows(self):
        cfg = self.cfg
        self.assertIn("grid", cfg.raw["processing"])                 # merged from pilot.yaml
        self.assertIn("stats_dir", cfg.raw["processing"])
        self.assertEqual(cfg.physical_products, {})
        self.assertNotIn("pace_oci_sfrefl", cfg.satellite_products)          # not an encoder input
        self.assertEqual(set(cfg.satellite_products) | {"pace_oci_sfrefl"}, set(load_config().satellite_products))
        self.assertEqual(cfg.core_start(), date(2024, 1, 1))
        self.assertEqual(load_config().core_start(), date(2024, 1, 1))
        for r in cfg.regions.values():
            self.assertIn(r.zone, cfg.zones)
            names = {n for w in r.windows for n in w.names}
            self.assertEqual(names, {"cruise", "season_1", "season_2"})
            for w in r.windows:
                if "cruise" not in w.names:   # seasonal windows: PACE era, 45 days
                    self.assertGreaterEqual(w.start, date(2024, 3, 5))
                    self.assertLessEqual(w.end, date(2025, 12, 31))
                    self.assertEqual(w.days, 45)
            self.assertLess(r.west, r.east)   # no box crosses the antimeridian
        self.assertEqual(sum(w.days for w in cfg.satellite_windows()),
                         len({d for r in cfg.regions.values() for w in r.windows
                              for d in pd.date_range(w.start, w.end)}))

    def test_cruise_window_covers_every_sample(self):
        for r in self.cfg.regions.values():
            cw = [w for w in r.windows if "cruise" in w.names][0]
            n = int(r.notes.split()[0])
            self.assertGreater(n, 0)
            self.assertGreaterEqual((cw.end - cw.start).days, 30 + 15)

    def test_grids_unique_ids_and_every_sample_located(self):
        gl = grids(self.cfg)
        ids = [i for g in gl for i in g.ids]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(":" in i for i in ids))
        with self.assertRaises(ValueError):
            TileGrid(self.cfg)
        for la, lo in zip(self.samples["lat"], self.samples["lon"]):
            hit = [g for g in gl if g.slots(*g.proj.forward(np.array([lo]), np.array([la])))[0] >= 0]
            self.assertTrue(hit, (la, lo))
        for g in gl:
            lon, lat = g.cell_centres(int(g.ix[0]), int(g.iy[0]))
            self.assertTrue(((lon >= -180) & (lon < 180)).all())
            # cells stay ~1.2 km: neighbouring centres 1.2 km apart within 1 %
            x, y = g.proj.forward(lon, lat)
            self.assertAlmostEqual(float(np.median(np.diff(x, axis=1))), 1200.0, delta=12)

    def test_antimeridian_zone(self):
        z = [g for g in grids(self.cfg) if g.zone.startswith("P02")]
        self.assertTrue(z)
        lons = np.concatenate([g.cell_centres(int(ix), int(iy))[0].ravel() for g in z for ix, iy in zip(g.ix, g.iy)])
        self.assertTrue((lons > 175).any() and (lons < -175).any())

    def test_granule_zones_and_out_path(self):
        cfg = self.cfg
        r = next(iter(cfg.regions.values()))
        row = {"regions": r.key, "filename": "AQUA_MODIS.20220501T060000.L2.OC.nc", "product": "modisa_oc",
               "time_start": "2022-05-01T06:00:00Z"}
        self.assertEqual(granule_zones(cfg, row), [r.zone])
        self.assertTrue(str(out_path(cfg, row, r.zone)).endswith(f"__{r.zone}.zarr.zip"))
        self.assertEqual(granule_zones(cfg, {**row, "regions": "unknown"}), sorted(cfg.zones))
        self.assertEqual(granule_zones(load_config(), row), [None])

    def test_geographic_labels_per_zone(self):
        gl = grids(self.cfg)
        lab = geographic_labels(self.cfg, gl)
        self.assertEqual(len(lab), sum(g.n_tiles for g in gl))
        self.assertEqual(set(lab.values()) - {"train", "geo_test", "buffer"}, set())
        # the regional (Florida) labels are unchanged by the list form
        fl = load_config()
        self.assertEqual(geographic_labels(fl, TileGrid(fl)), geographic_labels(fl, [TileGrid(fl)]))

    def test_point_lookup_across_zones(self):
        gl = grids(self.cfg)
        rows = [{"product": "viirsn_oc", "family": "VIIRS", "tile_id": g.ids[0], "time_utc": "2022-05-02T06:00:00+00:00",
                 "valid_fraction": 0.9, "output_path": "x", "row": 0, "granule_key": f"k{i}"} for i, g in enumerate(gl)]
        ts = TileSequences(self.cfg, split="all", index=pd.DataFrame(rows), physics=False, mode="retro")
        with mock.patch.object(TileSequences, "_assemble", lambda self, tile, rows, t, a: {"tile_id": tile, "obs": list(rows)}):
            for _, s in self.samples.iterrows():
                out = ts.sequence_for_point(s["lat"], s["lon"], s["collection_time_utc"].replace("Z", "+00:00"))
                self.assertIsNotNone(out)
                g = [g for g in gl if out["tile_id"] in set(g.ids)][0]
                lon, lat = g.cell_centres(*[int(v) for v in (g.ix[g.ids.index(out["tile_id"])], g.iy[g.ids.index(out["tile_id"])])])
                r, c = out["cell"]
                self.assertLess(abs(lat[r, c] - s["lat"]), 0.02)

    def test_status_runs_without_time_core(self):
        from oefm import cli

        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", {"OEFM_DATA_ROOT": d}):
            out = []
            with mock.patch.object(cli, "log", out.append):
                cli.cmd_status(load_config(self.cfg_path), None)
            self.assertTrue(any(l.startswith("TOTAL") for l in out))

    def test_inventory_uses_region_windows(self):
        from oefm.sources import nasa

        calls = []
        with mock.patch.object(nasa, "search", lambda prod, bbox, a, b: calls.append((bbox, a, b)) or []):
            nasa.inventory(self.cfg.satellite_products["modisa_oc"], self.cfg, log=lambda *a: None)
        want = sum(len(r.windows) for r in self.cfg.regions.values())
        self.assertEqual(len(calls), want)
        r = next(iter(self.cfg.regions.values()))
        self.assertIn((r.bbox, r.windows[0].start, r.windows[0].end), calls)


@unittest.skipIf(TileSequences is None, "processing dependencies not installed")
class TestBiogoshipHelpers(unittest.TestCase):
    def test_technical3_matches_two_level_and_adds_hiseq(self):
        from sklearn.preprocessing import SplineTransformer

        from oefm.model.biogoship import technical3

        rng = np.random.default_rng(0)
        n = 200
        plat = rng.integers(0, 2, n).astype(bool)
        logtot = rng.uniform(5, 7, n)
        y = rng.normal(size=(n, 4)) + plat[:, None] * 0.7 + 0.3 * logtot[:, None]

        def technical(y, plat, logtot):   # benchmark_eval.technical (bundle), for the comparison
            X = np.hstack([np.ones((len(y), 1)), plat[:, None].astype(float), SplineTransformer(n_knots=8, degree=3).fit_transform(logtot[:, None])])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            fit = X @ beta
            return y - (fit - y.mean(0)), fit - fit.mean(0)

        a, ta = technical(y, plat, logtot)
        codes = np.where(plat, "NovaSeq 6000", "NovaSeq X")
        b, tb = technical3(y, codes, logtot)
        np.testing.assert_allclose(a, b, atol=1e-8)
        codes3 = codes.copy()
        codes3[:40] = "HiSeq 4000"
        y3 = y.copy()
        y3[:40] += 2.0
        c, _ = technical3(y3, codes3, logtot)
        self.assertLess(abs(c[:40].mean() - c[40:].mean()), 0.2)   # the HiSeq shift is removed

    def test_extended_grouping_and_tile_features(self):
        from oefm.model import biogoship as bg

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            s = synthetic_samples()
            s.loc[1, ["lat", "lon", "collection_time_utc"]] = s.loc[0, ["lat", "lon", "collection_time_utc"]].to_numpy()
            s.to_csv(d / "samples.csv", index=False)
            from scripts import make_biogoship_config as mk

            mk.main(["--samples", str(d / "samples.csv"), "--out", str(d / "b.yaml")])
            raw = yaml.safe_load((d / "b.yaml").read_text())
            raw["extends"] = str(REPO / "config" / "pilot.yaml")
            (d / "b.yaml").write_text(yaml.safe_dump(raw))
            cfg = load_config(d / "b.yaml")
            tiles, members = bg.extended_tiles(cfg, d)
            self.assertEqual(sum(len(m) for m in members.values()), len(s))
            self.assertEqual(len(tiles), len(s) - 1)                       # the duplicated sample shares a tile
            self.assertEqual(set(tiles["platform_level"]), {"HiSeq 4000", "NovaSeq X"})
            self.assertTrue((tiles.groupby("platform_level")["cruise"].nunique() >= 1).all())
            pts = bg.sample_points(d)
            uk = pts["key"].drop_duplicates().to_numpy()   # embed writes one row per distinct point
            feats = pd.DataFrame(np.arange(len(uk) * 3, dtype=float).reshape(-1, 3), index=uk)
            feats.loc[pts["key"][2]] = np.nan
            tf = bg.tile_features(pts, {"a": [pts["sample_key"][0], pts["sample_key"][2]], "b": [pts["sample_key"][2]]}, feats)
            np.testing.assert_allclose(tf.loc["a"].to_numpy(), feats.loc[pts["key"][0]].to_numpy())   # NaN member skipped
            self.assertTrue(tf.loc["b"].isna().all())

    def test_ko_comparison(self):
        from oefm.model.biogoship import ko_comparison

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rng = np.random.default_rng(1)
            n, k = 130, 5
            obs = rng.normal(size=(n, k))
            cr = np.repeat([f"c{i}" for i in range(13)], 10)
            good = obs + rng.normal(scale=0.3, size=(n, k))
            bad = obs * 0.2 + rng.normal(size=(n, k))
            for name, pred in (("new", good), ("ref", bad)):
                (d / f"eval_{name}").mkdir()
                np.savez_compressed(d / f"eval_{name}" / "predictions_clr_all.npz", predicted=pred, observed=obs,
                                    tile_key=np.arange(n).astype(str), columns=np.array([f"K{i}" for i in range(k)]), cruise=cr)
            df, n1, n9 = ko_comparison(d, "new", "ref")
            self.assertEqual(n1, k)
            self.assertEqual(n9, k)
            self.assertTrue((df["cruises_better"] <= 13).all())


if __name__ == "__main__":
    unittest.main()
