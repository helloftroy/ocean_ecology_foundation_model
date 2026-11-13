"""Model checks on synthetic batches (skipped where PyTorch is not installed)."""

import unittest

import numpy as np

try:
    import torch

    from oefm.model.data import collate, sample_mask
    from oefm.model.net import Model, effective_rank
    from oefm.model.sigreg import SIGReg
except ImportError:
    torch = None

TINY = {"arm": "B", "model": {"dim": 32, "heads": 4, "mlp_ratio": 2.0, "drop_path": 0.0, "spatial_depth": 1,
                              "temporal_depth": 1, "pred_dim": 32, "pred_heads": 4, "pred_temporal_depth": 1,
                              "pred_spatial_depth": 1, "decoder_depth": 1, "n_fourier": 8, "pace_bands": 6,
                              "checkpoint": False}}


def fake_obs(product, wl, olci=False, dt=0.0, rng=None, pace=False):
    rng = rng or np.random.default_rng(0)
    hw = 128 if olci else 32
    c = len(wl)
    fam = {"viirsn_oc": "VIIRS", "olcia_efr_oc": "OLCI", "pace_oci_aop": "PACE", "modisa_oc": "MODIS"}[product]
    o = {"product": product, "family": fam, "view": "rrs", "olci": olci, "wl": np.asarray(wl, float), "fwhm": np.full(c, 10.0),
         "values": rng.standard_normal((c, hw, hw)).astype(np.float16), "band_mask": np.ones((c, hw, hw), bool),
         "quality": np.zeros((9, 32, 32), np.uint8), "cell_valid": np.ones(1024, bool), "dt_days": dt, "is_pace": pace,
         "raw_mean": np.full(1024, 0.004, np.float32)}
    o["band_mask_ctx"] = o["band_mask"]
    if pace:
        o["pace_resid"] = rng.standard_normal((c, 1024)).astype(np.float16)
        o["pace_resid_mask"] = np.ones((c, 1024), bool)
    return o


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestModel(unittest.TestCase):
    def test_sigreg_detects_collapse(self):
        g = torch.Generator().manual_seed(0)
        s = SIGReg(256)
        iso = s(torch.randn(2048, 64, generator=g))
        col = s(torch.randn(2048, 1, generator=g).repeat(1, 64))
        self.assertLess(float(iso), 5.0)
        self.assertGreater(float(col), 100 * float(iso))

    def test_tokenizer_keeps_amplitude_and_view(self):
        """Design document: no LayerNorm on the token; spectra with the same shape scaled by 0.5, 1
        and 2 give distinguishable tokens, and a linear probe on tokens recovers log amplitude
        (R^2 >= 0.95) at initialisation. Rrs vs rhos at the same wavelengths also differ."""
        from oefm.model.net import SpectralTokenizer
        from oefm.spectral import GRID_NM, Normalizer, srf_matrix

        torch.manual_seed(0)
        tok = SpectralTokenizer(64, 8)
        wl = np.array([410, 443, 486, 551, 671.0])
        srf = torch.tensor(srf_matrix(wl, np.full(5, 20.0)), dtype=torch.float32)
        flat = {"coef": [0.0], "center": 500.0, "scale": 100.0}
        norm = Normalizer({"rrs": {"s": {"coef": [float(np.log(1e-4))], "center": 500.0, "scale": 100.0, "log": True},
                                   "mean": {"coef": [3.5], "center": 500.0, "scale": 100.0}, "std": {**flat, "coef": [1.0]}}})

        def t(raw, view=0):
            y, _ = norm.transform("rrs", wl, raw.reshape(5, 1, 1).repeat(32, 1).repeat(32, 2))
            g = {"values": torch.from_numpy(y)[None], "olci": False, "srf": srf, "grid": torch.tensor(GRID_NM, dtype=torch.float32),
                 "quality": torch.zeros(1, 9, 32, 32), "view": view, "raw_mean": torch.full((1, 1024), float(raw.mean()))}
            return tok(g, torch.ones(1, 5, 32, 32, dtype=torch.bool))[0, 0]

        def rel(a, b):
            return float((a - b).norm() / a.norm())

        shape = np.array([0.008, 0.007, 0.005, 0.002, 0.0003])
        for scale in (0.5, 2.0):
            self.assertGreater(rel(t(shape), t(scale * shape)), 0.05, scale)
        self.assertGreater(rel(t(shape, 0), t(shape, 1)), 0.05)
        rng = np.random.default_rng(0)
        X, Y = [], []
        for _ in range(300):
            sh = shape * np.exp(rng.normal(0, 0.3, 5))
            a = np.exp(rng.uniform(np.log(0.25), np.log(4.0)))
            X.append(t(a * sh).detach().numpy())
            Y.append(np.log(a))
        X, Y = np.asarray(X), np.asarray(Y)
        Xc = np.c_[X, np.ones(len(X))]
        w = np.linalg.solve(Xc[:200].T @ Xc[:200] + 1e-3 * np.eye(Xc.shape[1]), Xc[:200].T @ Y[:200])
        pred = Xc[200:] @ w
        r2 = 1 - np.sum((pred - Y[200:]) ** 2) / np.sum((Y[200:] - Y[200:].mean()) ** 2)
        self.assertGreaterEqual(r2, 0.95)
        k = tok.kernels(torch.tensor(srf_matrix([1249.0, 2131.0], [30.0, 50.0]), dtype=torch.float32),
                        torch.tensor(GRID_NM, dtype=torch.float32), False, 1)
        self.assertTrue(torch.isfinite(k).all() and k.abs().sum() > 0)

    def test_e2_reference_target(self):
        from oefm.model.data import MODIS_REF_WL, reference_target
        from oefm.spectral import Normalizer

        flat = {"coef": [0.0], "center": 500.0, "scale": 100.0}
        norm = Normalizer({"rrs": {"s": {"coef": [float(np.log(1e-4))], "center": 500.0, "scale": 100.0, "log": True},
                                   "mean": {"coef": [3.5], "center": 500.0, "scale": 100.0}, "std": {**flat, "coef": [1.0]}}})
        pwl = np.arange(340.0, 720.0, 2.5)
        raw = np.repeat((0.01 * np.exp(-((pwl - 450) / 80) ** 2))[:, None, None], 32, 1).repeat(32, 2)
        o = fake_obs("pace_oci_aop", pwl, pace=True)
        o["tgt_cells"] = np.zeros(1024, bool)
        o["tgt_cells"][:100] = True
        ref = reference_target(o, raw, pwl, norm)
        self.assertEqual(ref["values"].shape, (len(MODIS_REF_WL), 32, 32))
        self.assertEqual(int(ref["tgt_cells"].sum()), 100)
        self.assertFalse(ref["ctx_cells"].any())

    def test_group_sigreg(self):
        from oefm.model.sigreg import GroupSIGReg

        g = torch.Generator().manual_seed(0)
        s = GroupSIGReg(["MODIS", "VIIRS", "PACE"], 256, min_tokens=256)
        iso = {"MODIS": torch.randn(1000, 32, generator=g), "VIIRS": torch.randn(1000, 32, generator=g),
               "PACE": torch.randn(50, 32, generator=g)}
        tot, logs = s(iso, 32, torch.device("cpu"))
        self.assertIn("pooled", logs)
        self.assertLess(float(tot), 5.0)
        col = dict(iso)
        col["VIIRS"] = torch.randn(1000, 1, generator=g).repeat(1, 32)
        tot2, _ = s(col, 32, torch.device("cpu"))
        self.assertGreater(float(tot2), 20 * float(tot))

    def test_mask_sampler_keeps_context(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            obs = [fake_obs(p, [443, 555], dt=d) for p, d in (("viirsn_oc", -3), ("modisa_oc", 0), ("viirsn_oc", 4))]
            mode = sample_mask(obs, rng)
            if mode == "view":  # whole view = a whole view group (both VIIRS observations together)
                self.assertEqual(len({bool(o["tgt_cells"].any()) for o in obs if o["family"] == "VIIRS"}), 1)
            self.assertTrue(any(o["ctx_cells"].any() for o in obs), mode)
            self.assertTrue(any(o["tgt_cells"].any() for o in obs), mode)
            for o in obs:
                self.assertFalse((o["ctx_cells"] & o["tgt_cells"]).any())

    def test_forward_all_arms_and_group_size_equal_band_count(self):
        rng = np.random.default_rng(2)
        # five VIIRS observations with five bands: regression test for band metadata subsetting
        items = []
        for b in range(2):
            obs = [fake_obs("viirsn_oc", [410, 443, 486, 551, 671], dt=float(t), rng=rng) for t in range(-2, 3)]
            obs.append(fake_obs("olcia_efr_oc", [400, 443, 560], olci=True, dt=0.5, rng=rng))
            obs.append(fake_obs("pace_oci_aop", [350, 400, 450, 500, 550, 600], dt=-1, rng=rng, pace=True))
            sample_mask(obs, rng)
            items.append({"obs": obs, "mode": "x"})
        batch = collate(items)
        batch = {**batch, "device": torch.device("cpu"), "dtype": torch.float32}
        for arm in ("A", "B", "C"):
            m = Model({**TINY, "arm": arm})
            from oefm.model.sigreg import GroupSIGReg

            out = m(batch, sigreg=GroupSIGReg(["MODIS", "VIIRS", "OLCI", "PACE", "SURFACE"], 64), lam=0.05, mu=0.1,
                    sigreg_tokens=512)
            self.assertTrue(torch.isfinite(out["loss"]), arm)
            out["loss"].backward()
            e, mask = m.embed_e1(batch)
            self.assertEqual(tuple(e.shape), (2, 1024, 32))
            from oefm.model.net import embed_e3
            z, gm, _ = m.encode_context(batch)
            self.assertEqual(tuple(embed_e3(e, z, gm).shape), (2, 1024, 64))           # E3 = 2 D
            if arm in ("B", "C"):
                e2 = m.embed_e2(batch)  # default MODIS-Aqua reference bands
                self.assertEqual(tuple(e2.shape), (2, 1024, 32))
                self.assertTrue(torch.isfinite(e2).all())
            self.assertGreater(effective_rank(e[mask][:500]), 1.0)


if __name__ == "__main__":
    unittest.main()
