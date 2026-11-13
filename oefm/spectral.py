"""Band response functions and the reflectance transform (numpy only; shared by the CPU pipeline
and the PyTorch model).

Spectral response functions (SRFs). The OB.DAAC Level-2 files carry centre wavelengths but no band
widths, so v1 uses Gaussian SRFs with each sensor's nominal FWHM (below). Measured relative
spectral response tables (OB.DAAC publishes them per sensor) can replace these later through
`srf_matrix(..., table=...)` without changing anything else. All SRFs are sampled on one 1 nm grid,
340-2300 nm, which covers the SWIR bands of PACE SFREFL (to ~2258 nm).

Transform (design document, "Negative and missing reflectance" and "Normalization"):
    y = (asinh(x / s(lambda)) - m(lambda)) / sd(lambda)
with s, m, sd smooth functions of wavelength fitted on training data pooled over all sensors
(`oefm.normstats`), so the same wavelength gets the same normalisation whatever the sensor.
Values below -5 s(lambda) are implausible and become missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GRID_NM = np.arange(340.0, 2301.0, 1.0)  # every view: Rrs (340-720 in practice) and SFREFL rhos (to ~2258)

# Nominal FWHM (nm) per family; exceptions per centre wavelength (nearest within 2 nm).
FWHM_DEFAULT = {"PACE": 5.0, "VIIRS": 20.0, "OLCI": 10.0, "MODIS": 10.0}
FWHM_BANDS = {
    "MODIS": {412: 15.0, 469: 20.0, 555: 20.0, 645: 50.0},
    "OLCI": {400: 15.0, 674: 7.5, 681: 7.5},
    "VIIRS": {},
    "PACE": {},
}
SFREFL_FWHM_SWIR = {1038: 75.0, 1249: 30.0, 1618: 75.0, 2131: 50.0, 2258: 75.0}
IMPLAUSIBLE_K = 5.0


def nominal_fwhm(family: str, wavelengths) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype="f8")
    out = np.full(wl.shape, FWHM_DEFAULT.get(family, 10.0))
    table = dict(FWHM_BANDS.get(family, {}))
    if family == "PACE":
        table.update(SFREFL_FWHM_SWIR)
    for c, f in table.items():
        out[np.abs(wl - c) <= 2.0] = f
    out[(family == "PACE") & (wl > 900) & ~np.isin(np.round(wl), list(SFREFL_FWHM_SWIR))] = 20.0
    return out


def srf_matrix(wavelengths, fwhm, grid: np.ndarray = GRID_NM) -> np.ndarray:
    """[C, G] response of each band on the grid, each row summing to 1 (Gaussian v1)."""
    wl = np.asarray(wavelengths, dtype="f8")[:, None]
    sig = np.asarray(fwhm, dtype="f8")[:, None] / 2.3548
    r = np.exp(-0.5 * ((grid[None, :] - wl) / sig) ** 2)
    s = r.sum(axis=1, keepdims=True)
    return np.where(s > 0, r / np.maximum(s, 1e-12), 0.0)


def interp_matrix(src_wl, grid) -> np.ndarray:
    """[G, P] linear-interpolation weights from samples at src_wl to the grid (0 outside)."""
    src = np.asarray(src_wl, dtype="f8")
    M = np.zeros((grid.size, src.size))
    j = np.clip(np.searchsorted(src, grid) - 1, 0, src.size - 2)
    a = (grid - src[j]) / (src[j + 1] - src[j])
    inside = (grid >= src[0]) & (grid <= src[-1])
    M[np.arange(grid.size), j] = np.where(inside, 1 - a, 0.0)
    M[np.arange(grid.size), j + 1] += np.where(inside, a, 0.0)
    return M


def convolution_matrix(spectrum_wl, band_wl, band_fwhm) -> np.ndarray:
    """[P, C]: hyperspectral samples -> band values (SRF-weighted), a linear map."""
    swl = np.asarray(spectrum_wl, dtype="f8")
    grid = np.arange(np.floor(swl.min()), np.ceil(swl.max()) + 1.0)
    R = srf_matrix(band_wl, band_fwhm, grid)          # [C, G]
    return (R @ interp_matrix(swl, grid)).T            # [P, C]


def convolve(spectrum_wl, spectrum, band_wl, band_fwhm, min_mass: float = 0.8) -> np.ndarray:
    """Values a sensor with the given bands would record from hyperspectral spectra.
    spectrum [..., P] on spectrum_wl [P] -> [..., C]; NaN where less than min_mass of a band's
    response falls on finite samples (missing samples are left out and the rest renormalised)."""
    M = convolution_matrix(spectrum_wl, band_wl, band_fwhm)
    x = np.asarray(spectrum, dtype="f8")
    ok = np.isfinite(x)
    mass = ok.astype("f8") @ M
    val = np.where(ok, x, 0.0) @ M
    full = M.sum(axis=0)
    return np.where(mass >= min_mass * full, val / np.maximum(mass, 1e-12) * full, np.nan)


class Normalizer:
    """The frozen transform, from normstats.json (one entry per view type: 'rrs', 'surface')."""

    def __init__(self, stats: dict) -> None:
        self.stats = stats

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        return cls(json.loads(Path(path).read_text()))

    @staticmethod
    def _curve(c: dict, wl) -> np.ndarray:
        x = (np.asarray(wl, dtype="f8") - c["center"]) / c["scale"]
        x = np.clip(x, -1.2, 1.2)  # no wild extrapolation outside the fitted wavelength range
        y = np.polyval(c["coef"], x)
        return np.exp(y) if c.get("log") else y

    def params(self, view: str, wl) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        st = self.stats[view]
        return self._curve(st["s"], wl), self._curve(st["mean"], wl), self._curve(st["std"], wl)

    def transform(self, view: str, wl, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x [C, ...] raw reflectance -> (y float32, valid bool); band axis first."""
        s, m, sd = (p.reshape((-1,) + (1,) * (x.ndim - 1)) for p in self.params(view, wl))
        valid = np.isfinite(x) & (x >= -IMPLAUSIBLE_K * s)
        y = (np.arcsinh(np.where(valid, x, 0.0) / s) - m) / sd
        return np.where(valid, y, 0.0).astype("f4"), valid
