"""Final per-sensor quality policy (step 2.1), read from config `processing.qc`.

Three levels, all defined by l2_flags names (oefm/inspect.py MASKS):
- training mask: pixels averaged into training tiles. HISATZEN is a tag for PACE and VIIRS and
  a hard mask for MODIS; SFREFL has its own set.
- strict mask: the independent spectral-evaluation subset (NASA's full L3 set, HISATZEN included).
  Stored per cell as the weight of strict-passing pixels, so the subset is a selection, not a copy.
- always: non-finite reflectance is never averaged in.

Every pixel's flags are still OR-ed into its cell, so any other policy can be applied later.
The red-band angle weight is applied by the loader (step 2.5), not baked into stored values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .inspect import MASKS


@dataclass(frozen=True)
class Policy:
    product: str
    training_mask: str
    strict_mask: str

    @property
    def training_flags(self) -> list[str]:
        return MASKS[self.training_mask]

    @property
    def strict_flags(self) -> list[str]:
        return MASKS[self.strict_mask]


def policy(cfg: Config, product: str) -> Policy:
    qc = cfg.raw["processing"]["qc"]
    t = qc["training_mask"].get(product, qc["training_mask"]["default"])
    s = qc["strict_mask"].get(product, qc["strict_mask"]["default"])
    for m in (t, s):
        if m not in MASKS:
            raise ValueError(f"unknown mask {m!r} for {product}; known: {', '.join(MASKS)}")
    return Policy(product, t, s)


def red_angle_weight(cfg: Config, product: str, wavelength_nm, senz_deg) -> np.ndarray:
    """Weight in (0, 1] per (band, pixel): 1 below min_wavelength_nm or within the tolerance,
    tolerance / |ratio - 1| otherwise, with ratio the measured angle effect for that sensor and
    zenith class. Products without a measured ratio (e.g. SFREFL) get 1 everywhere."""
    ra = cfg.raw["processing"]["qc"]["red_angle"]
    wl = np.asarray(wavelength_nm, dtype="f8")
    z = np.asarray(senz_deg, dtype="f8")
    ratios = ra["ratio"].get(product)
    out = np.ones(np.broadcast_shapes(wl.shape, z.shape), dtype="f4")
    if ratios is None:
        return out
    edges = np.asarray(ra["classes_deg"], dtype="f8")
    k = np.clip(np.digitize(z, edges) - 1, 0, len(ratios) - 1)
    dev = np.abs(np.asarray(ratios, dtype="f8")[k] - 1.0)
    with np.errstate(divide="ignore"):
        w = np.where(dev > ra["tolerance"], ra["tolerance"] / dev, 1.0)
    w = np.where(np.isfinite(z), w, 1.0)
    return np.where(wl >= ra["min_wavelength_nm"], w, 1.0).astype("f4") * out
