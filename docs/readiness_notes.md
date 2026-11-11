### Known issues (found while building the pipeline, 2026-11-11)

These come from archive checks, not from the downloaded files. Confirm or revise them after
the smoke test and QC runs.

- **OLCI coverage at OB.DAAC stops in April 2026.** NASA's OLCI EFR L2 (R2022.0) has about
  one granule per day per satellite over the pilot domain through March 2026. The last
  Sentinel-3A granule anywhere in CMR is from 2026-04-14. Sentinel-3B is thin in Apr-Jul 2025
  (2 granules in Jun 2025) and essentially absent after March 2026. The NRT collections end at
  the same time. Unless OB.DAAC resumes, OLCI from April 2026 on would have to come from
  EUMETSAT/CDSE `OL_2_WFR` (different atmospheric correction, Baseline Collection 003). The
  CDSE download code in `biogoship_granite_kegg` (`biogoship/cdse.py`) can be reused for that.
  Mixing the two processors needs a cross-calibration check on overlapping dates.
- **MODIS Aqua ends 2026-05-31** in CMR. Expect no MODIS in the last ~5 months of the core window.
- **Processing versions differ across sensors:** MODIS, VIIRS SNPP/NOAA-20 and OLCI are R2022.0,
  NOAA-21 VIIRS is 2025.0, PACE AOP is 3.2 and PACE SFREFL is 3.1. Spectral harmonisation should
  key on (sensor, version), and the manifest records the version of every file.
- **Duplicate CMR records:** many MODIS Aqua files from 2021-07 to 2025-11 are listed twice under
  different GranuleURs. They are deduplicated by file name, and both concept ids are kept.
- **IMERG Final (V07) ends 2025-09-30.** The Late run fills later days (`imerg_late`). Late is
  satellite-only with climatological gauge calibration, so it is not identical to Final. Mark it
  as a separate source in training.
- **CCMP v3.1 lags by about 6 weeks** (latest file 2026-09-30 when checked). Altimetry MY ends
  2026-05-15 and GlobCurrent MY ends 2026-03-31. NRT datasets fill the rest automatically.
- **GlobCurrent's Ekman component** is fitted to drifters but driven by ERA5 wind stress (reanalysis
  forcing). It is listed as observation-based analysis with that caveat.
- **Swath vs. region size:** PACE SFREFL granules are ~730 MB each, and most of each swath lies
  outside the regions. Raw storage is therefore dominated by SFREFL and MUR (global daily files,
  ~680 MB). The regional-extraction estimate in section 2 shows the reduction.
- **Coastal 300 m:** OB.DAAC flags STRAYLIGHT and LAND-adjacent pixels. The QC table compares the
  default L3 mask with and without STRAYLIGHT, split by GEBCO depth (0-20 m, 20-200 m, >200 m), to
  show how much near-shore OLCI data a mask policy would cost. No pixels are removed from files.
- **Physical lead-in:** with 90 days before every window, the physical windows run almost
  continuously from 2017-11 to the present. Downloading physical products for full years costs
  little extra.
- **Swath-edge geometry dominates the mask choice** (smoke test, 2025-06-10). When the Keys and
  shelf fall near a swath edge, HISATZEN removes 99-100% of PACE and NOAA-21 pixels under the
  default L3 mask. The minimal mask keeps about half. Decide the HISATZEN policy per sensor before
  computing coverage statistics for training.
- **Negative red/NIR Rrs in clear water:** in one clear-water granule, 10-35% of PACE values at
  689-703 nm were negative. This is retained as-is.
- **Red-band disagreement between sensors:** on the first same-day matchups, blue-green Rrs agreed
  within about 10-15% (r 0.9-0.98), but 665 nm ratios were 0.4-0.7. That band needs explicit
  harmonisation (or uncertainty weighting) before it can be shared across sensors.
