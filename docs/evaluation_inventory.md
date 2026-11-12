# Evaluation-data inventory (Phase 2)

Checked 2026-11-12 against the pilot configuration, **including the new eDNA cruise windows**
(`config/edna_cruise_windows.yaml`). "Satellite-matched" means the observation's full 30-day causal
optical context lies inside the downloaded satellite windows. Counts are observation events
(sample groups, casts, survey events), not rows. Region boxes are the pilot regions in
`config/pilot.yaml`.

## Summary

| Benchmark | Source | Observations in pilot regions | Satellite-matched | Spatial coverage | Temporal coverage | Feasible grouped splits | Status |
|---|---|---|---|---|---|---|---|
| Microbial communities (16S, 18S): NMF components, dominant taxa, composition | SEUS MBON (local `eDNA_data_sources/processed/`) | 1,088 surface sample groups (shelf 888, Keys 639; boxes overlap); 915 from 2018 | **915 of 915** (2018+); 184 in 2024; 115 in the PACE era | SW Florida shelf and Keys; 0 offshore | 2015-06 – 2024-07; ~21 cruises 2018 – Jan 2024, plus 2024 | Station-grouped spatial folds; cruise-grouped time blocks; 2024 as a temporal hold-out (also the PACE-era test) | ✅ ready |
| Measured nutrients at eDNA samples (PO₄, NO₃, Si) | SFER, joined on `keyfield` | 553 eDNA groups with nutrients (2018+) | 553 | shelf, Keys | 2018 – 2024 | as above | ✅ ready |
| SFER nutrients and fluorometric chlorophyll, all samples (≤ 5 m) | SFER (`physiochem_data_sources/SFER_physicochem_clean_wide.csv`) | PO₄/Si 3,987, NO₃ 3,968, Chl 3,568 (63 cruises) | PO₄/Si 2,551, NO₃ 2,532, Chl 2,139 | shelf, Keys; 0 offshore | 2014 – 2025 | cruise-grouped time blocks; station-grouped spatial folds | ✅ ready (Chl kept as a separate target from HPLC) |
| Reef fish, visual census | NOAA CRCP Florida (local) | 5,216 survey events | 2,606 | Keys only | 2016 – 2024 | year blocks × Keys sub-region | ✅ ready |
| Reef fish, video MaxN | `multimodal_granite/fish/GF_maxn` (local) | 4,592 events (shelf 3,772, Keys 1,087) | 1,517 | shelf, Keys; median depth 38 m | 2016 – 2024 | year blocks × spatial blocks | ✅ ready |
| Inshore fish, FWC Fisheries-Independent Monitoring | FWC FIM (local) | 100,311 events (shelf 96,829, Keys 4,220) | 7,166 | mostly estuaries and bays (Tampa Bay, Charlotte Harbor …) | 1996 – 2023 | year blocks × estuary | ⚠️ most sites are in estuaries, where 1.2 km pixels are land-adjacent; usable only after a distance-to-shore filter |
| HPLC pigments | BCO-DMO 851250 (BLOOFINZ-GoM, R/V Nancy Foster) | 18 surface casts (≤ 10 m), 11 in the offshore box; 31 pigments incl. DV-Chl a, Zea, But/Hex-fuco, Perid, Fuco | **0**: May 2017 and May 2018 are outside the satellite windows | central-eastern Gulf (25.4–28.3°N, 90.2–87.3°W) | 2017-05, 2018-05 | too few for grouped splits | ❌ not feasible alone |
| HPLC pigments | SeaBASS | Gulf HPLC found in `LAMONT_GOM` (cruises JUL16GOM, JUL19gom) and Lohrenz NSF_Gulf_Rapid 2017 (northern Gulf); none found on the Florida shelf or Keys | not yet matched | to check | 2016, 2017, 2019 | – | ⏳ counts need a SeaBASS file search (see below) |
| Sargassum, same-period and 7-day forecast | USF SaWS `FA_UNET_DENSITY` (weekly) | **unknown**: no public numerical grids | – | – | – | – | ❌ blocked; see below |

The offshore Gulf box has no eDNA, SFER or fish observations. As agreed, it stays in pretraining
and is used for benchmarks that cover it (HPLC if enough is found; Sargassum in a separate
evaluation region).

## PACE-era coverage of the evaluation samples (2026-11-12)

PACE starts 2024-03-05. From `config/eval_points.csv` (all samples in the pilot regions, 2018+;
`scripts/make_eval_points.py`), by date only:

| Dataset | Samples 2018+ | By year (2018–2025) | In PACE era | Full 30-day PACE history | PACE-era cruises / blocks |
|---|---|---|---|---|---|
| eDNA 16S (18S the same) | 913 | 5 / 0 / 82 / 157 / 225 / 261 / 183 / 0 | 115 (all 2024) | 92 | 3 (Mar, May, Jul 2024) |
| SFER ≤ 5 m, nutrients or Chl | 3,174 | 301 / 298 / 257 / 438 / 422 / 482 / 511 / 465 | 860 (2024: 395, 2025: 465) | 797 | 11 |
| Fish, CRCP | 3,018 | 772 / 0 / 0 / 364 / 825 / 0 / 1,057 / 0 | 1,057 (all 2024) | 1,057 | 6 |
| Fish, video MaxN | 3,805 | 346 / 554 / 584 / 554 / 581 / 570 / 616 / 0 | 616 (all 2024) | 616 | 5 |

Counts of samples with a *usable* PACE pixel (valid under the training mask, in the sample's
cell or its 3 × 3 neighbourhood, within the 30 days before or ± 1 day) come from step 2.4b
(`tiles/v1/index/eval_coverage_summary.csv`) and will be added here. The year plan for PACE
(pretraining, spectral-retention test, eDNA 2024 hold-out) is in the cluster_instructions.md
decision log. MaxN event dates are all the 15th of a month, i.e. month precision only; same-day
matching is not meaningful for that dataset.

## Microbial communities (Florida metabarcoding)

- Restricted to the SW Florida shelf and Keys, where the samples are (your item 4).
- Inputs: the replicate-averaged, prevalence-filtered 16S (6,203 ASVs) and 18S (9,084 ASVs)
  tables, and the existing Hellinger NMF exports (k = 5–40, 16S and 18S). Targets: individual NMF
  components, dominant taxa, and overall composition (Hellinger distances / PCA).
- **Data issue:** `seusmbon_16S_metadata.tsv` and `seusmbon_18S_metadata.tsv` have
  `decimalLatitude` and `decimalLongitude` swapped. `scripts/make_edna_windows.py` detects and
  corrects it; any other code reading those files must too.
- Splits must be fixed before any training, grouped by station (replicates and repeat visits
  share conditions) and by cruise. Feature selection uses training folds only.

## Sargassum

What exists (checked 2026-11-12):

| Source | What it is | Resolution | Period | Numerical, georeferenced? | Fit for tile-level labels |
|---|---|---|---|---|---|
| SaWS `FA_UNET_DENSITY` ("UNET_DENSITY 7DAY MEAN", shown as GCOOS `7DAY.FAD`) | U-Net-classified Sargassum density over the past week, MODIS | 1 km (0.009091°) | – | **No public grids.** The public browser shows colour-mapped PNGs (scale 0–0.4) with world files, so they are georeferenced, but the values are colours. The files sit under a `subscription` path; no download listing. | Only via a data request to USF |
| NOAA AOML ERDDAP `noaa_aoml_atlantic_oceanwatch_AFAI_1D/3D/7D` | USF Alternative Floating Algae Index | 0.015° (~1.5 km), 0–38°N, 98–38°W | 2016-06 – present | Yes (ERDDAP griddap, netCDF) | It is the *index*, not classified Sargassum: also responds to cloud edges, other floating material and blooms. Turning it into labels needs USF's background removal and classification, which we would be approximating |
| NCEI 0190272 | Monthly Sargassum coverage and density from MODIS | 0.5° | density grids 2001–2010; Caribbean/Central Atlantic text series and July maps to 2018 | Yes (netCDF/GeoTIFF/ASCII) | No: nothing gridded for the Gulf in our 2018–2026 windows, and 0.5° (~55 km) is larger than a tile |
| USF Digital Commons "Monthly Sargassum Wet Biomass … Western North Atlantic" (CC BY 4.0) | Monthly biomass per sub-region, MODIS, 2000-03 – 2024-02 | 0.5° internally; published as sub-region totals | 2000 – 2024 | Sub-region series (Sargasso Sea, Gulf Stream, Antilles Current, Atlantic belt, **NW Gulf**); whether gridded files are attached is unconfirmed | Climatology or context only; the NW Gulf sub-region is outside our boxes |

Conclusions:

1. **The positive-count check by region, month and year cannot be done yet**, because no public
   source gives weekly classified Sargassum for 2018–2026 as numbers. Per your instruction, labels
   are not built from the PNGs or approximated from AFAI.
2. **Recommended route:** ask the USF Optical Oceanography Lab (Chuanmin Hu) for the weekly
   `FA_UNET_DENSITY` grids, or the daily classified maps, as netCDF/GeoTIFF over the Gulf and
   Florida Straits for 2018–2026. The AOML AFAI grids can serve as an independent sanity check
   once we have them, not as labels.
3. When labels exist: count positive weeks per region, month and year. If the pilot boxes rarely
   see mats, add a **separate Sargassum evaluation region** (e.g. eastern Gulf / Florida Straits
   where mats are regularly observed) rather than enlarging the offshore control box. That region
   would also need satellite and physical downloads.
4. Design for the forecast (priority over detection): inputs end before the target week starts,
   with no overlap. Baselines: persistence, seasonal climatology, and simple advection of the last
   map by Copernicus Marine surface currents (GlobCurrent, already downloaded). SaWS stays out of
   pretraining. Caveat: SaWS is derived from MODIS reflectance, which the encoder also sees, so
   same-period detection partly measures retention of the floating-algae signal.

## HPLC pigments

- **BCO-DMO 851250** (downloaded and checked): 115 samples from 18 CTD casts. 23 samples from the
  18 casts are within 10 m of the surface; 11 of those casts are in the offshore box. All are from
  May 2017 and May 2018, outside our satellite windows. Covering both cruises with a 30-day context
  would need about 90 more satellite days (~80 GB) plus physical lead-in for 2017.
- **SeaBASS**: has no documented search API, and its file search form is built in JavaScript.
  The Gulf experiments with HPLC found so far are `LAMONT_GOM` (JUL16GOM, JUL19gom) and the
  Lohrenz NSF_Gulf_Rapid 2017 cruises (northern Gulf, probably north of the domain). None were
  found on the Florida shelf or Keys.
  **To finish the count:** run the SeaBASS file search
  (https://seabass.gsfc.nasa.gov/search) with measurement type *pigment*, area 22.8–29.3°N,
  92–78.8°W, 2017 onward, depth 0–10 m, and save the result list (or paste the cruise list).
- **Assessment:** in the Gulf/Florida pilot area, HPLC looks too sparse for grouped train/test
  splits. Keep it optional, possibly as a small cross-check pooled with SeaBASS. SFER fluorometric
  chlorophyll (2,139 satellite-matched samples) remains the main phytoplankton target, kept
  separate from HPLC.

## Sources

- SaWS: https://optics.marine.usf.edu/projects/SaWS.html
- AOML AFAI: https://cwcgom.aoml.noaa.gov/erddap/griddap/noaa_aoml_atlantic_oceanwatch_AFAI_7D
- NCEI 0190272: https://www.ncei.noaa.gov/archive/accession/0190272 (files under
  https://www.ncei.noaa.gov/data/oceans/archive/arc0139/0190272/2.2/data/0-data/)
- USF biomass: https://digitalcommonsdata.usf.edu/datasets/zcyd5wvncc
- BCO-DMO 851250: https://www.bco-dmo.org/dataset/851250 (doi:10.26008/1912/bco-dmo.851250.1)
- SeaBASS LAMONT_GOM: https://seabass.gsfc.nasa.gov/experiment/LAMONT_GOM
- Lohrenz 2017 HPLC: https://bco-dmo.org/dataset/789061 and https://ncei.noaa.gov/archive/accession/0277900
