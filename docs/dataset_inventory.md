# Dataset inventory and first checks (2026-11-11)

These results come from running the pipeline on a laptop on 2026-11-11, before the Ptolemy runs:

- `catalog`;
- a full dry-run `inventory` with no downloads (CMR searches and the Copernicus Marine catalogue);
- a one-day end-to-end smoke test (2025-06-10) that downloaded, verified, inspected, indexed and
  matched one or more real files of every NASA product.

Ptolemy runs replace the throughput figures. The other numbers will change only if the archives do.

## Products, versions and archive coverage

| key | archive name | version | archive | origin | archive coverage (checked 2026-11-11) |
|---|---|---|---|---|---|
| modisa_oc | MODISA_L2_OC | R2022.0 | OB.DAAC (OB_CLOUD) | observed | 2002-07-04 .. **2026-05-31** (Aqua ended) |
| viirsn_oc | VIIRSN_L2_OC | R2022.0 | OB.DAAC | observed | 2012-01-02 .. 2026-10-01 |
| viirsj1_oc | VIIRSJ1_L2_OC | R2022.0 | OB.DAAC | observed | 2017-11-29 .. 2026-10-01 |
| viirsj2_oc | VIIRSJ2_L2_OC | **2025.0** | OB.DAAC | observed | 2022-11-10 .. 2026-10-01 (first pilot-window file 2023-06-01) |
| olcia_efr_oc | OLCIS3A_L2_EFR_OC | R2022.0 | OB.DAAC | observed | 2016-04-05 .. **2026-04-14** |
| olcib_efr_oc | OLCIS3B_L2_EFR_OC | R2022.0 | OB.DAAC | observed | 2018-04-25 .. ~2026-03 (sparse Apr-Jul 2025) |
| pace_oci_aop | PACE_OCI_L2_AOP | **3.2** | OB.DAAC | observed | 2024-03-05 .. 2026-10-01 |
| pace_oci_sfrefl | PACE_OCI_L2_SFREFL | **3.1** | OB.DAAC | observed | 2024-03-05 .. 2026-10-01 |
| mur_sst | MUR-JPL-L4-GLOB-v4.1 | 4.1 | PO.DAAC | obs.-based analysis | 2002-05-31 .. 2026-11-09 |
| ccmp_wind | CCMP_WINDS_10M6HR_L4_V3.1 | 3.1 | PO.DAAC | obs.-based analysis (ERA5 background) | 1993 .. 2026-09-30 |
| imerg_final | GPM_3IMERGDF | V07 | GES DISC | obs.-based analysis (gauge-calibrated) | 1998 .. **2025-09-30** |
| imerg_late | GPM_3IMERGDL | V07 | GES DISC | obs.-based analysis (satellite only) | used 2025-10-01 .. 2026-11-10 |
| cmems_sealevel | SEALEVEL_GLO_PHY_L4_MY_008_047 → NRT_008_046 | MY 202411 / NRT 202506 | Copernicus Marine | obs.-based analysis | MY 1993 .. 2026-05-15, NRT 2024-07-01 .. now |
| cmems_globcurrent | MULTIOBS_GLO_PHY_MYNRT_015_003 | 202411 | Copernicus Marine | obs.-based analysis (Ekman fit uses ERA5 stress) | MY 1993 .. 2026-03-31, NRT 2022-05 .. now |
| gebco | GEBCO_2026 grid (CEDA) | 2026 | BODC/CEDA | obs.-based compilation | static, 7.47 GB |

Deferred, identified but not downloaded:

- **ARMOR3D** MLD/T/S (`MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012`): observation-based. MY runs to
  2025-06-30, NRT from 2024-09-08.
- **GLORYS12** MLD (`GLOBAL_MULTIYEAR_PHY_001_030`): model reanalysis.
- **Multi-obs SSS** (`MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013`).

Excluded from pretraining inputs: modelled chlorophyll, primary production and nutrients.

## Regions

| region | bbox [W, S, E, N] | W-E × S-N km | area km² |
|---|---|---|---|
| swfl_shelf | [-84.5, 24.8, -80.9, 28.3] | 358 × 389 | 139,340 |
| florida_keys | [-83.3, 23.8, -79.8, 25.95] | 353 × 239 | 84,405 |
| offshore_gulf | [-91.0, 24.0, -86.5, 27.0] | 452 × 334 | 150,641 |
| domain (union + 1°, physical subsets) | [-92.0, 22.8, -78.8, 29.3] | 1319 × 723 | 952,578 |

swfl_shelf and florida_keys share [-83.3, 24.8, -80.9, 25.95] (30,833 km²). Swaths covering both
are stored once.

## Time windows

- **Satellite:** 18 merged windows, 2,005 days. These are three 45-day seasonal windows per year
  for 2018-2023 (winter Feb 1-Mar 17, summer Jun 1-Jul 15, fall Sep 15-Oct 29), extended for the
  2018 red-tide peak, Piney Point 2021 and Idalia 2023. Core is 2024-01-01 to today.
- **Physical (90-day lead):** 6 windows, 3,276 days. These are almost continuous from 2017-11-03.

## Dry-run inventory (no downloads)

| product | files | raw GB 2024+ | raw GB 2018-23 | raw GB | regional GB (upper bound) |
|---|---|---|---|---|---|
| modisa_oc | 3,275 | 42 | 68 | 110 | 49 |
| viirsn_oc | 3,928 | 152 | 177 | 329 | 69 |
| viirsj1_oc | 3,945 | 161 | 188 | 348 | 69 |
| viirsj2_oc | 2,262 | 156 | 16 | 172 | 40 |
| olcia_efr_oc | 1,735 | 176 | 246 | 422 | 281 |
| olcib_efr_oc | 1,646 | 172 | 237 | 409 | 274 |
| pace_oci_aop | 1,776 | 381 | 0 | 381 | 252 |
| pace_oci_sfrefl | 1,776 | 1,350 | 0 | 1,350 | 180 |
| mur_sst | 3,280 | 845 | 1,381 | 2,225 | 4 |
| ccmp_wind | 3,234 | 37 | 72 | 108 | 0.2 |
| imerg_final + late | 3,275 | 33 | 63 | 97 | 0.2 |
| cmems (2 products) | 219 | 0.4 | 0.8 | 1.1 | 1.1 |
| gebco | 1 | 7.5 | 0 | 7.5 | <0.1 |
| **total** | | **3,511** | **2,449** | **5,960** | **~1,220** |

Satellites total 3.5 TB and physical products 2.4 TB. MUR is 2.2 TB because its global daily files are
kept whole; that is less than 1% of Ptolemy's free scratch. The aggregate rate measured on the
laptop (6 connections) was 37 MB/s, which would take about 45 h for the whole pilot. Ptolemy's
rate replaces this after the smoke test.

## Smoke test on real files (2025-06-10)

Download: 17 files, 4.2 GB, all verified. MUR was also verified against its CMR MD5. One MODIS
file's CMR size was stale: CMR listed it twice, and one record held a different size. The served
Content-Length is now treated as authoritative and the difference recorded in `size_note`.

| product | variable / bands | wavelengths nm | pixel spacing at nadir (scan × track km) |
|---|---|---|---|
| MODIS Aqua | `Rrs_412 … Rrs_678`, 10 bands, 2-D each | 412-678 | 0.98 × 0.98 |
| VIIRS SNPP / N20 / N21 | `Rrs_<nm>`, 5 bands | 410/411-671/667 | 0.77 × 0.74 |
| OLCI-A EFR | `Rrs_<nm>`, 11 bands | 400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709 | **0.32 × 0.29** (300 m confirmed) |
| PACE AOP | `Rrs(lines, pixels, wavelength_3d)`, 172 bands | 346-719 | 1.13 × 1.21 |
| PACE SFREFL | `rhos(lines, pixels, wavelength_3d)`, 122 bands | 346-2258 (68 bands in 600-900 nm, SWIR 1038, 1249, 1618, 2131, 2258) | 1.13 × 1.21 |

All files share one structure: `geophysical_data`, `navigation_data/{latitude,longitude}`,
`sensor_band_parameters`, and `l2_flags` with `flag_meanings` / `flag_masks`. Rrs is in sr⁻¹;
rhos has no units.

PACE SFREFL has bands at or very near every common floating-algae index wavelength: 645, 667,
709, 747, 860, 870, 1249, 1618 and 2131 nm. For FAI/AFAI-type baselines the red/NIR coverage is
complete. The SWIR 1240/1640 bands are offset by +9 and -22 nm.

What the flags showed for this one day (one granule each; not yet representative):

- **Satellite zenith.** PACE and NOAA-21 caught the Keys and shelf near their swath edges, so
  HISATZEN flags 99-100% of pixels. The default L3 mask then keeps about 0%, while the minimal
  mask (LAND/CLDICE/ATMFAIL/NAVFAIL/HILT) keeps about 50%. The mask policy for swath edges matters
  more than any other choice.
- **OLCI navigation.** OLCI-A shows a strip about 0.5° wide flagged NAVFAIL along its western
  swath edge, which covered most of the shelf box that day.
- **Common flags.** PRODFAIL is set on 36-100% of pixels and COASTZ / OPSHAL on 20-40% in the
  coastal boxes. PRODFAIL concerns derived products (chlorophyll etc.), not Rrs, and is not in any
  candidate mask.
- **PACE red/NIR noise.** In clear water, 10-35% of PACE Rrs values at 689-703 nm are negative,
  which is expected noise at low red/NIR signal. Keep them (do not clip) and decide during
  harmonisation.

PACE vs other sensors on the same day, binned to 0.02° and compared under the minimal mask
(r / median ratio other/PACE):

| pair (Keys) | Δt | 443 | 490 | 555/560 | 665 |
|---|---|---|---|---|---|
| PACE vs VIIRS SNPP | 0.4 h | 0.91 / 0.92 | 0.97 / 0.96 | 0.98 / 0.97 | 0.92 / 0.43 |
| PACE vs MODIS Aqua | 2.6 h | 0.74 / 0.93 | 0.93 / 0.98 | 0.96 / 0.85 | 0.84 / 0.63 |
| PACE vs OLCI-A | 2.3 h | 0.74 / 0.97 | 0.92 / 0.96 | 0.94 / 0.90 | 0.81 / 0.63 |
| PACE vs VIIRS N21 | 1.7 h | 0.68 / 0.89 | 0.91 / 0.93 | 0.94 / 0.86 | 0.55 / 0.69 |

The sensors match geographically and temporally. Blue-green bands agree within about 10-15%.
The red band differs systematically, because the signal there is near zero in clear water,
which is where spectral harmonisation will matter most.
