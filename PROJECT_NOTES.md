# Project notes (state for the next session)

## Where things are
- **cluster_instructions.md**: the user's run order, step status, decision log (with evidence)
  and paste-back list. Update it whenever steps, decisions or results change.
- Briefs: docs/phase1_instructions.md, docs/phase2_instructions.md (Phase 2 received 2026-11-12;
  its storage revision overrides Obj. 1: keep global MUR in raw/, crops in processed/regional/).
- Repo: https://github.com/helloftroy/ocean_ecology_foundation_model (local:
  ~/multimodal_seusmbol/ocean_ecology_foundation_model). Clone on Ptolemy into
  /scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model.
- Ptolemy: SLURM account `191001-366746` (on the command line, never in #SBATCH headers).
  Internet only on `service`. GPUs on `gpu-a100` (QOS: at most 4 A100s, 128 CPUs, 1 TB, 48 h per
  user). There is no CPU-only compute partition. Every job uses `--time=48:00:00`.
  Cluster facts come from biogoship_granite_kegg/cluster/ptolemy/README.md (check run 2026-10-30).
- An agent cannot SSH in (password + Duo). The workflow is: push, the user pulls and submits, the
  user pastes logs back.

## Status (2026-11-12)
- Ptolemy: env, smoke test (23/23 verified, incl. Copernicus Marine and GEBCO), inventory and the
  satellite downloads are done. All six MODIS/VIIRS/OLCI products are verified (1.79 TB, 0 failed,
  2018-2026). The first inventory stopped before PACE; the user fixed and re-ran it, so PACE and the
  physical products are now in the manifest but not yet downloaded.
- Measured on ptolemy-dtn-1: ~200 MB/s aggregate with 8 connections (NASA ~25 MB/s each).
- Fixed: thread-unsafe manifest writes (job 116018 crashed at 9103/9129), throughput accounting.
- New: `python -m oefm zenith` (HISATZEN diagnostic), run by qc.sbatch after `overlap`.

- (2026-11-12) Steps 1.4, 1.4b, 1.5 and 1.6 done: all 15 products downloaded (6.44 TB) except 8
  MUR files that failed md5 (step 1.7, re-fetch). QC reviewed; HISATZEN settled per sensor (see
  Decisions). The user's QC paste-back is in `onepoint5/` (untracked).

- (2026-11-12) Phase 2 steps 2.0-2.4b built and tested on the local smoke day:
  oefm/provenance.py (layout + processing.sqlite), oefm/qcpolicy.py (config processing.qc),
  oefm/crop.py (physical crops, values identical to raw), oefm/grid.py (spherical LAEA, 272 tiles,
  IDs X+iiY+jj), oefm/tiling.py (area-weighted footprint regridding, one zipped Zarr per granule,
  index parquet), oefm/examples.py, oefm/evalcoverage.py (usable PACE per evaluation sample);
  cluster/prep.sbatch runs them. OLCI EFR geolocation repeats each lon over 2-3 columns:
  tiling.dequantise() fixes it (else 300 m stripes). zarr pinned <3 (setup_env rerun needed).
- PACE era: all PACE-era eDNA/fish samples are 2024 (eDNA 115, 92 with full 30-d PACE history);
  SFER has 2024+2025. Year plan APPROVED 2026-11-12 (config training.partitions): pretrain on
  2018-2025 imagery incl. 2024 PACE (hold-out is for labels only); PACE retention train
  2024-03..2025-12 / val 2026-01..02 / gap 2026-03 / test 2026-04..10, plus geo-blocked test
  (4x4-tile blocks, 20%, 1-tile buffer); eDNA+fish labels 2024 = test; SFER labels 2025 = test
  (primary PACE on/off); NMF etc. refitted on training labels only (2.7). MaxN dates are
  month precision (all on the 15th).
- 2.5 built: oefm/physics.py (physics cube), oefm/loader.py (TileSequences, partitions, masking,
  sequence_for_point), `sequences` summary; tests/test_loader.py (leakage checks).

## Next steps
1. Step 1.7: re-fetch the 8 MUR files (download.sbatch; if they fail again, re-inventory mur_sst).
2. P.0 setup_env rerun, then prep.sbatch trial (LIMIT=40), then full; review examples and
   eval_coverage_summary.csv (usable PACE per evaluation sample).
3. 2.6/2.7 built (2026-11-12): config `targets.pisces_nutrients` (cmems_mod_glo_bgc_my_0.25deg_P1D-m
   v202406 to 2026-08-31, then cmems_mod_glo_bgc-nut_anfc_0.25deg_P1D-m v202311; surface; verified
   in the catalogue), oefm/targets.py (pisces_daily.zarr, log+z from training partition; evalsets
   join), cluster/targets.sbatch, scripts/make_eval_targets.py -> ../eval_targets/ (outside repo;
   NMF k 5/10/15/20 refitted on train labels, ASV prevalence on train, PCA, genera incl. PR2 18S).
   The user's local PISCES file only reaches 87.5 W, so a fresh download is needed.
4. 2.8 waits on 4 answers (cluster_instructions.md under 2-II): GPU env, SIGReg reference,
   Stage 1 budget, headline NMF k.
4. Update docs/dataset_inventory.md and docs/readiness_notes.md with full-pilot numbers.
4. Phase 2 (docs/phase2_instructions.md, incl. the physics/nutrient revision and the user's three
   adjustments): planned order and decision log in cluster_instructions.md. Stage 1 = optical
   A/B/C, Stage 2 = physics / PISCES-nutrient targets / both; evaluations = Florida SEUS MBON
   16S/18S (NMF components, taxa, composition, SFER nutrients), SaWS Sargassum, HPLC, fish.
5. eDNA cruise windows approved and added (config/edna_cruise_windows.yaml via
   scripts/make_edna_windows.py; +487 satellite days); the user runs inventory + download as step
   1.4b. NB the SEUS MBON metadata TSVs have decimalLatitude/decimalLongitude swapped.
6. Evaluation inventory in docs/evaluation_inventory.md. Sargassum is blocked: no public numerical
   SaWS FA_UNET_DENSITY; ask USF (Hu lab). HPLC is sparse; SeaBASS manual search pending. Neither
   may delay Stage 1.

## Decisions
- Copernicus Marine products are stored as monthly domain subsets with all variables (the service
  is ARCO; there are no "original files"). NASA products are stored as original granules.
- The manifest is SQLite on GPFS with DELETE journalling: run one download job at a time.
- Diagnostic grids (0.05° coverage, 0.02° matchups) are for QC only; no training grid is chosen.
- (2026-11-12) **OLCI stops at OB.DAAC's end (2026-04).** No EUMETSAT/CDSE OL_2_WFR backfill.
- (2026-11-12) **MUR stays as whole global daily files** (2.2 TB) so global training needs no
  re-download. Regional crops will be made soon, as derived files; the originals are kept.
- (2026-11-12) **HISATZEN is a lower-confidence tag, not a hard mask.** Pilot mask
  `l3_no_hisatzen` = L3 default minus HISATZEN; ATMFAIL, NAVFAIL, HILT, cloud, land, missing
  reflectance etc. still apply. The flag and the reconstructed sensor zenith are kept per
  observation (index columns `hisatzen_fraction`, `senz_median`, `senz_max`) so high-angle data
  can be excluded from strict spectral-validation subsets or down-weighted in training. No
  arbitrary new angle threshold; revisit per sensor once `oefm zenith` has run on the full data.
  Sensor zenith is not in the L2 files; it is reconstructed from geolocation + orbit altitude
  (OLCI: nadir = minimum-spacing column) and validated against the flag: the HISATZEN edge falls
  at 59.4-60.2 deg on the smoke files, and l2gen satzen = 60, masksatzen = 0 in every file.
- (2026-11-12, full QC run) **HISATZEN per sensor:** relaxed (tag only) for PACE and the three
  VIIRS, which pass the rule (decode edge 59.8-60.2 deg, >= 0.996 agreement; 412-555 nm within
  +/-10% of low angle at 60-90 deg with 31-45 pairs). **MODIS keeps HISATZEN in the hard mask**
  (412 nm 1.11 at 60-65 deg; relaxing gains only 1-7% of pixels). OLCI never exceeds 57 deg.
  665 nm is angle-dependent from ~45 deg (SNPP 0.78, NOAA-21 0.81, OLCI-B 0.73, PACE 1.20 at
  45-60 deg) -> per-sensor angle weights from 45 deg in step 2.1. Strict spectral subset uses
  l3_default for all sensors. Table in cluster_instructions.md decision log.
