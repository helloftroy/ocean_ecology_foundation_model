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

## Next steps
1. download.sbatch without START (PACE, MUR, CCMP, IMERG, CMEMS, GEBCO; ~4.2 TB). One job at a time.
2. qc.sbatch once status shows 0 failed. Review report section 4 (incl. the HISATZEN diagnostic:
   decode check, coverage recovered, angle effect from same-day matchups) and data/reports/zenith/.
3. Update docs/dataset_inventory.md and docs/readiness_notes.md with full-pilot numbers.
4. Phase 2 (docs/phase2_instructions.md, incl. the physics/nutrient revision and the user's three
   adjustments): planned order and decision log in cluster_instructions.md. Stage 1 = optical
   A/B/C, Stage 2 = physics / PISCES-nutrient targets / both; evaluations = Florida SEUS MBON
   16S/18S (NMF components, taxa, composition, SFER nutrients), SaWS Sargassum, HPLC, fish.
5. Awaiting the user's OK: satellite windows around the eDNA cruises 2018-2023 (+467 days,
   ~420 GB). Only 119 of 731 2018-23 surface samples have a full 30-day context now.
   NB the SEUS MBON metadata TSVs have decimalLatitude/decimalLongitude swapped.

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
