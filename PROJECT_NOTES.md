# Project notes (state for the next session)

## Where things are
- Repo: https://github.com/helloftroy/ocean_ecology_foundation_model (local:
  ~/multimodal_seusmbol/ocean_ecology_foundation_model). Clone on Ptolemy into
  /scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model.
- Ptolemy: SLURM account `191001-366746` (on the command line, never in #SBATCH headers).
  Internet only on `service`. GPUs on `gpu-a100` (QOS: at most 4 A100s, 128 CPUs, 1 TB, 48 h per
  user). There is no CPU-only compute partition. Every job uses `--time=48:00:00`.
  Cluster facts come from biogoship_granite_kegg/cluster/ptolemy/README.md (check run 2026-10-30).
- An agent cannot SSH in (password + Duo). The workflow is: push, the user pulls and submits, the
  user pastes logs back.

## Status (2026-11-11)
- Phase 1 pipeline written and tested locally: 13 unit tests, a full dry-run inventory (5.96 TB),
  and a one-day real-data smoke test of every NASA product (download → verify → inspect → index
  → matchup → report). Results are in docs/dataset_inventory.md.
- Not yet run on Ptolemy: setup_env, smoke_test, inventory, downloads, Copernicus Marine downloads
  (need COPERNICUSMARINE_SERVICE_* credentials), GEBCO.

## Next steps
1. Ptolemy: setup_env.sbatch → smoke_test.sbatch. Paste back logs/oefm_smoke_*.out and
   data_smoke/reports/pilot_acquisition_report.md.
2. inventory.sbatch, then download.sbatch with START=2024-01-01 (core first), then physical, then
   history.
3. qc.sbatch once a few weeks of core data are on disk; review mask scenarios and matchups.
4. Decide: OLCI after 2026-04 (EUMETSAT/CDSE OL_2_WFR?), HISATZEN policy, whether MUR stays as
   whole global files (2.2 TB) or is subset.

## Decisions
- Copernicus Marine products are stored as monthly domain subsets with all variables (the service
  is ARCO; there are no "original files"). NASA products are stored as original granules.
- The manifest is SQLite on GPFS with DELETE journalling: run one download job at a time.
- Diagnostic grids (0.05° coverage, 0.02° matchups) are for QC only; no training grid is chosen.
