# ocean_ecology_foundation_model

A multisensor ocean foundation model: MODIS, VIIRS, Sentinel-3 OLCI and PACE OCI reflectance,
plus ocean physics and atmospheric forcing, for plankton, eDNA, HAB and Sargassum applications.

**Running it on Ptolemy: start with [cluster_instructions.md](cluster_instructions.md)** (order of
operations, status of each step, decision log, and what to paste back for review).

**Phase 1 (now):** acquire and check a pilot dataset over Florida and the Gulf of Mexico. There is
no training code and no architecture-dependent preprocessing yet. The brief is in
[docs/phase1_instructions.md](docs/phase1_instructions.md), and archive findings are in
[docs/dataset_inventory.md](docs/dataset_inventory.md).

Compute runs on **Ptolemy** (HPC2, 4 × A100 per job). See [Running on Ptolemy](#running-on-ptolemy).

## Layout

```
config/pilot.yaml      regions, time windows, products (the only thing the pipeline reads)
oefm/                  python -m oefm <command>
  config.py            regions (areas), window merging
  sources/nasa.py      earthaccess search/inventory (OB.DAAC, PO.DAAC, GES DISC)
  sources/cmems.py     Copernicus Marine monthly subsets (MY first, NRT after)
  sources/http.py      GEBCO
  manifest.py          SQLite manifest (+ CSV export), transfers and runs tables
  download.py          resumable, retrying, verifying downloads
  inspect.py           file structure, wavelengths, flags, mask scenarios, maps
  index.py             observation index, coincidences, sensor matchups
  report.py            pilot acquisition report (markdown)
cluster/               Ptolemy SLURM jobs
docs/                  brief, archive inventory, readiness notes
tests/                 python -m unittest discover -s tests
```

Data are written to `data/` next to the repo on scratch (gitignored):
`data/raw/<product>/<YYYY>/<MM>/<original file>`, `data/manifest/manifest.sqlite` (+ `manifest.csv`),
`data/index/`, `data/reports/`.

## Commands

| command | what it does | internet |
|---|---|---|
| `regions` | bounding boxes, sizes, areas, overlaps | no |
| `windows` | merged satellite windows and physical windows (with 90-day lead) | no |
| `catalog` | collection versions and time ranges at the archives → `reports/catalog.json` | yes, no login |
| `inventory` | dry run: search every product/region/window, fill the manifest, print counts, GB and ETA | yes, no login (CMEMS size estimate needs login) |
| `download` | fetch planned files (`--per-product N`, `--products`, `--start/--end`, `--region`) | yes |
| `verify` | re-check files on disk (size, HDF5/netCDF signature, MD5 where published) | no |
| `inspect` | per-file structure JSON, regional QC stats, mask scenarios, maps | no |
| `index` | `data/index/observations.csv` (+ parquet), counts, multisensor coincidences | no |
| `overlap` | PACE vs OLCI/MODIS/VIIRS matchups on a temporary 0.02° grid | no |
| `status` / `report` | manifest summary / `data/reports/pilot_acquisition_report.md` | no |

How resumption works. The manifest has one row per file, keyed by `<product>:<filename>`, so a
swath that crosses two regions is downloaded once. Files are written to `.part`, resumed with
HTTP Range, and renamed only after verification. Verified files are never fetched again. A
re-run inventory adds new files and keeps download state.

## Running on Ptolemy

The account is `191001-366746` (not Morrill's `191001-364393`). It goes on the `sbatch` command
line and is not written into the scripts. Jobs that need internet use `service`. Every job
requests 48 h. Run `sbatch` from the repo root, because the scripts locate the repo through
`$SLURM_SUBMIT_DIR`.

```bash
# once, on a Ptolemy login node
mkdir -p /scratch/ptolemy/users/$USER && cd /scratch/ptolemy/users/$USER
git clone https://github.com/helloftroy/ocean_ecology_foundation_model.git
cd ocean_ecology_foundation_model
cp credentials.sh.example credentials.sh && chmod 600 credentials.sh   # fill in Earthdata + Copernicus Marine
sbatch --account=191001-366746 cluster/setup_env.sbatch                # Miniforge + venv on scratch (~15 min)

# 1. end-to-end test, one week of every product (own data root: data_smoke/)
sbatch --account=191001-366746 cluster/smoke_test.sbatch
#    -> logs/oefm_smoke_<id>.out, data_smoke/reports/pilot_acquisition_report.md

# 2. full dry-run inventory (no downloads): counts, GB, ETA
sbatch --account=191001-366746 cluster/inventory.sbatch

# 3. download, 2024-2026 multisensor first, then physical, then history
sbatch --account=191001-366746 --export=ALL,START=2024-01-01,CHAIN=3 cluster/download.sbatch
sbatch --account=191001-366746 --export=ALL,PRODUCTS=physical cluster/download.sbatch
sbatch --account=191001-366746 --export=ALL,END=2023-12-31,CHAIN=3 cluster/download.sbatch

# 4. QC, observation index, matchups, HISATZEN diagnostic, report (any time; reruns only process new files)
sbatch --account=191001-366746 cluster/qc.sbatch
```

Run only one `download.sbatch` at a time, because there is a single SQLite manifest writer.
`CHAIN=N` re-queues the job up to N times past the 48 h limit. To check progress, run
`source cluster/env.sh && python -m oefm status` on a login node.

Earthdata needs these apps approved in the profile: *OB.DAAC Data Access*, *PO.DAAC Drive*, and
*NASA GESDISC DATA ARCHIVE* (for IMERG).
