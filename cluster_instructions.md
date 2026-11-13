# Cluster instructions: order of operations on Ptolemy

This is the single place to check what to run next, what has already run, and what to paste
back for review. It is updated with every change to the steps. Last updated: **2026-11-12** (Phase 2 steps 2.0–2.4 built and tested; PACE-era evaluation counts and the proposed PACE years).

- Briefs: [docs/phase1_instructions.md](docs/phase1_instructions.md),
  [docs/phase2_instructions.md](docs/phase2_instructions.md)
- Details of each command: [README.md](README.md); project state: [PROJECT_NOTES.md](PROJECT_NOTES.md)

Status key: ✅ done · ▶️ next · ⏳ waiting on an earlier step · 🛠 not built yet (scripts still to write)

---

## Rules for every step

1. **`git pull` before every `sbatch`.** A queued job runs whatever code is in the repo when it
   *starts*, so pull before it starts.
2. **Pass the account on the command line:** `sbatch --account=191001-366746 ...`. It is not in
   the script headers.
3. **Only one download job at a time.** Run `squeue -u hmp278` first. Several jobs writing the one
   SQLite manifest on GPFS can corrupt it. `scancel` any leftover `CHAIN` follow-on before
   submitting a new download.
4. Work from `/scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model`. Never write to `$HOME`.
5. Every job asks for 48 h. Downloads and QC run on the `service` partition (the only one with
   internet; there is no CPU-only partition). Training will use `gpu-a100`.
6. Logs: `logs/<job name>_<job id>.out` and `.err` (e.g. `logs/oefm_download_116018.out`).
7. Progress at any time, on a login node:
   `source cluster/env.sh && python -m oefm status`
8. `credentials.sh` holds the Earthdata and Copernicus Marine logins. Never commit it, and never
   paste it into a chat.

**If something stops part-way:** every step is resumable. Downloads restart unfinished files from
their `.part` files. Files left in status `downloading` by a crashed job are retried on the next
run. Verified files are never fetched again. Resubmit the same command.

---

## Phase 1: pilot data acquisition

| # | Step | Status |
|---|---|---|
| 1.0 | Build the Python environment on scratch | ✅ |
| 1.1 | Smoke test: one day of every product | ✅ 23/23 verified |
| 1.2 | Inventory: search the archives, fill the manifest | ✅ (first run stopped before PACE; re-run fixed it) |
| 1.3 | Download MODIS, VIIRS, OLCI (2018–2026) | ✅ 1.79 TB, 0 failed |
| 1.4 | Download PACE, MUR, CCMP, IMERG, Copernicus Marine, GEBCO | ✅ except 8 MUR files (md5 mismatch), see 1.7 |
| 1.4b | Satellite windows around the Florida eDNA cruises: inventory, then download (+487 satellite days, ~0.45 TB) | ✅ 21 windows loaded, all files verified |
| 1.5 | QC, observation index, matchups, HISATZEN diagnostic, report | ✅ |
| 1.6 | Paste back the QC results for review | ✅ reviewed 2026-11-12 (see the HISATZEN decision) |
| 1.7 | Re-fetch the 8 MUR files that failed md5 verification | ▶️ |

### 1.0 Environment ✅
```bash
sbatch --account=191001-366746 cluster/setup_env.sbatch
```
Re-run only if `requirements.txt` changes (the notes for that step will say so).

### 1.1 Smoke test ✅
```bash
sbatch --account=191001-366746 cluster/smoke_test.sbatch
```
Writes to `data_smoke/` (a test copy). You can delete `data_smoke/` once 1.5 has run on the real data.

### 1.2 Inventory ✅
```bash
sbatch --account=191001-366746 cluster/inventory.sbatch
# only some products:
sbatch --account=191001-366746 --export=ALL,PRODUCTS="pace_oci_aop mur_sst" cluster/inventory.sbatch
```
Downloads nothing; it can be re-run safely and keeps all download state. **Check:**
`python -m oefm status` lists all 15 products (MODIS, 3 VIIRS, 2 OLCI, PACE AOP and SFREFL, MUR,
CCMP, IMERG Final and Late, 2 Copernicus Marine, GEBCO) with TOTAL ≈ 5,960 GB expected.

### 1.3 Satellite downloads ✅
Done with `START=2024-01-01` and then without it. Nothing to re-run.

### 1.4 Remaining downloads ✅
```bash
git pull
squeue -u hmp278                      # nothing else downloading?
sbatch --account=191001-366746 --export=ALL,CHAIN=2 cluster/download.sbatch
```
- No `START`/`END`, so it fetches every file not yet verified.
- Expect roughly 6 h at the ~200 MB/s measured last time. MUR (2.2 TB) and PACE SFREFL (1.35 TB)
  are most of it.
- `CHAIN=2` re-queues the job up to twice if it reaches 48 h. A follow-on that finds nothing left
  ends within minutes.
- If many files fail with HTTP 429/503 (throttling), resubmit with `CONNECTIONS=4`.

**Done when** `python -m oefm status` shows every product with `verified` = `files` and
`failed` = 0, about 5.96 TB on disk.

### 1.4b eDNA cruise windows ✅ (approved 2026-11-12)
Adds a window from 30 days before to 3 days after every SEUS MBON surface sampling day in the pilot
regions, 2018 to January 2024 (`config/edna_cruise_windows.yaml`, 21 cruises). This gives all 915
surface samples from 2018 on their full 30-day satellite context.
Wait until the 1.4 job has finished (`squeue -u hmp278` shows no `oefm_download`), then:
```bash
git pull
sbatch --account=191001-366746 cluster/inventory.sbatch
# when it has finished:
sbatch --account=191001-366746 --export=ALL,CHAIN=1 cluster/download.sbatch
```
- The inventory only adds the new files; download state is kept. Expect roughly +0.45 TB of
  MODIS/VIIRS/OLCI (estimated from ~0.9 GB per historical day) and a few GB of physical files for
  13 extra days. PACE starts in 2024, so it is barely affected.
- **Check:** `python -m oefm windows | grep -o 'edna_[0-9]*' | sort -u | wc -l` prints 21, and `status` shows the higher
  file counts with 0 failed after the download.
- Regenerate the windows only if the eDNA metadata changes (see `scripts/make_edna_windows.py`;
  the eDNA tables are not on Ptolemy, so this is done on the laptop and committed).

### 1.5 QC ✅
```bash
git pull
sbatch --account=191001-366746 cluster/qc.sbatch
```
Runs, in order: verify physical files → inspect sample files and make maps → observation index →
sensor matchups → HISATZEN/zenith diagnostic → report. No internet needed. Re-running only
processes new files.

Outputs, under `data/`:
- `reports/pilot_acquisition_report.md`, the Phase 1 report
- `reports/inspect/`: per-product structure JSON and maps
- `index/observations.csv`, `counts_by_sensor_region_month.csv`, `multisensor_coincidences.csv`
- `reports/overlap/`: sensor matchup plots
- `reports/zenith/`: HISATZEN diagnostic CSVs and figures

### 1.6 Paste back ✅
See [What to paste back](#what-to-paste-back), row 1.5. Reviewed 2026-11-12: the HISATZEN rule is
applied per sensor (decision log); the QC tables, maps and observation counts raise no blockers.

### 1.7 Re-fetch 8 MUR files ▶️
The report lists 8 `mur_sst` files as `failed` with `verify: md5 mismatch` (5.9 GB). Every other
file of all 15 products is verified. A failed file is deleted and fetched cleanly on the next run:
```bash
git pull
squeue -u hmp278                      # nothing else downloading?
sbatch --account=191001-366746 --export=ALL,CHAIN=0 cluster/download.sbatch
```
**Done when** `python -m oefm status` shows `mur_sst` with 3,290 verified and 0 failed.
If the same 8 fail again, the checksum in the manifest is probably stale (PO.DAAC republished the
files after our inventory). Refresh it, then download again:
```bash
sbatch --account=191001-366746 --export=ALL,PRODUCTS="mur_sst" cluster/inventory.sbatch
# when it has finished:
sbatch --account=191001-366746 --export=ALL,CHAIN=0 cluster/download.sbatch
```
and paste `sqlite3 data/manifest/manifest.sqlite "SELECT filename,last_error FROM granules WHERE product='mur_sst' AND status!='verified'"`.
This does not block Phase 2 work (MUR is cropped in 2.2; 8 missing days would only be gaps).

---

## Phase 2: preprocessing and pilot experiments (2.0–2.4 built; the rest 🛠)

Briefs: `docs/phase2_instructions.md` holds the original brief, its storage revision, and the
physics/nutrient revision with your three adjustments (Florida metabarcoding replaces Bio-GO-SHIP;
nutrient supervision kept; Florida SaWS for Sargassum). Below is the planned order. Each step gets
exact commands once it is built and tested locally. 1.5 has run and been reviewed (2026-11-12), so Phase 2 can start.

### Running 2.0–2.4b on Ptolemy

**P.0, once (new dependency `zarr`):**
```bash
git pull
sbatch --account=191001-366746 cluster/setup_env.sbatch
```
**Copy the private evaluation points once** (video MaxN locations are kept out of the public repo):
```bash
# on your laptop, from multimodal_seusmbol/ocean_ecology_foundation_model
scp config/eval_points_private.csv hmp278@ptolemy-login-2.arc.msstate.edu:/scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model/config/
```
Without it, 2.4b still runs on eDNA, SFER and CRCP.
**Then a short trial** (first 40 files of crop and of tiles, ~10 min), to check speed and output before the full run:
```bash
sbatch --account=191001-366746 --export=ALL,LIMIT=40 cluster/prep.sbatch
```
Paste back: `tail -40 logs/oefm_prep_<id>.out` (and `.err` if it failed).
**Then the full run** (the trial's outputs are reused; it continues where the trial stopped):
```bash
sbatch --account=191001-366746 cluster/prep.sbatch
```
Runs layout → grid → crop → tiles → examples → evalcover. Measured locally: ~0.5–1.5 s per granule per
worker, so ~35,000 granules on 12 workers should take a few hours; physical crops are minutes.
Resumable: resubmit the same command after a crash or the 48 h limit. Only `data/processed/` is
written. Expected size: ~120 GB of tiles (OLCI 300 m and PACE dominate) + ~15 GB of physical crops.
Run 1.7 (MUR re-fetch) first if you can; otherwise the 8 missing MUR days are cropped on a later re-run.

Outputs, under `data/processed/regional/`:
- `processing.sqlite`: every output with its raw sources (sha256), git commit, parameters, transformations
- `physical/<product>/<year>/*.nc`: cropped physical files, values and encoding unchanged
- `tiles/v1/grid/`: `tiles.csv` (272 tiles, fixed IDs like `X+05Y-07`), `static.zarr.zip` (lon/lat/GEBCO depth per cell), `grid.json`
- `tiles/v1/<product>/<year>/<granule>.zarr.zip`: one per overpass (layout in `oefm/tiling.py`)
- `tiles/v1/index/tile_observations.parquet` (the loader's entry point), `summary_by_product.csv`,
  `eval_coverage.csv` and `eval_coverage_summary.csv` (2.4b)
- `tiles/v1/examples/*.png` with `*_panels.csv` (2.4)

### 2-I Data preparation

| # | Step | Needs | Runs on | Status |
|---|---|---|---|---|
| P.0 | Rebuild the Python environment (adds `zarr`) | – | service | ▶️ |
| 2.0 | `data/processed/regional/` and `data/processed/global/` (global stays empty), plus a processing manifest (source files, code version, subset, transformations) | – | service | ▶️ built, run in `prep.sbatch` |
| 2.1 | Final QC policy: hard rejects per sensor, HISATZEN as a tag (not for MODIS), 665 nm angle weights from 45°, a strict subset for spectral evaluation, SFREFL rules kept separate | 1.5 reviewed ✅ | – | ✅ in `config/pilot.yaml` (`processing.qc`), applied by 2.3 |
| 2.2 | Regional crops of MUR and the other physical fields, with a context margin (raw global files kept) | 1.4 | service | ▶️ built, run in `prep.sbatch` |
| 2.3 | Tile grid (38.4 km tiles, 32×32 at 1.2 km, OLCI 128×128 at 300 m, local equal-area projection, fixed tile IDs) and per-overpass gridding to Zarr | 2.1 | service | ▶️ built, run in `prep.sbatch` |
| 2.4 | Visual check: a few multisensor tile examples | 2.3 | service | ▶️ built, run in `prep.sbatch` |
| 2.4b | Which evaluation samples have usable PACE (and OLCI/MODIS/VIIRS) pixels, per dataset and year | 2.3 | service | ▶️ built, run in `prep.sbatch` |
| 2.5 | Sequence builder and loader (causal 30 d / ±15 d, physics lookback 90 d, lengths 8/16/24, five masking types), grouped train/validation splits by tile and time block. **PACE years per the PACE-era decision below (needs your OK)** | 2.3 | service | 🛠 |
| 2.6 | PISCES nutrient targets (PO₄, NO₃, Si), regridded to the tiles **as targets only**, never encoder inputs | 2.3 | service | 🛠 |
| 2.7 | Evaluation datasets, each matched to tiles and times, with grouped splits fixed before training (see [docs/evaluation_inventory.md](docs/evaluation_inventory.md)): Florida 16S/18S on the shelf and Keys (NMF components, dominant taxa, composition), SFER nutrients and fluorometric Chl, reef fish (CRCP, video MaxN). Sargassum and HPLC are optional and must not delay Stage 1 | 2.3 | service | 🛠 |

### 2-II Stage 1: choose the optical objective (optical inputs only, matched data and capacity)

| # | Step | Runs on | Status |
|---|---|---|---|
| 2.8 | Encoder plus short debug runs (1 GPU), then memory and throughput benchmarks (4 GPUs) | gpu-a100 | 🛠 |
| 2.9 | Train **A** masked reconstruction, **B** JEPA + SIGReg, **C** B + PACE spectral residual | gpu-a100 | 🛠 |
| 2.10 | Compare: collapse checks and effective rank, PACE spectral retention (frozen encoder + identical decoder, error by wavelength), frozen-embedding probes on 2.7, training behaviour | gpu-a100 | 🛠 |
| 2.11 | **Pick the Stage 1 winner** (reconstruction is kept if it beats JEPA). You decide; recorded in the decision log | – | 🛠 |

### 2-III Stage 2: environmental information (starting from the winner)

| # | Step | Runs on | Status |
|---|---|---|---|
| 2.12 | **Physics**: physical fields as inputs, with modality dropout so the model can be evaluated with and without them | gpu-a100 | 🛠 |
| 2.13 | **Nutrient targets**: optical inputs plus a small auxiliary head predicting PISCES PO₄/NO₃/Si (configurable loss weight) | gpu-a100 | 🛠 |
| 2.14 | **Physics + nutrient targets** | gpu-a100 | 🛠 |

### 2-IV Evaluations (frozen embeddings, all six configurations, the same fixed splits)

| # | Evaluation | Status |
|---|---|---|
| 2.15 | **Florida metabarcoding:** per NMF component (16S and 18S), dominant taxa, overall composition. Splits grouped by station and by cruise/time. Baselines: metadata only, direct physical variables, Granite. Per-component gains, so it is visible which components benefit from physics vs nutrient supervision. Measured SFER nutrients as an independent target | 🛠 |
| 2.16 | **Sargassum (optional, blocked on data):** the 7-day forecast first (inputs end before the target week), against persistence, seasonal climatology and current-driven transport; same-period detection second. A separate Sargassum evaluation region if needed; SaWS never enters pretraining | ❌ needs numerical SaWS grids from USF |
| 2.17 | **Fish (CRCP Keys, video MaxN shelf/Keys) and pigments:** optical-only vs physical-only vs combined. SFER fluorometric Chl is the main phytoplankton target; HPLC only if SeaBASS adds enough | 🛠 |
| 2.18 | OLCI 300 m branch proof of concept vs upsampled 1.2 km | 🛠 |
| 2.19 | Consolidated report on the six configurations: failure cases, negative results, recommendation, questions for Andre | 🛠 |

Storage rule (Phase 2 storage revision): `data/raw/` is the immutable archive shared by every
dataset. Regional and global processed data live in separate directories, and each records the raw
files, code version, subset and transformations it was built from, so it can be regenerated
without re-downloading.

---

## Decision log

Each entry gives the decision, the evidence, the reasoning, and what would reopen it. Newest first.

### PACE era vs the 2024 hold-out year. *Proposed (2026-11-12), needs your OK before 2.5.*
**The constraint (your note).** PACE data start 2024-03-05. Counting every evaluation sample in the
pilot regions from 2018 on (`config/eval_points.csv`, from `scripts/make_eval_points.py`) by date:

| Dataset | Samples 2018+ | In PACE era (≥ 2024-03-05) | With a full 30-day PACE history | PACE-era years | PACE-era cruises / survey blocks |
|---|---|---|---|---|---|
| eDNA 16S (18S identical) | 913 | **115** (13%) | **92** | 2024 only (Mar–Jul) | 3 (Mar 5–9, May 18–24, Jul 23–30, 2024); the March cruise has almost no PACE history |
| SFER nutrients / Chl (≤ 5 m) | 3,174 | 860 | 797 | 2024 (395), 2025 (465) | 11 (5 in 2024, 6 in 2025) |
| Fish, CRCP visual census | 3,018 | 1,057 | 1,057 | 2024 only | 6 |
| Fish, video MaxN | 3,805 | 616 | 616 | 2024 only | 5 (dates are the 15th of each month: month precision only) |

These are dates only. How many of them have a **usable** (cloud-free, valid) PACE pixel at the
sample is counted by step 2.4b on Ptolemy (`eval_coverage_summary.csv`); expect well under half.

**What this means.** If 2024 is the eDNA test year, *every* PACE-era eDNA sample is a test sample:
the eDNA probes are trained without any PACE-era sample, and a 2024 PACE hold-out from pretraining
would remove 31% of all PACE observations (2024: 2,480 of 8,093 PACE region-observations; 2025: 3,211; 2026: 2,402).

**Proposal.**
1. **Encoder pretraining (self-supervised, no labels) uses PACE from 2024 and 2025.** Imagery from a
   test year is not label leakage; holding 2024 out of pretraining as well costs 31% of PACE. A
   sensitivity run with 2024 excluded from pretraining can be added if a reviewer asks.
2. **PACE spectral-retention test (Objective 6) years:** train on PACE 2024-03-05 – 2025-12-31,
   validation 2026-01-01 – 2026-02-28, a 30-day gap (March 2026, unused), **test 2026-04-01 – 2026-10-01**
   (the last PACE file). Every season is in training; the test lacks winter, so a second test holds out
   20% of tiles (spatial blocks) across all PACE dates, and errors are reported by wavelength and month.
3. **eDNA:** keep 2024 as the temporal hold-out (184 samples, incl. the January 2024 cruise). The
   PACE question is answered *on the same 2024 test samples* by a paired ablation: embeddings with
   PACE visible vs PACE masked (the loader's sensor masking), so the probe never needs PACE-era
   training samples. A within-PACE cruise-left-out check (92 samples, 2 cruises) is reported only as
   exploratory.
4. **SFER is the one dataset with PACE-era samples in two years:** PACE-vs-no-PACE probes there use
   train 2024 / test 2025 (395 / 465 samples).
5. Every evaluation table reports, per dataset and year, the number of samples with usable PACE
   (from 2.4b) next to the totals.

### Florida eDNA cruise windows: added (2026-11-12, approved)
Only 119 of 731 surface samples from 2018–2023 had their full 30-day satellite context, because
the cruises (mostly January, March, May, July/August, October and December) fall outside the
seasonal windows. You approved adding a window from 30 days before to 3 days after each surface
sampling day in the pilot regions. Generated by `scripts/make_edna_windows.py` into
`config/edna_cruise_windows.yaml` (21 cruises, 2018 – January 2024; the January 2024 cruise needed
context from December 2023). Result: **all 915 surface samples from 2018 on are covered**. Cost:
+487 satellite days (2,006 → 2,493) and +13 physical days. Downloaded in step 1.4b.

**Data note.** In `eDNA_data_sources/seusmbon_16S_metadata.tsv` and `..._18S_metadata.tsv`, the
`decimalLatitude` and `decimalLongitude` columns are swapped (latitude holds ~−82). The script
detects and corrects this; other analyses using those files should too.

### Evaluation inventory and scope (2026-11-12)
Full table: [docs/evaluation_inventory.md](docs/evaluation_inventory.md). In short:
- **Ready:** Florida 16S/18S (915 matched samples, shelf and Keys only), measured nutrients (553
  at eDNA samples; ~2,550 SFER samples overall), SFER fluorometric Chl (2,139), reef fish (CRCP
  Keys 2,606 events; video MaxN 1,517).
- **Microbial evaluation is restricted to the shelf and Keys** (your instruction). The offshore
  Gulf box has no eDNA, SFER or fish data; it stays in pretraining and serves other benchmarks.
- **Do not delay Stage 1** for Sargassum or HPLC; both are optional.

### Evaluation targets: Florida metabarcoding replaces Bio-GO-SHIP (2026-11-12)
Your adjustment 1. The microbial evaluation uses the SEUS MBON 16S/18S tables. Inputs: the
replicate-averaged, prevalence-filtered tables in `eDNA_data_sources/processed/`, the existing
Hellinger NMF exports (k = 5–40; the k to use is still to choose, probably the one used in the
CatBoost runs), dominant taxa, and overall composition (e.g. Bray–Curtis / Hellinger PCA).
Splits are grouped by station and by cruise/time, because replicates and repeat visits share
conditions. The Bio-GO-SHIP gene panel (414 genes, pho/pst) is not used in this phase. Its
phosphate question carries over as: do NMF components linked to phosphate gain more from nutrient
supervision, checked against measured SFER PO₄?

### Nutrient supervision: PISCES as targets only (2026-11-12)
Revision Stage 2 plus your adjustment 2. PO₄, NO₃ and Si from a documented PISCES product are
auxiliary *targets*; they never enter the encoder. Candidate: Copernicus Marine
`GLOBAL_MULTIYEAR_BGC_001_029` (PISCES-based reanalysis). A monthly 0.25° Florida/Gulf subset of it
(`cmems_mod_glo_bgc_my_0.25deg_P1M-m`, 2016–2025) is already in
`physiochem_data_sources/copernicus_grided_products/`. Still to settle: version, daily vs monthly
(a daily product exists), and the NRT product (`GLOBAL_ANALYSISFORECAST_BGC_001_028`) for dates
after the reanalysis ends. At 0.25° (~25 km), a 38.4 km tile holds only one or two PISCES cells,
so the target is a tile-level value, not per-pixel. Measured SFER nutrients are the independent check.

### Sargassum benchmark: blocked on numerical data (2026-11-12)
**Finding.** SaWS's weekly `FA_UNET_DENSITY` is public only as colour-mapped PNGs (georeferenced
by world files, 1 km, scale 0–0.4). The files sit under a `subscription` path with no download
listing, and there are no public numerical grids. Alternatives checked:
- NOAA AOML ERDDAP serves USF **AFAI** grids (1/3/7-day, 0.015°, 2016 – now). That is the
  floating-algae *index*, not classified Sargassum.
- NCEI 0190272 has gridded density only for 2001–2010 at 0.5°.
- The USF biomass dataset is monthly sub-region totals (its only Gulf sub-region is the NW Gulf).

**Decision (per your instructions).** No labels from PNGs, and no approximate labels from AFAI.
The positive-count check by region, month and year waits for real data.
**Next action:** request numerical weekly `FA_UNET_DENSITY` (or daily classified maps) for the Gulf
and Florida Straits, 2018–2026, from the USF Optical Oceanography Lab (Chuanmin Hu).
If the pilot boxes see too few mats, add a separate Sargassum evaluation region; don't enlarge the
offshore control box. The forecast has priority over same-period detection: inputs end before the
target week, with persistence, seasonal-climatology and current-advection baselines.

### HPLC pigments: too sparse so far; SFER Chl is the main target (2026-11-12)
BCO-DMO 851250 has 18 surface casts (May 2017 and May 2018, central-eastern Gulf; 11 in the
offshore box), all outside our satellite windows. SeaBASS Gulf HPLC found so far: `LAMONT_GOM`
(2016, 2019) and Lohrenz 2017 (northern Gulf); nothing on the Florida shelf or Keys. SeaBASS has no
search API; the remaining check is a manual SeaBASS file search (instructions in the evaluation
inventory). HPLC stays optional. SFER fluorometric Chl is a separate, main target.

### HISATZEN (high sensor-zenith flag): tag, don't mask, except MODIS. *Settled on the full QC run (2026-11-12).*

**Result of the full QC run (1.5, reviewed in 1.6).** 40 granules per product for the decode
check; same-day matchups against a reference sensor viewing at < 45°. Numbers are
`ratio vs < 45°`, median over reference sensors (range in brackets). Source:
`data/reports/zenith/decode_check.csv`, `angle_effect_by_target.csv`, report section 4.

| Sensor | Decode: flag edge (granules), agreement median (min) | Valid share, L3 → without HISATZEN (shelf / Keys / offshore) | 412–555 nm at 60–90° (pairs) | 665 nm at 45–60° / 60–90° | Rule | Policy |
|---|---|---|---|---|---|---|
| PACE OCI AOP | 59.86° (34), 0.997 (0.981) | 0.30→0.33 / 0.23→0.26 / 0.23→0.26 | 1.03, 1.02, 1.02, 1.08 (31) | 1.20 / 1.47 | ✅ pass | tag; 665 nm weighted from 45° |
| PACE OCI SFREFL | 59.83° (32), 0.997 (0.980) | 0.47→0.51 / 0.42→0.47 / 0.24→0.29 | not tested (matchups use Rrs) | – | decode ✅ | tag; its own rules (2.1) |
| VIIRS SNPP | 60.15° (28), 0.996 (0.988) | 0.19→0.28 / 0.14→0.23 / 0.14→0.16 | 0.97, 0.98, 0.97, 0.94 (45) | 0.78 / 0.64 | ✅ pass | tag; 665 nm weighted from 45° |
| VIIRS NOAA-20 | 60.17° (25), 0.996 (0.990) | 0.16→0.20 / 0.14→0.18 / 0.15→0.16 | 0.99, 0.99, 0.97, 0.95 (39) | 0.95 / 0.86 | ✅ pass | tag; 665 nm weighted from 60° |
| VIIRS NOAA-21 | 60.15° (23), 0.997 (0.982) | 0.20→0.26 / 0.18→0.24 / 0.09→0.11 | 1.02, 1.00, 0.99, 0.91 (42) | 0.81 / 0.70 | ✅ pass (555 nm at the limit) | tag; 665 nm weighted from 45° |
| MODIS Aqua | 59.91° (29), 0.999 (0.981) | 0.30→0.31 / 0.26→0.27 / 0.18→0.19 | **1.11** (0.90–1.31), 0.99, 0.95, 0.91 (37) | 0.92 / 0.95 (0.37–1.48) | ❌ 412 nm just outside ±10% | **HISATZEN stays a hard mask** |
| OLCI S3A / S3B | never flagged (max zenith 54.3° / 56.9°) | unchanged | no 60–90° class | S3A 0.87, S3B 0.73 at 45–60° | n/a | nothing to decide |

Maps (`*_example_zenith.png`, `*_example_Rrs*.png`, `*_reflectance_vs_zenith.png`): no stripes or
discontinuities at the swath edge. Reflectance is continuous across the 60° contour in the PACE,
VIIRS and SFREFL examples. SNPP 665 nm falls toward zero above 60° in the zenith-bin plot (the
lower quartile goes negative), the same signal the matchups show.

**Why MODIS goes back to the hard mask.** It fails the stated rule only narrowly (412 nm 1.11 at
60–65°, one reference as high as 1.31), and relaxing the flag recovers only 1–7% more pixels
(relaxed/L3 = 1.01–1.07), against 9–56% for VIIRS and 10–20% for PACE. Little to gain, so we follow
the rule. MODIS never exceeds 65.4° here.

**Red band.** 665 nm is angle-dependent from about 45° in four sensors (SNPP 0.78, NOAA-21 0.81,
OLCI-B 0.73, PACE 1.20), not only above 60°. It also disagrees between sensors at low angles. So
the red band gets an angle-dependent uncertainty/weight starting at 45° (NOAA-20: 60°), set in 2.1
from these ratios, and is never dropped. The sign differs (PACE high, VIIRS low), so it is a
per-sensor term, not one shared curve.

**OLCI-B at 45–57°** also shows 412 nm 0.91 and 555 nm 0.86, but only from 13 pairs against 3
references. Not a HISATZEN question; recheck with more pairs once tiles exist (2.3).

**Applied in 2.1:** the training mask is `l3_no_hisatzen` for PACE and the three VIIRS, and
`l3_default` (HISATZEN included) for MODIS. The strict spectral-evaluation subset uses
`l3_default` for every sensor. The zenith angle and the flag stay with every observation.

The text below is the reasoning from before the full run, kept for the record.

**Decision so far (2026-11-12, confirmed by the Phase 2 brief):** HISATZEN is removed from the
hard mask for the pilot. The flag and the sensor zenith angle are kept with every observation, so
high-angle pixels can be tagged as lower confidence, left out of the strict spectral-evaluation
subset, or down-weighted in training. No new angle threshold is chosen yet.

**Still a hard mask, whatever happens to HISATZEN:** ATMFAIL (failed atmospheric correction),
NAVFAIL, HILT (saturation), missing reflectance, land and cloud. The pilot mask is
`l3_no_hisatzen`: NASA's standard Level-3 flag set minus HISATZEN.

**Why the question came up.** On the smoke-test day, HISATZEN removed 99–100% of PACE and NOAA-21
pixels over the Keys and the shelf, because those regions fell near the swath edge. Under NASA's
default Level-3 mask, PACE on the shelf had 0% valid pixels.

**Why relaxing it is defensible.**
- NASA treats HISATZEN as a Level-3 exclusion, not a Level-2 failure. Every file records l2gen's
  `satzen = 60` and `masksatzen = 0`: the pixel is flagged, but its values are processed normally.
- We are learning representations from Level-2 reflectance, not reproducing NASA's Level-3 quality
  standard.

**Known risks at high angles:** longer atmospheric path (atmospheric-correction uncertainty),
larger and distorted pixel footprints at the swath edge, and possible sensor-specific spectral
biases. PACE's two-day global coverage is specified only up to 60°.

**How it is being tested** (`python -m oefm zenith`, run by `qc.sbatch`). Sensor zenith is not
stored in the L2 files, so it is reconstructed for every pixel from the file's geolocation and the
orbit altitude. PACE's ±20° tilt is included; for OLCI, nadir is the column with the smallest
pixel spacing. Five checks, all per sensor:

| Check | What it shows | Smoke-test day (2025-06-10) |
|---|---|---|
| Decode check | Whether the HISATZEN bit and our angle agree | Bit 32 in every product. The flag edge falls at 59.4–60.2°, and flag and angle agree for 99.3–99.8% of pixels. Decoding and geometry are both right. |
| Coverage with/without the flag | How much data relaxing it recovers | PACE shelf 0% → 34% valid, SNPP shelf 3% → 35%, NOAA-20 offshore 8% → 21%. MODIS and OLCI never exceed 60° here (OLCI's maximum is ~54°), so nothing changes for them. |
| Zenith and reflectance maps | Spatial artefacts at the swath edge | No obvious artefacts on the one day. The flag edge lies on the 60° line. |
| Reflectance by zenith bin | Implausible values, negatives | Blue-green bands look normal. SNPP 665 nm is negative in 23% of pixels at 55–60° and 52% at 65–75°. Caveat: high angles fell over different water (the shelf) than low angles, so geography is mixed in. |
| Same-day matchups vs a sensor viewing at < 45° | Angle-dependent spectral bias, with sensor offsets divided out | NOAA-20 at > 60° vs < 45°: 412–555 nm within 5% (0.95–1.00); 665 nm 0.70. NOAA-21 at 45–60°: all bands within 3%. Based on 1–4 pairs, so not yet evidence. |

**Reading so far:** high-angle blue-green reflectance looks usable, and the red band (665 nm)
degrades at high angles. The red band already disagreed between sensors at every angle (median
ratios of 0.4–0.7 vs PACE), so it needs harmonisation or uncertainty weighting regardless.

**Proposed acceptance rule, to confirm on the full QC run (1.5).** For each sensor, keep high-angle
pixels as tagged training data if:
1. the decode check holds (flag edge within 60 ± 1°, ≥ 99% agreement);
2. the maps show no swath-edge stripes or discontinuities;
3. in the matchups, 412–555 nm stay within about ±10% of the same sensor's low-angle ratio at
   60–90°, with at least ~20 pairs behind the number.

Bands that fail (likely 665 nm and the red/NIR) stay in the data but get an angle-dependent
uncertainty or weight instead of being removed. The **strict subset** for independent spectral
evaluation uses NASA's full Level-3 mask (HISATZEN included).

**What would reopen it:** a sensor failing the decode check (a bug), stripes at the swath edge,
or blue-green bias beyond ~10% at high angles in the full matchups. For such a sensor, HISATZEN
goes back into the hard mask.

### MUR SST: keep the global archive, crop for the pilot (2026-11-12)
Keep all ~2.2 TB of global daily files in `data/raw/mur_sst/`, so global training needs no
re-download. The Phase 2 revision confirms this and overrides Objective 1's "do not store". The
pilot uses regional crops with a context margin (step 2.2) in `processed/regional/`, keeping the
original timestamps, units, metadata and provenance. The same rule applies to every product:
reusable originals in `raw/`, separate regional and global processed sets.

### OLCI: stop at April 2026 (2026-11-12)
NASA's OB.DAAC OLCI record ends 2026-04-14 (Sentinel-3A) and ~2026-03 (Sentinel-3B). We do not
backfill from EUMETSAT/CDSE. That would bring a different atmospheric correction and need a
cross-calibration, which isn't worth it for the pilot. Reopen if global training needs OLCI after
April 2026.

### One download job at a time (2026-11-11)
The manifest is a single SQLite file on GPFS, which does not lock reliably across nodes. Concurrent
jobs had already overlapped once without harm (they found nothing to do), but they are a corruption
risk. A crash in job 116018 (threads sharing one database connection) is fixed in `01d3ed4`.

### Physical products as monthly subsets (Copernicus Marine), originals for NASA (2026-11-11)
Copernicus Marine serves cloud-native datasets with no "original files", so we store monthly
subsets of the pilot domain with all variables. NASA products are stored as the original granules.

### Earlier facts worth remembering
- IMERG Final ends 2025-09-30; IMERG Late fills later days and is kept as a separate source.
- MODIS Aqua ends 2026-05-31.
- Processing versions differ across sensors (R2022.0, 2025.0, PACE 3.1/3.2). Harmonisation keys on
  (sensor, version).
- The red band (665 nm) disagrees between sensors even at low angles.

---

## What to paste back

Paste the text output (not the data files). If a file is long, the listed `tail`/`sed` is enough.

| After step | Paste this | So that I can check |
|---|---|---|
| any failed job | `tail -40 logs/<name>_<id>.err` and `tail -20 logs/<name>_<id>.out` | the error, and whether it is safe to resubmit |
| 1.2 inventory | `source cluster/env.sh && python -m oefm status` | all 15 products present, TOTAL ≈ 5,960 GB expected |
| 1.4 downloads | `source cluster/env.sh && python -m oefm status` | every product verified, 0 failed, the measured rate |
| 1.4b inventory | `python -m oefm status` and `python -m oefm windows \| grep -o 'edna_[0-9]*' \| sort -u \| wc -l` | the new files are in and 21 eDNA windows are loaded |
| P.0 env | `tail -5 logs/oefm_env_<id>.out` | zarr installed |
| 2.0–2.4b trial (`LIMIT=40`) | `tail -40 logs/oefm_prep_<id>.out` | speed per granule, no failures |
| 2.0–2.4b full | `tail -60 logs/oefm_prep_<id>.out`; `cat data/processed/regional/tiles/v1/index/summary_by_product.csv data/processed/regional/tiles/v1/index/eval_coverage_summary.csv`; copy back `data/processed/regional/tiles/v1/examples/` | tile counts, the PACE-usable evaluation counts, the visual check |
| 1.7 MUR re-fetch | `python -m oefm status` (and the sqlite query in 1.7 if they fail again) | mur_sst 3,290 verified, 0 failed |
| 1.4, if failures | `sqlite3 data/manifest/manifest.sqlite "SELECT product,status,filename,last_error FROM granules WHERE status!='verified' LIMIT 30"` | why those files fail |
| 1.5 QC | Sections 4–6 of `data/reports/pilot_acquisition_report.md`: `sed -n '/^## 4\./,$p' data/reports/pilot_acquisition_report.md` | mask scenarios, matchups, HISATZEN tables, coverage counts |
| 1.5 QC | `cat data/reports/zenith/decode_check.csv data/reports/zenith/angle_effect_by_target.csv` | the HISATZEN acceptance rule, per sensor |
| 1.5 QC, figures | Copy back (`scp`) or attach: `data/reports/zenith/*_example_zenith.png`, `*_reflectance_vs_zenith.png`, `*_overlap_angle_effect.png` | swath-edge artefacts, which only show up visually |
| 1.5 QC, sizes | `du -sh data/raw/* data/index data/reports` | storage matches the inventory |
| Phase 2 steps | Listed here as each step is built | – |

To copy figures to your laptop:
```bash
scp -r hmp278@ptolemy-login-2.arc.msstate.edu:/scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model/data/reports/zenith ./zenith_figs
```
(adjust the login host if you use a different one)

---

## Job history

Newest first. Add a line when something notable happens.

| Date | Job | What happened |
|---|---|---|
| 2026-11-12 | 116100 | prep trial (`LIMIT=40`): layout, grid (272 tiles), 40 CCMP crops OK; all 40 tile jobs failed on 2018 files without `scan_line_attributes/time`. Fixed: fall back to year/day/msec, then to the file's start/end time. Failed granules are retried automatically on the next run |
| 2026-11-12 | qc.sbatch | QC run on all data, reviewed (1.6). 34,000+ files verified across 15 products, 6.44 TB; 8 MUR files failed md5 (step 1.7). 21 eDNA windows loaded. HISATZEN settled per sensor (MODIS keeps the hard mask). 7,245 region-days with 2+ instrument families, 2,610 with PACE |
| 2026-11-12 | download (1.4, 1.4b) | PACE, physical products and the eDNA-window days downloaded; ~220–270 MB/s aggregate |
| 2026-11-11 | 116027–116029 | Three download jobs ran at once and found nothing to do; the six satellite products are fully verified (1.79 TB) |
| 2026-11-11 | 116018 | Satellite download crashed at 9103/9129 (thread-unsafe manifest writes; fixed in `01d3ed4`); follow-on 116019 finished the rest |
| 2026-11-11 | 116016 | Smoke test, 23/23 verified; ~25 MB/s per NASA connection |
| 2026-11-11 | (inventory) | First inventory stopped before PACE; re-run by the user |
