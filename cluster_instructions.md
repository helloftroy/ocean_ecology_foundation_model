# Cluster instructions: order of operations on Ptolemy

This is the single place to check what to run next, what has already run, and what to paste
back for review. It is updated with every change to the steps. Last updated: **2026-11-12**.

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
| 1.4 | Download PACE, MUR, CCMP, IMERG, Copernicus Marine, GEBCO | ▶️ ~4.2 TB |
| 1.5 | QC, observation index, matchups, HISATZEN diagnostic, report | ⏳ after 1.4 |
| 1.6 | Paste back the QC results for review | ⏳ after 1.5 |

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

### 1.4 Remaining downloads ▶️
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

### 1.5 QC ⏳
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

### 1.6 Paste back ⏳
See [What to paste back](#what-to-paste-back), row 1.5.

---

## Phase 2: preprocessing and first training experiments (🛠 not built yet)

The brief is in `docs/phase2_instructions.md`. Below is the planned order; it is filled in with
exact commands as each part is built and tested locally. **Do not start Phase 2 jobs until 1.5
has run and been reviewed.** The QC results feed the mask policy (2.1) and the gridding (2.3).

| # | Step | Needs | Runs on | Status |
|---|---|---|---|---|
| 2.0 | Create `data/processed/regional/` and `data/processed/global/` (global stays empty), plus a processing manifest | – | login node | 🛠 |
| 2.1 | Finalise the QC policy: hard-reject flags per sensor, HISATZEN as a tag, a strict subset for spectral evaluation, SFREFL rules kept separate | 1.5 reviewed | – | 🛠 |
| 2.2 | Regional crops of MUR and the other physical fields, with a context margin, into `processed/regional/physical/` (raw global files kept) | 1.4 | service | 🛠 |
| 2.3 | Tile grid (38.4 km tiles, 32×32 at 1.2 km, OLCI 128×128 at 300 m, local equal-area projection, fixed tile IDs) and per-overpass gridding to Zarr | 2.1 | service | 🛠 |
| 2.4 | Visual check: a few multisensor tile examples | 2.3 | service | 🛠 |
| 2.5 | Sequence builder and loader (causal 30 d / ±15 d, physics lookback 90 d, lengths 8/16/24, masking types), grouped train/validation splits by tile and time block | 2.3 | service | 🛠 |
| 2.6 | Encoder plus short debug runs (1 GPU), then memory and throughput benchmarks (4 GPUs) | 2.5 | gpu-a100 | 🛠 |
| 2.7 | Experiments A (masked reconstruction), B (JEPA + SIGReg), C (B + PACE spectral residual) | 2.6 | gpu-a100 | 🛠 |
| 2.8 | PACE spectral-retention test (frozen encoders, same decoder, per-wavelength errors) | 2.7 | gpu-a100 | 🛠 |
| 2.9 | Embedding diagnostics (collapse, effective rank, modality removal), physics ablation, first eDNA/pigment probes | 2.7 | gpu-a100 | 🛠 |
| 2.10 | OLCI 300 m branch proof of concept vs upsampled 1.2 km | 2.7 | gpu-a100 | 🛠 |
| 2.11 | Phase 2 report and recommendation (questions for Andre) | all | – | 🛠 |

Storage rule (Phase 2 revision): `data/raw/` is the immutable archive shared by every dataset.
Regional and global processed data live in separate directories, and each records the raw files,
code version, subset and transformations it was built from, so it can be regenerated without
re-downloading.

---

## Decision log

Each entry gives the decision, the evidence, the reasoning, and what would reopen it. Newest first.

### HISATZEN (high sensor-zenith flag): tag, don't mask. *Under evaluation.*

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
| 2026-11-11 | 116027–116029 | Three download jobs ran at once and found nothing to do; the six satellite products are fully verified (1.79 TB) |
| 2026-11-11 | 116018 | Satellite download crashed at 9103/9129 (thread-unsafe manifest writes; fixed in `01d3ed4`); follow-on 116019 finished the rest |
| 2026-11-11 | 116016 | Smoke test, 23/23 verified; ~25 MB/s per NASA connection |
| 2026-11-11 | (inventory) | First inventory stopped before PACE; re-run by the user |
