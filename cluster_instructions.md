# Cluster instructions: order of operations on Ptolemy

This is the single place to check what to run next, what has already run, and what to paste
back for review. It is updated with every change to the steps. Last updated: **2026-11-23** (G0 and G done: B and C fail the amplitude test, so arm D gets an amplitude-retention term; next: arm D at ν = 0.1 and 0.3 = stage 1.5, fork 2 (**H**), then evaluate all five (**I**); in parallel the Bio-GO-SHIP global benchmark (**J**)).

- Briefs: [docs/phase1_instructions.md](docs/phase1_instructions.md),
  [docs/phase2_instructions.md](docs/phase2_instructions.md)
- Details of each command: [README.md](README.md); project state: [PROJECT_NOTES.md](PROJECT_NOTES.md)

Status key: ✅ done · ▶️ next · ⏳ waiting on an earlier step · 🛠 not built yet (scripts still to write)

---

## ▶️ What to run now (2026-11-23)

Prep v2 (job 116124) has finished. The remaining steps before training, in order. A and B are
independent and can run at the same time; C needs B; D needs A, B and C.

**A. Evaluation targets + PISCES nutrients (2.6, 2.7).** ✅ (job 116167: PISCES 3,295 of 3,301 days,
3,218 reanalysis + 77 forecast, median tile coverage 1.00; evaluation sets matched).

**A2. Refreshed eDNA targets: sequencing platform and depth (2026-11-20).** ✅ joined (job 116353; sample counts unchanged; confirm the `seq_` column count once). Any time before F; it
runs on `service`, so it can run during training. The two eDNA files now also carry `seq_*` columns
(platform, runs, Element share, replicates, input and table reads); every other column is
identical to the copy from 116167 (checked to 3e-13). From the laptop, in `multimodal_seusmbol/`:
```bash
scp eval_targets/edna_16S.csv.gz eval_targets/edna_18S.csv.gz eval_targets/fit_info.json hmp278@ptolemy-login-2.arc.msstate.edu:/scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model/data/processed/regional/eval/
```
Then on Ptolemy (only the join; nothing is downloaded):
```bash
git pull
sbatch --account=191001-366746 --export=ALL,STEPS=evalsets cluster/targets.sbatch
zcat data/processed/regional/eval/matched/edna_16S.csv.gz | head -1 | tr ',' '\n' | grep -c '^seq_'   # expect 6
```
Paste back: `tail -20 logs/oefm_targets_<id>.out` and the number from the last line.

**B. GPU environment (2.8b),** at the same time as A (if you already ran it before the xarray pin, run it again after `git pull`; pip downgrades xarray in place):
```bash
sbatch --account=191001-366746 cluster/setup_gpu_env.sbatch
```
Paste back: `tail -10 logs/oefm_gpuenv_<id>.out`.

**C. GPU check,** when B has finished (a few minutes on 4 A100s):
```bash
sbatch --account=191001-366746 cluster/gpu_check.sbatch
```
Paste back: `tail -30 logs/oefm_gpucheck_<id>.out`.

**D. Benchmarks (2.8c).** First run (job 116176) failed on every configuration with `anchors 0`:
the GPU environment has pandas 3, which parses dates at microsecond resolution, so the loader's
times came out 1,000× too small (all in January 1970) and no observation fell in the training years.
Not a memory problem. Fixed in `oefm/loader.py` (tests now pass under pandas 3.1). An empty
training set now stops with a clear error, and the benchmark stops if its first configuration fails.

D1. ✅ (rebuilt in 116188). Check whether prep's training statistics were affected (if the CPU environment also had pandas 3
when prep ran, `normstats` silently skipped):
```bash
git pull
source cluster/env.sh && python -c "import pandas; print(pandas.__version__)"
grep -h "normstats\|pace ridge" logs/oefm_prep_116124.out
```
If the grep shows "no training observations ... skipped" (or nothing), or pandas is 3.x, rebuild the
sequence summary and statistics (minutes to an hour):
```bash
sbatch --account=191001-366746 --export=ALL,STEPS="sequences normstats" cluster/prep.sbatch
```
D2. Benchmark ✅ (116191, 116200, 116201; table in the job history). Settings now in
`config/train_stage1.yaml`: length 16, per-GPU batch 32 (no accumulation), 30,000 steps × 128 sequences
(~4 passes over the 932,992 training anchors) for every arm: ~25 h (A), ~27 h (B, C).

**E. Train the three arms (2.9).** Each fits in one 48 h job; `CHAIN=1` queues one follow-on as a
safety net, which cancels itself when the arm has finished. The QOS allows 4 A100s per user, so the
arms run one after another (~3.5 days in all):
```bash
git pull
sbatch --account=191001-366746 --export=ALL,ARM=A,CHAIN=1 cluster/train.sbatch
sbatch --account=191001-366746 --export=ALL,ARM=B,CHAIN=1 cluster/train.sbatch
sbatch --account=191001-366746 --export=ALL,ARM=C,CHAIN=1 cluster/train.sbatch
```
Check now and then: `tail -3 data/runs/stage1_<ARM>/log.jsonl` (per-sensor losses, SIGReg,
effective rank of E1). Warning signs: `effective_rank_e1` falling steadily toward single digits
(collapse), a loss rising for one sensor while the others fall, or NaN.

**F. Evaluation (2.10)** ✅ job 116376 (1 h 30 min; results in the decision log, "Stage 1 results"). **F2 below is still to run.** Original instructions, when all three arms have finished (`weights_final.pt` in each
`data/runs/stage1_<ARM>/`; it needs a GPU, and training holds all four until arm C is done):
```bash
git pull
sbatch --account=191001-366746 cluster/evaluate.sbatch
```
Runs on 1 A100 (a few hours): embeddings at every evaluation sample for each arm (causal and
retrospective windows, PACE visible and dropped), ridge probes on the held-out labels, and the PACE
spectral-retention decoder on April–October 2026 and the geographic test blocks. eDNA probes are
reported with and without the sequencing covariates (column `covariates`: `seq` / `none`; decision
log, 2026-11-20); needs A2 first. Results in
`data/runs/eval_stage1/`. Paste back `tail -120 logs/oefm_eval_<id>.out` (embedding ranks, the probe
table, the PACE table) and copy back `data/runs/eval_stage1/*.csv` for the full tables.
Optional, to try it on intermediate weights once a GPU is free:
`--export=ALL,TAG=check,WEIGHTS=checkpoint,EXTRA="--limit 300 --pace-anchors 100 50"`.

**F2. Probes again (2026-11-23), after two fixes:** ✅ job of 2026-11-23 10:05 (SFER results and PACE on/off in the decision log; CSVs in `eval_results/stage1/`). (1) SFER ran with 0 targets: Chl is missing for
13% of the samples and any target with a missing training value was dropped; each target now uses
its own measured samples. (2) The summary now uses the median over the targets of a family and
leaves out targets whose test spread is < 0.25 × the training spread (18S NMF k15/k20 component 10
and *Chloropicon*: almost absent in 2024, R² of −15 to −31 dominated the means). Only the probes
rerun (the embeddings are reused; a few minutes):
```bash
git pull
sbatch --account=191001-366746 --export=ALL,EXTRA="--steps probes" cluster/evaluate.sbatch
```
Paste back `tail -80 logs/oefm_eval_<id>.out` and copy `data/runs/eval_stage1/probes*.csv` to the
laptop (into `multimodal_seusmbol/eval_results/stage1/`, outside the repo).

**G0. Refreshed targets + NASA MOANA matchups (2026-11-23, second review)** ✅ (MOANA matchups used in G). On `service`, minutes
plus a ~1 GB download; run it before G. New target columns, every existing column identical (checked):
`phyto_*` in both eDNA files (CLR of phytoplankton groups: 18S diatoms, Dinophyceae, haptophytes,
prasinophytes, other Chlorophyta; 16S Prochlorococcus, Synechococcus, plastids) and `hab_*` habitat
covariates in both fish files (CRCP survey depth and habitat type; MaxN depth and distance to shore).
`moana` downloads PACE_OCI_L4M_MOANA daily 4 km files (~6 MB each) only for the days within ±3 d of a
PACE-era eDNA or SFER sample and writes `eval/moana_matchups.csv.gz`. From the laptop, in
`multimodal_seusmbol/`:
```bash
scp eval_targets/* hmp278@ptolemy-login-2.arc.msstate.edu:/scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model/data/processed/regional/eval/
```
Then on Ptolemy (`credentials.sh` must be sourced for Earthdata, as for the downloads):
```bash
git pull
sbatch --account=191001-366746 --export=ALL,STEPS="evalsets moana" cluster/targets.sbatch
zcat data/processed/regional/eval/matched/edna_18S.csv.gz | head -1 | tr ',' '\n' | grep -c '^phyto_'   # expect 5
zcat data/processed/regional/eval/matched/fish_crcp.csv.gz | head -1 | tr ',' '\n' | grep -c '^hab_'    # expect 9
```
Paste back `tail -20 logs/oefm_targets_<id>.out` (the `moana:` lines give the files found and the
samples with a matchup) and the two numbers.

**G. Robustness and focused checks on A, B, C (2026-11-23)** ✅ job 116439 (results: decision log "G results"). Two evaluation fixes since (phyto point deltas, CRCP groups) take effect in I. They reuse the stage 1
embeddings (windows causal and retro), so no re-embedding is needed. Decision log: "Stage 1.5 and
robustness checks". About 2–4 h on 1 A100, mostly CPU fits. Steps:
- test-year rotation (`robust_rotate*.csv`);
- paired bootstrap intervals on the official split (`robust_bootstrap.csv`);
- PACE visible vs dropped, with probes trained on PACE-era samples (`robust_pace_era.csv`);
- input retention (`robust_retention.csv`);
- the design-document amplitude test on the trained weights, token and E1 levels (`amplitude_check.csv`);
- **focused phyto:** PACE-era eDNA only, phytoplankton groups, leave-one-cruise-out, E1 PACE visible
  vs dropped and vs the MOANA baseline (`focused_phyto*.csv`; the ±1/±3 d windows come in I);
- **focused fish:** Poisson ridge on counts with habitat covariates, added value over habitat alone,
  D², level-free D² and Spearman (`focused_fish*.csv`; the 90-day seasonal window comes in I).
```bash
git pull
sbatch --account=191001-366746 --export=ALL,EXTRA="--steps robust amplitude focused" cluster/evaluate.sbatch
```
Paste back `tail -250 logs/oefm_eval_<id>.out`. Copy `data/runs/eval_stage1/robust_*.csv`,
`focused_*.csv` and `amplitude_check.csv` to `multimodal_seusmbol/eval_results/stage1/`.

**H. Stage 1.5: train arm D at two reconstruction weights** ▶️ G's amplitude test is read: B and C
fail it (E1 moves 0.1–0.3 % under a 2× brightness change; threshold 5 %), so arm D now also has an
amplitude-retention term, κ = 0.1 (`loss.kappa`; a linear head on the context tokens predicts each
cell's two amplitude features). Both D runs use it, so the ν comparison stays clean. Effective rank
is fine for every arm (E1 A 116, B 103, C 127). Arm D is fork 2, hybrid JEPA + light
reconstruction: arm C's loss plus arm A's band decoder at weight ν (`loss.nu`): ν = 0.1 (`stage1_D`)
and ν = 0.3 (`stage1_D3`). Config, data and 30,000 steps are the same as the other arms. Each run
takes about 28–32 h; peak memory should be about 30 GB of 80. Each needs all 4 GPUs, so they run one
after the other (the second queues behind the first).
```bash
git pull
sbatch --account=191001-366746 --export=ALL,ARM=D,CHAIN=1 cluster/train.sbatch
sbatch --account=191001-366746 --export=ALL,ARM=D,NAME=stage1_D3,OVERRIDES="loss.nu=0.3",CHAIN=1 cluster/train.sbatch
```
After about 10 minutes of each, paste back the first lines of `logs/oefm_train_<id>.out` (it prints
the run name and overrides) and `tail -3 data/runs/<run>/log.jsonl`. The log should show `recon`,
the `recon_*` keys and `amp` next to `pred`, `sigreg` and `pace_residual`; for D3, check
`data/runs/stage1_D3/train_config.json` has `"nu": 0.3`.

**I. Evaluate all five runs (after H)** ⏳. Use a new tag. `embed` writes every run into the same
`embeddings_*.npz`, so evaluating only D under the tag `stage1` would overwrite A, B and C. Windows:
causal and retro as before, plus ±1 d and ±3 d matchups (eDNA and SFER) and the 90-day seasonal
window (fish). Each embedding file now records the span actually covered (`span_days`; at most 16
observations nearest the sample are used, so the nominal ±15 d may cover only a few days; the log
prints the median).
```bash
sbatch --account=191001-366746 --export=ALL,RUNS="stage1_A stage1_B stage1_C stage1_D stage1_D3",TAG=stage1_5,EXTRA="--windows causal retro retro1 retro3 seasonal --steps embed probes pace robust amplitude focused" cluster/evaluate.sbatch
```
The run takes about 6–10 h. Paste back `tail -400 logs/oefm_eval_<id>.out` and copy
`data/runs/eval_stage1_5/*.csv` to `multimodal_seusmbol/eval_results/stage1_5/`.

Also paste back from the finished prep, if not done yet: `tail -80 logs/oefm_prep_116124.out`,
`cat data/processed/regional/tiles/v2/index/summary_by_product.csv data/processed/regional/tiles/v2/index/eval_coverage_summary.csv`,
and copy back `data/processed/regional/tiles/v2/examples/`.

**J. Bio-GO-SHIP global benchmark (2026-11-23; runs alongside H on `service`)** ▶️. A downstream
test outside Florida: the Granite / Bio-GO-SHIP metagenome benchmark (same samples, targets and
protocol; instructions in `multimodal_seusmbol/instructions_agent_biogoship.md`). Decision log:
"Bio-GO-SHIP as a global downstream benchmark". Its own config (`config/biogoship.yaml`, generated
by `scripts/make_biogoship_config.py`) and data root (`data_biogoship/`), so nothing in `data/`
changes: 16 cruises (2011–2025), 1,977 samples → 50 zones (each its own LAEA projection, ≤ 1,500 km),
377 corridor boxes, ~13,400 tiles. Windows per box: the cruise window (30 d before the first sample
to 15 d after the last) plus two 45-day PACE-era windows in other seasons. The seven encoder
products as in Florida (no PACE SFREFL: not an encoder input); no physical products (stage 1 is optical).

*J1. Copy the bundle and link GEBCO* (from the laptop, in `multimodal_seusmbol/`):
```bash
scp biogoship_granite_kegg/data/results/benchmark_export.zip hmp278@ptolemy-login-2.arc.msstate.edu:/scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model/
```
Then on Ptolemy:
```bash
git pull
mkdir -p data_biogoship/raw data_biogoship/processed/regional/eval
ln -s "$PWD/data/raw/gebco" data_biogoship/raw/gebco
unzip -q benchmark_export.zip -d data_biogoship/processed/regional/eval/
ls data_biogoship/processed/regional/eval/benchmark_export   # benchmark_eval.py granite kegg samples.csv targets targets_sample tiles.csv within_tile_pairs.csv
```
(If the zip unpacks without the `benchmark_export/` folder, move its contents into one.)

*J2. Dry-run inventory (downloads nothing; ~9,000 CMR searches, a few hours).* `credentials.sh` sourced:
```bash
sbatch --account=191001-366746 --export=ALL,OEFM_CONFIG=config/biogoship.yaml,OEFM_DATA_ROOT=$PWD/data_biogoship cluster/inventory.sbatch
```
Paste back `tail -60 logs/oefm_inventory_<id>.out` (the per-product file counts and GB). I check the
size (estimate 8–10 TB raw) before anything downloads.
✅ Inventory job 116443 (2026-11-24): the searches all ran and the manifest was written per product;
the closing status table crashed (`KeyError: 'core'`, it assumed Florida's `time.core`; fixed).
PACE SFREFL alone was 10,546 files, 7,632 GB. SFREFL is not an encoder input
(`training.sensors.include`), so config/biogoship.yaml now has `exclude_products: [pace_oci_sfrefl]`:
it is never downloaded or tiled (its manifest rows stay `planned`, unused). Per-product totals
without it, on the login node (reads the manifest only):
```bash
git pull
source cluster/env.sh && OEFM_CONFIG=config/biogoship.yaml OEFM_DATA_ROOT=$PWD/data_biogoship python -m oefm status
```
✅ Status (2026-11-24), without SFREFL: 107,100 files, **10.2 TB** (7.7 TB in 2024+, mostly the two
PACE-era seasonal windows; 2.5 TB historical). MODIS 487 GB (from 2011-08-30), OLCI-A 2,306, OLCI-B
2,171, PACE AOP 1,762, VIIRS-J1 1,289, VIIRS-J2 899, VIIRS-SNPP 1,299 GB. Approved for download (J3).

*J3 — superseded by K4 (download-tile-delete, tiles only). Kept for reference:*
```bash
sbatch --account=191001-366746 --export=ALL,OEFM_CONFIG=config/biogoship.yaml,OEFM_DATA_ROOT=$PWD/data_biogoship,CHAIN=3 cluster/download.sbatch
# when the download has finished:
sbatch --account=191001-366746 --export=ALL,OEFM_CONFIG=config/biogoship.yaml,OEFM_DATA_ROOT=$PWD/data_biogoship,STEPS="layout grid tiles sequences" cluster/prep.sbatch
```

*J4. Embed with the Florida-trained runs, then score* (embedding ~1 h on one A100; scoring ~10–20 h
on `service`):
```bash
sbatch --account=191001-366746 --export=ALL,RUNS="stage1_A stage1_B stage1_C",TAG=stage1 cluster/biogoship_embed.sbatch
# when it has finished:
sbatch --account=191001-366746 --export=ALL,TAG=stage1 cluster/biogoship_bench.sbatch
```
Paste back `tail -40` of the embed log and `tail -200` of the bench log; copy
`data_biogoship/runs/eval_stage1/biogoship_*.csv` to `multimodal_seusmbol/eval_results/biogoship_stage1/`.
The bench log starts with the Granite reproduction check (must give all KOs 17.60, C 53.80, D 13.96,
A 13.71, Dp 16.36, B 10.33, De 18.89). After H, rerun J4 with D and D3 added (`TAG=stage1_5`); after
the global retrain (J5), with the retrained winner.

*J5 (after 2.11).* Retrain the 2.11 winner on Florida + the Bio-GO-SHIP corridors, then rerun J4.
Not built yet: a training set over two data roots, and normalisation statistics over both.

**K. Tiling v3: catalog_v1, download-tile-delete (2026-11-24)** ▶️ (decision log "Tiling v3").
Built: `oefm/catalog.py` (catalog_v1: 196 zones, scale error ≤ 0.21 % in band zones, ≤ 0.86 % at the
cap rims), `oefm/tiling3.py` (full swath, int16 at source scale, flag shares, clean mean in mixed
cells, angles, all single-layer fields, PACE Rrs_unc, SFREFL int16 with land cells empty (cloud kept, K1c),
OLCI at 300 m only), `oefm/stream.py` (download → tile → delete), `python -m oefm tiles3 | index3 |
stream | purge-raw`, PACE BGC (`pace_oci_bgc`, v3.2) in the products. Tiles go to the shared store
`data/processed/global/tiles/v3/` (catalog.json, `<product>/<year>/<granule>__<zone>.zarr.zip`,
`index/tile_observations_<phase>.parquet` and `index/overpasses_<phase>.parquet`). The v2 tiles and
everything reading them (H, I, J4 with Florida-trained runs on v2) are unchanged. Not built yet: the
v3 loader (needed before J4 on BioGO-SHIP and the stage-2 baseline) and MUR tiling into catalog_v1.
Check the 2 GB write test before each step (storage pool).

*K1. Trial on Florida (3 granules per product, raw kept):*
```bash
git pull
sbatch --account=191001-366746 --export=ALL,STEPS=tiles3,PER_PRODUCT=3 cluster/prep.sbatch
```
Paste back `tail -40 logs/oefm_prep_<id>.out` (seconds per granule, tiles and overpasses per
product, clipped values) and:
```bash
du -sh data/processed/global/tiles/v3/*/ ; du -ch data/processed/global/tiles/v3/*/*/*.zarr.zip | tail -1
```
I extrapolate the full Florida size and time from it before K2.

*K1 result (job 116488, 2026-11-24):* 3 granules per product, all done, none failed. See the
decision log "Tiling v3" for the numbers. Extrapolated to all Florida granules: ~3.0 TB of v3 tiles
vs ~4.0 TB raw. PACE AOP tiles are 2.2x their raw size, so K2 waits for the size breakdown.

*K1b. Size cuts 1-4 on the same granules (2026-11-24, your decision; decision log "Tiling v3").*
The re-tile with the new rules (FORCE=1) was submitted 2026-11-24, after the first breakdown script
ran > 20 min without output (it recompressed every tile; now it reads only the zip directories, in
seconds). K1 timing: 24/24 done, 16.1 s per granule wall (12 workers). The forced re-tile replaced
the K1 stores, so the before-per-field split is gone; the before totals are the K1 `du` above.
When the job is done:
```bash
git pull
python scripts/v3_size_breakdown.py
du -sh data/processed/global/tiles/v3/*/ ; grep "s per granule" logs/oefm_prep_<id>.out
```
Paste back all three. The header line of each product says which rules its stores were written with;
if SFREFL says "SFREFL empty CLDICE+LAND", the job ran the code before the cloud change and step K1c.1
re-tiles it.

*K1b result (job 116510, 2026-11-24):* 24/24 done with the size rules and SFREFL clouds kept
(stores say "SFREFL empty LAND", so K1c.1 is not needed). Florida ~2.3 TB (was ~3.0 TB) but
59.5 s per granule wall (was 16.1 s): zstd level 9. All of Florida at that rate: ~400 h on 12
workers. Numbers in the decision log "Tiling v3".

*K1d. zstd level (login node, read-only, ~1-3 min):*
```bash
git pull
python scripts/v3_codec_bench.py
```
Paste back the table. I set `processing.v3_store.zstd_level` from it (every level is lossless).

*K1c. Before deleting raw at scale (after the K1b job has finished; do not `git pull` while a
tiles3 job runs, because its worker processes re-read the code):*
1. **(Not needed: job 116510 already ran with this rule.) SFREFL keeps cloud pixels (2026-11-24, your decision):** re-tile the trial's SFREFL granules
   with the new rule:
   ```bash
   git pull
   sbatch --account=191001-366746 --export=ALL,STEPS=tiles3,PER_PRODUCT=3,FORCE=1,PRODUCTS=pace_oci_sfrefl cluster/prep.sbatch
   ```
   Then rerun the breakdown and `du` as in K1b step 3.
2. **Fill-value check on more granules** (login node, read-only): Florida with 8 granules per
   product, plus the Bio-GO-SHIP files already downloaded (other latitudes):
   ```bash
   python scripts/flag_fill_check.py --per-product 8 --out data/reports/flag_fill_check_8.csv 2>&1 | tail -80
   OEFM_CONFIG=config/biogoship.yaml OEFM_DATA_ROOT=$PWD/data_biogoship python scripts/flag_fill_check.py --per-product 4 --out data_biogoship/reports/flag_fill_check.csv 2>&1 | tail -80
   ```
3. **test_ddp on Ptolemy** (two CPU processes over gloo, under a minute; it crashes locally from an OpenMP clash):
   ```bash
   python -m unittest discover -s tests -p test_ddp.py -v 2>&1 | tail -15
   ```
Paste back all of it.

*K2 (after review). Re-tile all Florida raw (raw kept until verified):*
`sbatch --account=191001-366746 --export=ALL,STEPS=tiles3 cluster/prep.sbatch`, then `purge-raw`
(report first, `--yes` after review) deletes the Florida raw satellite files.

*K3. PACE BGC for Florida (download → tile → delete, ~16 GB transient):*
```bash
sbatch --account=191001-366746 --export=ALL,PRODUCTS=pace_oci_bgc cluster/inventory.sbatch
sbatch --account=191001-366746 --export=ALL,PRODUCTS=pace_oci_bgc cluster/stream.sbatch
```

*K4. Bio-GO-SHIP via download-tile-delete (replaces J3).* The manifest already lists every product
(incl. SFREFL); add BGC, then stream. The 101 GB already downloaded is tiled and deleted first.
```bash
sbatch --account=191001-366746 --export=ALL,OEFM_CONFIG=config/biogoship.yaml,OEFM_DATA_ROOT=$PWD/data_biogoship,PRODUCTS=pace_oci_bgc cluster/inventory.sbatch
sbatch --account=191001-366746 --export=ALL,OEFM_CONFIG=config/biogoship.yaml,OEFM_DATA_ROOT=$PWD/data_biogoship,CHAIN=5 cluster/stream.sbatch
```

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
| 1.7 | Re-fetch the 8 MUR files that failed md5 verification | ✅ (all 3,290 MUR days cropped in prep) |

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

### 1.7 Re-fetch 8 MUR files ✅ (all 3,290 MUR days cropped in prep v2)
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

### Running 2.0–2.5a on Ptolemy

**P.0, once (new dependency `zarr`; rerun before 2.10 because `scikit-learn` was added for the probes):**
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
Trial ✅ (job 116101: 36 tiled, 4 empty, 0.6 s per granule on 12 workers).
**Then the full run** (the trial's outputs are reused; it continues where the trial stopped):
```bash
git pull
sbatch --account=191001-366746 cluster/prep.sbatch
```
Runs layout → grid → crop → tiles → examples → evalcover → physics → sequences. 24,335 granules at
0.6 s on 12 workers ≈ 4 h; crops and physics are minutes to an hour. Resumable: resubmit the same
command after a crash or the 48 h limit. Only `data/processed/` is written. Expected size:
~200 GB of tiles (OLCI ~31 MB per granule dominates), ~15 GB of physical crops, ~3 GB of physics.
If you already started a full run before pulling 2.5, add the new steps afterwards with
`sbatch --account=191001-366746 --export=ALL,STEPS="evalcover physics sequences" cluster/prep.sbatch`.
Run 1.7 (MUR re-fetch) first if you can; otherwise the 8 missing MUR days are cropped on a later re-run.

Outputs, under `data/processed/regional/`:
- `processing.sqlite`: every output with its raw sources (sha256), git commit, parameters, transformations
- `physical/<product>/<year>/*.nc`: cropped physical files, values and encoding unchanged
- `tiles/v2/grid/`: `tiles.csv` (272 tiles, fixed IDs like `X+05Y-07`), `static.zarr.zip` (lon/lat/GEBCO depth per cell), `grid.json`
- `tiles/v2/<product>/<year>/<granule>.zarr.zip`: one per overpass (layout in `oefm/tiling.py`)
- `tiles/v2/index/tile_observations.parquet` (the loader's entry point), `summary_by_product.csv`,
  `eval_coverage.csv` and `eval_coverage_summary.csv` (2.4b), `sequence_summary.csv` (2.5)
- `tiles/v2/physics/physics_daily.zarr`: daily physical fields on an 8 × 8 coarse grid (14.4 km) spanning 3 × 3 tiles around each tile (2.5a)
- `tiles/v2/examples/*.png` with `*_panels.csv` (2.4)

### After prep.sbatch: 2.6 nutrient targets and 2.7 evaluation sets ▶️ (step A of "What to run now")

Run these once the full `prep.sbatch` has finished (it writes the tile index and `eval_coverage.csv`).

**1. Copy the evaluation targets from the laptop** (built 2026-11-12 with
`python scripts/make_eval_targets.py --root ..`; they hold labels, so they stay out of the repo):
```bash
# on your laptop, from multimodal_seusmbol/
ssh hmp278@ptolemy-login-2.arc.msstate.edu mkdir -p /scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model/data/processed/regional/eval
scp eval_targets/* hmp278@ptolemy-login-2.arc.msstate.edu:/scratch/ptolemy/users/hmp278/ocean_ecology_foundation_model/data/processed/regional/eval/
```
What is in them (all fitted on training labels only; label split per `training.partitions.labels`):

| File | Samples (train / test) | Targets |
|---|---|---|
| `edna_16S.csv.gz` | 913 (730 / 183) | NMF k = 5, 10, 15, 20 (refitted on 2018–2023 only; 5,152 ASVs pass the ≥ 2% training-prevalence filter); 10 Hellinger PCs; 20 dominant genera. Covariates, not targets (since 2026-11-20): `seq_element_frac`, `seq_log10_input_reads` (used by the probes), `seq_n_replicates`, `seq_log10_table_reads`, `seq_platform`, `seq_runs` |
| `edna_18S.csv.gz` | 913 (730 / 183) | same; 7,148 ASVs kept |
| `sfer.csv.gz` | 3,174 (2,709 / 465) | PO₄, NO₃, Si, fluorometric Chl |
| `fish_crcp.csv.gz` | 3,018 (1,961 / 1,057) | presence and log1p count of the 20 most prevalent training species |
| `fish_maxn.csv.gz` | 3,805 (3,189 / 616) | MaxN of the 10 Granite species (month-precision dates) |

Plus `*_nmf_k*_components.csv.gz` (ASV loadings) and `fit_info.json` (settings, kept ASVs, genera, species).

**2. Nutrient targets and the evaluation join** (a download job: nothing else may be downloading, including 1.7):
```bash
git pull
squeue -u hmp278
sbatch --account=191001-366746 cluster/targets.sbatch
```
Runs: inventory of `pisces_nutrients` (daily surface PO₄/NO₃/Si, reanalysis 202406 to 2026-08-31,
then the forecast product 202311) → download (monthly domain subsets, a few GB) → `oefm targets`
(per tile and day, log + z-score from the training partition) → `oefm evalsets`.
Outputs: `tiles/v2/targets/pisces_daily.zarr`; `processed/regional/eval/matched/*.csv.gz` and
`matched/summary.csv` (per dataset × label split: samples, with usable optical data, with ≥ 1 and ≥ 2
usable PACE observations, years).
Paste back: `tail -40 logs/oefm_targets_<id>.out` (includes the summary table).

**3. Then Stage 1 (2.8+):** see "Running Stage 1 on Ptolemy" below.

### 2-I Data preparation

| # | Step | Needs | Runs on | Status |
|---|---|---|---|---|
| P.0 | Rebuild the Python environment (adds `zarr`) | – | service | ✅ |
| 2.0 | `data/processed/regional/` and `data/processed/global/` (global stays empty), plus a processing manifest (source files, code version, subset, transformations) | – | service | ✅ prep v2 (116124) |
| 2.1 | Final QC policy: hard rejects per sensor, HISATZEN as a tag (not for MODIS), 665 nm angle weights from 45°, a strict subset for spectral evaluation, SFREFL rules kept separate | 1.5 reviewed ✅ | – | ✅ in `config/pilot.yaml` (`processing.qc`), applied by 2.3 |
| 2.2 | Regional crops of MUR and the other physical fields, with a context margin (raw global files kept) | 1.4 | service | ✅ prep v2 (116124) |
| 2.3 | Tile grid (38.4 km tiles, 32×32 at 1.2 km, OLCI 128×128 at 300 m, local equal-area projection, fixed tile IDs) and per-overpass gridding to Zarr | 2.1 | service | ✅ prep v2 (116124) |
| 2.4 | Visual check: a few multisensor tile examples | 2.3 | service | ✅ prep v2 (116124) |
| 2.4b | Which evaluation samples have usable PACE (and OLCI/MODIS/VIIRS) pixels, per dataset and year | 2.3 | service | ✅ prep v2 (116124) |
| 2.5a | Physics cube: daily SST, SST error, wind u/v/speed, precipitation, ADT, SLA, geostrophic u/v, surface current u/v on a coarse grid around every tile; GEBCO depth static | 2.2 | service | ✅ prep v2 (116124) |
| 2.5 | Loader (`oefm/loader.py`): causal 30 d / retro ±15 d, physics lookback 90 d, lengths 8/16/24, five masking types; temporal, geographic and label partitions, sensor set and PACE visibility all configurable without rebuilding (config `training`) | 2.3 | service | ✅ built and tested; `sequences` reports what each split contains |
| 2.6 | PISCES nutrient targets (PO₄, NO₃, Si), regridded to the tiles **as targets only**, never encoder inputs: download (config `targets.pisces_nutrients`), then `oefm targets` | 2.3 | service | ✅ (116167) |
| 2.7 | Evaluation datasets, each matched to tiles and times, with grouped splits fixed before training (see [docs/evaluation_inventory.md](docs/evaluation_inventory.md)): Florida 16S/18S on the shelf and Keys (NMF components, dominant taxa, composition), SFER nutrients and fluorometric Chl, reef fish (CRCP, video MaxN). Label splits from `training.partitions.labels`; **target preprocessing (Hellinger, NMF, feature selection) refitted on training labels only** (the existing NMF exports used all samples, so they are not reused for the test). Sargassum and HPLC are optional and must not delay Stage 1. Targets built on the laptop (`scripts/make_eval_targets.py`), joined on Ptolemy (`oefm evalsets`) | 2.3 | laptop + service | ✅ (116167) |

### 2-II Stage 1: choose the optical objective (optical inputs only, matched data and capacity)

| # | Step | Runs on | Status |
|---|---|---|---|
| 2.8a | Frozen training statistics: reflectance transform curves and the PACE heritage ridge (`oefm normstats`, last step of `prep.sbatch`) | service | ✅ prep v2 (116124) |
| 2.8b | GPU environment (`setup_gpu_env.sbatch`), then `gpu_check.sbatch` | service, gpu-a100 | ✅ |
| 2.8c | Benchmarks: arms × sequence length 8/16/24 × per-GPU batch (`benchmark.sbatch`) | gpu-a100 | ✅ L16, batch 32 |
| 2.9 | Train **A** masked reconstruction, **B** JEPA + SIGReg, **C** B + PACE spectral residual (`train.sbatch`, same steps for each) | gpu-a100 | ✅ all three at step 30,000 |
| 2.10 | Compare: collapse checks and effective rank, PACE spectral retention (frozen encoder + identical decoder, error by wavelength), frozen-embedding probes on 2.7, training behaviour | gpu-a100 | ✅ 116376 (SFER probes rerun: F2 ✅) |
| 2.10b | Robustness: test-year rotation, bootstrap intervals, PACE-era visible vs dropped, input retention, amplitude test on trained weights (`--steps robust amplitude`, `oefm/model/robust.py`) | gpu-a100 | ✅ G 116439: B/C fail amplitude |
| 2.10c | Focused tests (`--steps focused`, `oefm/model/focused.py`): PACE-era phytoplankton groups vs NASA MOANA (`oefm/moana.py`), ±1/±3 d windows; fish added value over habitat (Poisson ridge, 90-day seasonal window) | gpu-a100 (+ `moana` on service) | ✅ G0, G (causal/retro); all windows in I |
| 2.9b | **Stage 1.5, arm D** (design document fork 2, hybrid JEPA + light reconstruction): arm C + arm A's band decoder at ν = 0.1 and 0.3, + amplitude term κ = 0.1 | gpu-a100 | ▶️ H, then evaluate all five (I) |
| 2.11 | **Pick the Stage 1 winner** (reconstruction is kept if it beats JEPA). You decide; recorded in the decision log | – | ⏳ after G–I (A leads so far) |

### Running Stage 1 on Ptolemy (2.8–2.9)

Model: `oefm/model/` (design document v1: "Ocean Foundation Model Design Decisions.md"); settings
`config/train_stage1.yaml`. ~41 M parameters per arm (A 40.1 M, B 41.4 M, C 41.8 M; D = C + A's
decoder and its own query embeddings). Tested locally on
CPU with the smoke tiles (all three arms train, resume and benchmark); not yet run on GPUs.

**0a. ✅ Done (116106 → 116124). If prep stopped with "database or disk is full" (job 116106):** check quotas with errors visible,
free the superseded data, compact the manifest, resubmit:
```bash
git pull
mmlsquota -u hmp278 --block-size auto; mmlsquota -g ngi --block-size auto     # paste these
rm -rf data/processed/regional/tiles/v1 data_smoke                             # superseded (v1) and the step-1.1 test copy
source cluster/env.sh && python -m oefm processed --prune-version v1           # removes v1 records, VACUUM
sbatch --account=191001-366746 cluster/prep.sbatch                             # continues where it stopped
```

**0. ✅ Done (job 116124). Re-run the data preparation (tiles are now v2, see the decision log):**
```bash
git pull
squeue -u hmp278                                   # cancel a still-running prep job first: scancel <id>
sbatch --account=191001-366746 cluster/prep.sbatch
```
The physical crops are reused; tiles, grid, index, coverage, physics, sequences and the new
`normstats` step (2.8a) are rebuilt under `data/processed/regional/tiles/v2/` (~4–5 h). When it has
finished, the old `tiles/v1/` can be deleted (`rm -rf data/processed/regional/tiles/v1`). Then rerun
`targets.sbatch` with `STEPS="build evalsets"` if it already ran (the PISCES download is reused).

**1. GPU environment, once** (Python 3.12 + PyTorch 2.6 / CUDA 12.4 + all of `requirements.txt`, so
the same environment also extracts embeddings and runs the probes; the Python 3.14 environment stays
for the data pipeline):
```bash
sbatch --account=191001-366746 cluster/setup_gpu_env.sbatch
# when it has finished:
sbatch --account=191001-366746 cluster/gpu_check.sbatch     # a few minutes on 4 A100s
```
Paste back `tail -30 logs/oefm_gpucheck_<id>.out`: driver, bf16 attention, NCCL all-reduce (expect 10),
one tiny benchmark per arm.

**2. Benchmarks** (design document "Required benchmarks"; 27 configurations × 200 steps, several hours):
```bash
sbatch --account=191001-366746 cluster/benchmark.sbatch
```
Paste back the summary table at the end of `logs/oefm_bench_<id>.out` (throughput, peak memory,
loader wait, projected hours per arm). I then set `per_gpu_batch`, `length` and `steps` (equal for
all arms) in `config/train_stage1.yaml`.

**3. Training, one job chain per arm** (after the benchmark values are committed):
```bash
git pull
sbatch --account=191001-366746 --export=ALL,ARM=A,CHAIN=1 cluster/train.sbatch
sbatch --account=191001-366746 --export=ALL,ARM=B,CHAIN=1 cluster/train.sbatch
sbatch --account=191001-366746 --export=ALL,ARM=C,CHAIN=1 cluster/train.sbatch
```
(4 GPUs each; the QOS allows 4 A100s per user, so they queue one after another.) Each job saves before
48 h and the chain resumes from `data/runs/stage1_<ARM>/checkpoint.pt`. Paste back
`tail -5 data/runs/stage1_<ARM>/log.jsonl` now and then (losses per modality, SIGReg, effective rank).

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
| 3.1 | **Bio-GO-SHIP corridors:** `config/biogoship.yaml` (50 zones, 377 boxes, ~13,400 tiles), zoned grids and tiling (`oefm/grid.py`, `oefm/tiling.py`) | ✅ built; J1–J3 ▶️ |
| 3.4 | **Bio-GO-SHIP metagenome benchmark** vs Granite (`oefm/model/biogoship.py`): reproduction check, paired benchmark on the 1,215 tiles with Granite rerun on the same tiles, PACE-era subset, extended set (1,977 samples, HiSeq as a third platform level), per-KO and per-cruise comparison | ✅ built; J4 after J3 |
| 3.5 | Retrain the 2.11 winner on Florida + corridors, rerun 3.4 | 🛠 after 2.11 |

Storage rule (Phase 2 storage revision): `data/raw/` is the immutable archive shared by every
dataset. Regional and global processed data live in separate directories, and each records the raw
files, code version, subset and transformations it was built from, so it can be regenerated
without re-downloading.

---

## Decision log

### Tiling v3: download, tile, delete; fixed global catalog (2026-11-24, agreed, being built)
**Your decision.** Raw granules are deleted once their tiles are verified, so tiles keep everything
that would otherwise need a redownload; design choices (flag policy, transform, normalisation,
windows, sampling) move to the loader.
- **Coverage:** every tile a downloaded granule touches (full swath), full resolution. Tiles under
  1 % valid cells store no arrays; every overpass is still recorded (an overpass index separate from
  the stored-tile index, so the loader never reads the large one). If storage runs short, delete
  whole tiles, never resolution.
- **catalog_v1 (fixed, versioned, definition stored with the tiles):** 15° latitude bands split
  into ~1,700 km zones, each its own LAEA (< 0.5 % distortion), ~180 zones plus polar caps; tile ID
  `<zone>:X…Y…` is permanent. Border tiles belong to both zones; a point (and every label matchup)
  uses the tile of its own zone, chosen from its lat/lon. The Gulf crosses 30° N, so the pilot
  exercises borders.
- **Per cell:** Rrs (all bands) = mean over pixels with valid Rrs, no flag mask, re-quantised to the
  granule's own int16 scale/offset (≤ half a source step; float sources stay float); l2_flags OR plus
  the share of pixels with each of the 32 bits; a clean mean (each product's current training mask)
  only in mixed cells; sun zenith/azimuth (computed), sensor zenith/azimuth (reconstructed); time;
  pixel count; FLH, chlor_a, Kd_490 where present; PACE SFREFL rhos; PACE_OCI_L2_BGC chlorophyll and
  carbon/pigment fields (product to add). OLCI at 300 m only (1.2 km built in the loader).
- **Gridded products:** MUR (and other gridded fields) tiled into catalog_v1, global daily files
  then deleted.
- **First check:** `scripts/flag_fill_check.py`: which flag bits co-occur with fill Rrs in standard
  L2 files (if HISATZEN / STRAYLIGHT / COCCOLITH pixels are filled, those policy choices cannot be
  applied from these files).
- **Order:** build + test; re-tile Florida raw into v3; keep the v2 tiles until stage 1, 1.5 and
  2.11 are done; delete Florida raw only after v3 is verified; Bio-GO-SHIP and later regions via
  download-tile-delete (tiles only). Before stage 2, retrain the stage-1.5 winner on v3 as the
  stage-2 baseline (no mixing of v2 and v3 tiles).
- **Flag/fill check (2026-11-24, 2 Florida granules per product, every 4th scan line).** Rrs is
  fill in every band for 100 % of pixels with ATMFAIL, CLDICE, LAND, HILT, NAVFAIL (OLCI),
  BOWTIEDEL (VIIRS), SEAICE, and 99.7 % with HISOLZEN (VIIRS-J1): these are baked in by NASA's
  processing, and all are hard-mask flags already. Pixels with HISATZEN (4–12 % valid), STRAYLIGHT
  (15–81 % valid), HIGLINT/MODGLINT (9–50 % valid), COASTZ, HIPOL, ATMWARN, LOWLW, MAXAERITER, CHLFAIL
  and COCCOLITH (≥ 99.8 % valid) keep Rrs values, so those parts of the policy can be deferred (the
  co-occurrence with fill reflects coincident clouds/land, not filling: none reaches 100 %). Pixels with
  no bit set are never fill. Data types: every Rrs (incl. PACE) is int16, scale 2e-6, offset 0.05;
  Kd_490, nflh, aot, angstrom, pic, poc, par int16; chlor_a and PACE avw float32; PACE Rrs_unc int16.
  PACE AOP has nflh, aot, angstrom, avw, Rrs_unc but no chlor_a / Kd_490 (BGC product). PACE SFREFL
  rhos is float32 and kept nearly everywhere (fill only with HILT 70 %; CLDICE 3 %): cloudy cells do
  not compress away, so full-swath SFREFL tiles are about raw size unless quantised or cloud-masked.
  Tables: data/reports/flag_fill_check*.csv.
- Blocked meanwhile: the /scratch data pool returns ENOSPC for large writes (2026-11-24; reads work).
- **K1 trial (job 116488, 2026-11-24, 3 Florida granules per product, raw kept).** All 24 done,
  0 failed, 16.1 s per granule wall with 12 workers (all of Florida, 24,335 granules: ~109 h); SFREFL clipped 1,142 cell-band values to the int16 range (rhos outside -0.055..1.255;
  about 2e-6 of the stored values). Stored tiles / overpasses (tiles under 1 % valid not stored):

  | product | stored tiles | overpasses | MB per granule (tiles) | MB per granule (raw) | tiles / raw | KB per tile | Florida extrapolated |
  |---|---|---|---|---|---|---|---|
  | modisa_oc | 3,410 | 10,428 | 34 | 34 | 1.00 | 30 | 140 GB (4,166 granules) |
  | olcia_efr_oc | 1,932 | 3,573 | 207 | 243 | 0.85 | 321 | 460 GB (2,221) |
  | olcib_efr_oc | 2,200 | 3,567 | 244 | 249 | 0.98 | 333 | 520 GB (2,129) |
  | pace_oci_aop | 4,126 | 12,235 | 465 | 215 | 2.17 | 338 | 826 GB (1,776) |
  | pace_oci_sfrefl | 4,671 | 12,235 | 167 | 760 | 0.22 | 107 | 296 GB (1,776) |
  | viirsj1_oc | 8,103 | 15,827 | 66 | 88 | 0.75 | 24 | 326 GB (4,930) |
  | viirsj2_oc | 7,943 | 15,841 | 63 | 76 | 0.83 | 24 | 155 GB (2,451) |
  | viirsn_oc | 8,268 | 15,914 | 67 | 84 | 0.80 | 24 | 326 GB (4,886) |
  | **total** | | | | | | | **~3.0 TB (raw ~4.0 TB)** |

  Trial total 3.7 GB. Granule counts are the v2 tiling counts (done + empty); raw MB per granule
  comes from the Florida inventory; 3 granules per product, so expect roughly ±30 %. The earlier
  estimate (1.1–1.7 TB) was about 2x too low. Except for SFREFL, tiles take about the raw size,
  so download-tile-delete saves ~25 % overall. The same ratios applied to Bio-GO-SHIP (~10.2 TB raw
  without SFREFL, 7.6 TB SFREFL) give ~12–14 TB of tiles. Next: per-array breakdown
  (`scripts/v3_size_breakdown.py`, step K1b) before K2.
- **Size cuts (2026-11-24, your decision: they lose nothing the model or the evaluations use).**
  1. Single-layer fields: only `processing.v3_store.extras`, i.e. nflh, chlor_a and Kd_490 (baselines;
     PACE AOP has only nflh), and for PACE BGC chlor_a, carbon_phyto and poc. Check this list against
     the BGC file at K3. Every other field the file had is named in the store attrs
     (`fields_not_stored`).
  2. flag_share (already uint8 x 255) only for the 15 bits a policy can use
     (`processing.v3_store.flag_share_bits`: HISATZEN, STRAYLIGHT, HIGLINT, MODGLINT, HIPOL, COASTZ,
     ATMWARN, MAXAERITER, LOWLW, ABSAER, COCCOLITH, TURBIDW, OPSHAL, CLDICE, LAND); bits that always
     come with fill Rrs (ATMFAIL, HILT, NAVFAIL, BOWTIEDEL, SEAICE, HISOLZEN) and unused ones are dropped.
     flags_or keeps the full 32-bit word (uint32). Read with `tiling3.flag_share_of(group, name)`.
  3. Storage: zstd level 9, per array the better of byte shuffle and bit shuffle, chunks of whole
     tiles up to ~1 MB uncompressed. Lossless.
  4. Sun zenith/azimuth not stored; the loader computes them from tile_time + dt and the cell centres
     (`tiling3.cell_solar_angles`). Sensor angles stay int16 in 0.01°.
  Synthetic tiles: the prediction from old stores (45 % of the old size) matched the new stores.
- **SFREFL keeps cloud pixels (2026-11-24, your decision).** Cloud-flagged pixels are kept, and the
  loader decides from the CLDICE flag share, as for Rrs: dense Sargassum can trip the cloud mask, and
  SFREFL is there mainly for floating seaweed. Dropping them at tiling would be irreversible once raw
  is deleted. Land pixels stay dropped. The int16 range widens to scale 4e-5, offset 1.2
  (-0.11..2.51; half a step 2e-5, well below the sensor noise), so bright cloud is not clipped. If
  storage becomes a concern, decide then.
- **K1b, size rules + SFREFL clouds kept (job 116510, 2026-11-24, same 3 granules per product).**

  | product | MB per granule K1 → K1b | x | tiles / raw | Florida | largest fields (share of bytes) |
  |---|---|---|---|---|---|
  | modisa_oc | 34 → 17.0 | 0.50 | 0.51 | 71 GB | refl 0.49, chlor_a 0.12, refl_clean 0.08 |
  | olcia_efr_oc | 207 → 134.4 | 0.65 | 0.55 | 299 GB | refl 0.61, chlor_a 0.13, refl_clean 0.10 |
  | olcib_efr_oc | 244 → 163.0 | 0.67 | 0.66 | 347 GB | refl 0.63, chlor_a 0.13, refl_clean 0.08 |
  | pace_oci_aop | 465 → 369.8 | 0.80 | 1.72 | 657 GB | Rrs_unc 0.47, refl 0.46, refl_clean 0.05 |
  | pace_oci_sfrefl | 167 → 309.2 (clouds now kept) | 1.85 | 0.41 | 549 GB | refl 0.97 |
  | viirsj1/j2/n | 66/63/67 → 30.0/29.4/30.1 | 0.45–0.47 | 0.34–0.39 | 148/72/147 GB | refl 0.39–0.40, chlor_a 0.18–0.19, flag_share 0.06–0.08 |
  | **total** | | | | **~2.3 TB (K1 3.0 TB; raw 4.0 TB)** | |

  SFREFL stored tiles doubled (4,671 → 9,292) because cloudy tiles now pass the 1 % rule.
  Bio-GO-SHIP at these ratios: ~6.8 TB without SFREFL + ~3.1 TB SFREFL ≈ 10 TB.
  Timing 59.5 s per granule wall (K1 16.1 s), from zstd level 9: level now a setting
  (`processing.v3_store.zstd_level`), chosen from `scripts/v3_codec_bench.py` (K1d). Compression
  now runs single-threaded inside each worker (blosc threads off).

### Bio-GO-SHIP as a global downstream benchmark (2026-11-23)
**Your request.** Use the Bio-GO-SHIP metagenome data (instructions from the Granite project
agent, `multimodal_seusmbol/instructions_agent_biogoship.md`) as a robust downstream test, with
Granite as the baseline, the same protocol as the Granite work, and training tiles in the same
general regions.
**Your choices (2026-11-23).** Imagery: each cruise's own window plus two 45-day windows per
corridor in other seasons (estimated 8–10 TB raw). Order: H trains on Florida as planned (2.11
compares arms on the same data); the corridors download in parallel; the 2.11 winner is then
retrained on Florida + corridors.
**Built.**
- Zones: `config.zones` (LAEA centre each), `Region.zone`, per-region `windows`, `extends`
  (inherit pilot.yaml: products, processing, training). `TileGrid(cfg, zone)`, tile IDs
  `<zone>:X+03Y-12`, longitudes wrapped across ±180 (P02-2022, HLY2202-2022 cross the
  antimeridian; HLY2202 reaches the pole). Inventory searches each box only over its own windows.
  Tiling writes one store per granule and zone. The loader looks a point up in every zone.
  Florida is unchanged (same IDs, same geographic blocks; tests).
- Zones are ≤ 1,500 km from their centre (LAEA scale error < 1 %); boxes ≤ 300 km and ≤ 7 days of
  samples, margin 0.3°, never across the antimeridian. Boxes poleward of 85° span all longitudes.
- Imagery at the sample dates enters pretraining (as for Florida: the hold-out is for labels, never
  imagery). The benchmark's leave-one-cruise-out applies to the read-out.
- Tile rule vs Granite (to report with results): Granite = one OLCI 42 × 42-pixel (12.6 km) scene,
  closest clear overpass (median 0.59 d), clear window moved up to 20 km. Here = the 3 × 3 cells
  (3.6 km) at the sample in a 38.4 km tile, up to 16 observations from all sensors within the
  causal 30 d (or ±15 d) window; no relocation. One feature vector per benchmark tile = the mean
  of its samples' embeddings.
- Extended set: samples grouped by the model's input (same cruise, 1.2 km cell, UTC day,
  platform); HiSeq 4000 is a third platform level (pre-2016 cruises and some 2016 samples).
**Not available yet.** P04E-2026 (279 samples, PACE-only cruise): needs the KO pipeline first (ask).
The module-predictability figure (`granite/ko_ceiling.csv`) is not recomputed yet.

### G results (job 116439, 2026-11-23): B/C lose amplitude; arm D gets an amplitude term
**Amplitude test on trained weights** (design document: log-amplitude R² ≥ 0.95, relative distance > 0.05):

| Level | Metric | A | B | C |
|---|---|---|---|---|
| E1 | log-amplitude R² | 0.926 FAIL | 0.985 | 0.984 |
| E1 | log blue/green R² | −1.549 | 0.916 | 0.921 |
| E1 | rel. distance ×0.5 | 0.388 | 0.002 FAIL | 0.001 FAIL |
| E1 | rel. distance ×2 | 0.278 | 0.003 FAIL | 0.001 FAIL |
| token | log-amplitude R² | 0.988 | 0.987 | 0.987 |
| token | log blue/green R² | 0.909 | 0.908 | 0.909 |
| token | rel. distance ×0.5 | 0.865 | 0.034 FAIL | 0.034 FAIL |
| token | rel. distance ×2 | 0.860 | 0.052 | 0.052 |

- B/C keep amplitude only in a direction of ~0.1–0.3 % of the E1 norm; already the tokenizer
  output moves only 3–5 % (A 86 %). The JEPA target is encoded by the same tokenizer, so dropping
  brightness makes targets easier to predict.
- On real samples it is an information loss, not only a scale: retention (CV R², causal) of log
  brightness A 0.981, B 0.296, C 0.316 (meta 0.379); band 443 A 0.971, B 0.124, C 0.178; log
  blue/green A 0.990, B 0.700, C 0.716. A's synthetic E1 failures (0.926; blue/green −1.55) are
  on uniform synthetic scenes; on real samples A retains both at ≥ 0.98.
- **Fix in the hybrid (review: "if either failed, fix it in the hybrid too"):** arm D's
  reconstruction term already needs amplitude; in addition D gets `amp_loss` at κ = 0.1 (linear
  head D → 2 on the context tokens, target the tokenizer's own two amplitude features per cell).
  `loss.kappa=0` turns it off. Caveat: D's brightness retention is then partly by construction;
  the downstream probes and the distance part of the amplitude test are the independent checks.

**Test-year rotation** (mean over test years of the family median R²; [years better than A/e1]):

| Target | A/e1 | B/e1 | C/e1 | best other |
|---|---|---|---|---|
| 16S NMF k5 causal | 0.705 | 0.575 | 0.608 [1/5] | – |
| 16S phyto CLR causal | 0.584 | 0.421 | 0.424 | seq 0.463 |
| 18S NMF k5 causal | 0.553 | 0.491 [1/5] | 0.513 [2/5] | B/e2 0.531 [3/5], C/e2 0.539 [3/5] |
| 18S phyto CLR causal | 0.365 | 0.241 | 0.249 | seq 0.245 |
| SFER Chl causal | 0.702 | 0.458 | 0.462 | raw 0.661 |
| SFER Si causal | 0.340 | 0.337 [4/8] | 0.339 [4/8] | C/e2 0.344 [4/8] |
| SFER PO4 causal | −0.025 | −0.074 | −0.073 | B/e2 0.057, C/e2 0.055 [8/8] |
| CRCP log1p causal | −0.287 | −0.067 | −0.088 | C/e2 −0.034 |
| MaxN causal | 0.083 | 0.018 | 0.038 | A/e3 0.084 |

- Sequencing covariates alone (`seq`) explain the phytoplankton CLR about as well as B/C
  (16S 0.463, 18S 0.245): platform and depth shift group shares.
- The stage 1 Si advantage of B/C is gone over years (≈ A). PO4 B/C E2 wins 8/8 but at R² ≈ 0.05.
- Official split bootstrap (95 %): SFER Chl A 0.800 (0.77, 0.83), B 0.573 (0.52, 0.63), C 0.601
  (0.54, 0.65), raw 0.740 (0.70, 0.78); 16S NMF k5 A 0.711 (0.64, 0.78), C 0.745 (0.67, 0.79).
- CRCP intervals were degenerate (e.g. 0.016 (0.02, 0.02)): CRCP groups were primary sample units,
  which are random per survey (3,018 samples, 2,406 units), so every group had one sample. Fixed:
  0.1° blocks, 62 groups of median 30 (`group_of`). MaxN PACE-era R² differences (−59 … +9) come
  from near-zero variance in small month groups; uninterpretable, superseded by focused fish.

**PACE-era visible − dropped (probes trained on PACE-era samples):** every E1 interval includes 0
except negative ones (16S phyto causal B/e3 −0.197 (−0.32, −0.06); 18S phyto causal A/e1 −0.179
(−0.34, −0.02)). No arm shows a PACE gain.

**Focused phyto** (PACE-era, leave-one-cruise-out, 3 cruises; MOANA ±3 d for causal, ±1 d for
retro; n = 46 causal, 50 retro after requiring PACE and a MOANA matchup): median R² over groups

| Features | 16S causal | 16S retro | 18S causal | 18S retro |
|---|---|---|---|---|
| MOANA | 0.333 (0.18, 0.46) | 0.562 (0.40, 0.70) | −0.077 | −0.224 |
| raw visible | 0.065 | −0.008 | −0.306 | −0.475 |
| A/e1 visible | 0.154 (−0.14, 0.34) | 0.113 | −0.655 | −0.363 |
| B/e1 visible | 0.078 | 0.005 | −0.706 | −0.291 |
| C/e1 visible | −0.071 | −0.056 | −0.519 | −0.155 |
| meta | −0.471 | −0.324 | −0.068 | −0.079 |

- MOANA (Pro, Syn, picoeukaryote cells) beats every embedding on 16S (E1 − MOANA: A −0.18
  (−0.51, 0.10), B −0.26 (−0.62, −0.10), C −0.40 (−0.62, −0.25) causal; all three below in retro,
  p_better ≤ 0.0005). Nothing predicts 18S groups across cruises (all ≤ 0; C/e1 retro vs MOANA
  +0.069 (−0.04, 0.27)).
- E1 visible − dropped: point values were NaN (ordering bug, fixed); intervals all include 0
  except A/e1 18S retro (−0.54, −0.03).
- ±1/±3 d embedding windows come in I (they need new embeddings).

**Focused fish** (Poisson ridge, habitat covariates in every model; official split): median over species

| Features | CRCP D² | CRCP D² level-free | CRCP Spearman | MaxN D² | MaxN Δ vs habitat (95 %) |
|---|---|---|---|---|---|
| habitat | 0.097 | 0.162 | 0.401 | 0.069 | 0 |
| meta | 0.150 | 0.195 | 0.477 | 0.093 | +0.024 (−0.001, 0.063) |
| raw | 0.121 | 0.154 | 0.428 | 0.069 | −0.000 (−0.03, 0.03) |
| A/e1 | 0.102 | 0.168 | 0.470 | 0.075 | +0.006 (−0.02, 0.05) |
| B/e1 | 0.123 | 0.194 | 0.416 | 0.100 | +0.030 (0.003, 0.054) |
| C/e1 | 0.131 | 0.189 | 0.424 | 0.099 | +0.030 (0.006, 0.059) |

- Year rotation: CRCP D² is negative for every set (habitat −0.024) but level-free 0.11–0.14 and
  Spearman 0.33–0.38: the stage 1 negative R² is a year-level shift, as the review suspected.
  MaxN rotation: meta 0.134, A 0.099, C 0.089, raw 0.085, B 0.071, habitat 0.054.
- Embeddings add a little over habitat (MaxN B/C +0.03, interval above 0) but not more than
  position and season (meta). CRCP intervals were degenerate (group bug above).

### Second review: what PACE can see, fish over habitat, two reconstruction weights (2026-11-23)
Built after a second pasted review (steps G0, G, H, I). Code: `oefm/model/focused.py`
(`--steps focused`), `oefm/moana.py` (`targets.sbatch STEPS=moana`), the `WINDOWS` registry in
`oefm/model/evaluate.py`, `phyto_*`/`hab_*` columns in `scripts/make_eval_targets.py`.

- **Phytoplankton targets, not whole-community NMF.** CLR of group read sums (all ASVs, 0.5-read
  pseudocount, one 'other' part). Mean read share over all 913 samples:

| Marker | Group (taxonomy) | Mean share |
|---|---|---|
| 18S | Dinophyceae (Syndiniales parasites are 'other') | 13.4% |
| 18S | Diatoms (Bacillariophyceae, Mediophyceae, Coscinodiscophyceae) | 7.2% |
| 18S | Prasinophytes (Mamiello-, Chloropico-, Pyramimonado-phyceae, Prasino-Clades) | 4.8% |
| 18S | Other Chlorophyta (mostly Trebouxiophyceae, e.g. *Picochlorum*) | 3.3% |
| 18S | Haptophytes | 1.5% |
| 16S | Synechococcus (Cyanobiaceae other than Prochlorococcus, incl. Cyanobium) | 19.7% |
| 16S | Plastid 16S (o__Chloroplast) | 9.6% |
| 16S | Prochlorococcus | 3.5% |

  Metazoa are ~40% of 18S reads, which is why NMF k5 mixes in signal optics can reach only
  indirectly. The phyto family also enters the ordinary probes (2018–2023 → 2024).
- **Sample size for the PACE test.** The review's 459 does not match the data. Per marker there
  are 138 PACE-era eDNA samples (all 2024), 100 with usable PACE in the 3 × 3 cells (±15 d), from
  3 cruises; tighter windows and the MOANA requirement reduce this further. Leave-one-cruise-out
  has 3 folds, so the intervals will be wide. Kept small as advised: 5 (18S) or 3 (16S) targets,
  ridge.
- **Tight matchups.** New windows `retro1` and `retro3` (±1 and ±3 d; eDNA and SFER). The eDNA set
  is surface samples only (as built in 2.7). Caveat found while building this: every window uses at
  most 16 observations nearest the sample (the training length), and the pilot area has several
  products per day, so the nominal ±15 d window may already cover only a few days. The embeddings
  now record the span actually covered (`span_days`), and I prints its median.
- **MOANA baseline.** PACE_OCI_L4M_MOANA v3.2 (OB_CLOUD; the only version in CMR on 2026-11-23),
  daily 4 km files: Prochlorococcus, Synechococcus and picoeukaryote cells/mL (Lange et al.
  2020), domain 85 W–25 E. 3 × 3-pixel median per sample and day, log10(1 + cells) averaged over
  the days in ±1 / ±3 d. Every feature set is scored on the samples that also have a MOANA value.
  Bootstrap within cruises of E1 visible − dropped and E1 visible − MOANA.
- **Fish over habitat.** Habitat covariates enter every model unpenalised:
  - CRCP: survey depth (log10) and habitat type (8 types ≥ 2% of training units; reef type ×
    vertical relief);
  - MaxN: GEBCO depth and distance to shore from the Granite manifest.

  No distance-to-reef layer exists in the repo. CRCP underwater visibility is left out (an in-situ
  optical measurement). The baseline is habitat alone; each feature set adds a penalised block.
- **Count models.** Poisson ridge GLM (Newton; checked against sklearn) on counts. CRCP counts per
  survey unit = expm1 of the log1p targets; MaxN as counts. Features standardised and reduced to
  ≤ 50 PCs fitted on the training labels, the same for every feature set. Metrics:
  - D² (deviance explained vs the test mean; penalises a year-level shift, like R²);
  - D² after rescaling the predictions to the test total (`d2_level_free`: separates CRCP's
    level shift from ranking skill; uses one number from the test labels, so it is a diagnostic);
  - Spearman.

  Official split with bootstrap intervals of the difference to habitat, plus test-year rotation.
- **Longer windows for fish.** `seasonal` = the mean of three causal 30 d windows ending at the
  survey, 30 d and 60 d before it (0–90 d). This keeps the encoder within the time offsets it was
  trained on (±15 d); a single 90 d window would use offsets never seen in training.
- **Optically shallow reefs.** Bottom reflectance contaminates Rrs at CRCP sites. This is printed
  as a caveat, and MaxN (deeper, 10–100 m) is the cleaner fish test. **Fish eDNA (12S):** none in
  `eDNA_data_sources/` (only 16S and 18S), so that suggestion needs new data.
- **Stage 1.5 at two weights.** ν = 0.1 (`stage1_D`) and ν = 0.3 (`stage1_D3`,
  `OVERRIDES="loss.nu=0.3"`, `NAME=stage1_D3`); same steps and data as stage 1.
- **Order:** the amplitude test on trained B/C (G) is read before H is submitted, so that a failure
  can be fixed in the hybrid first. Effective rank is fine for every arm (above).

### Stage 1.5 and robustness checks before 2.11 (2026-11-23)
**Proposed.** A review of the Stage 1 results (pasted 2026-11-23) proposed two things before the
2.11 decision:
- run fork 2 (hybrid JEPA + light reconstruction) as a stage 1.5;
- add uncertainty to the comparisons.

**Built**, with all tests passing (43):
- Arm D, as described in G–I of "What to run now".
- `--steps robust amplitude`.

The review's points, and what the data already say:

- **A keeps optical detail; B/C keep coarse community structure.**
  - B/C beat A on Si (0.40–0.41 vs 0.38–0.39) and NMF k5 (16S C/e1 0.745 vs 0.711; 18S C/e2 0.578
    vs 0.489).
  - The Si margins are 0.02–0.03, on 464 samples from 6 cruises.
  - G's bootstrap and test-year rotation show which of these gaps exceed the noise.
- **B/C below raw reflectance on Chl** (0.56–0.65 vs raw 0.74–0.77).
  - **Collapse:** effective rank at the evaluation samples is not lower for B/C (E1: A 115.9, B
    103.2, C 127.1; E2 about 244). That doesn't rule out a loss of *amplitude* (rank can't see it).
  - **Amplitude test:** the design-document test ran only at initialisation, where it passed. The
    "after training" check was never implemented. It now runs on trained weights (`amplitude`),
    at the token and the E1 level. The E1-level probe also reports the log blue/green ratio.
  - **Information loss:** `retention` measures whether E1/E2/E3 still linearly hold what the raw
    input had. It covers each MODIS band, brightness and the OC3M blue/green ratio, on about
    10,000 evaluation samples with no labels.
- **Judge PACE sensitivity on E1, not E2:** agreed. E2 is a MODIS-band query by design. B/C E1
  barely move either:
  - SFER Chl Δ +0.003 / +0.002 (causal);
  - eDNA PACE-usable subset |Δ| ≤ 0.014.
- **PACE value should show in phytoplankton composition (18S), not Chl.** The existing probes can't
  test this. Every eDNA training sample is from 2018–2023, and all 138 PACE-era samples are in the
  2024 test year (100 with usable PACE, from 3 cruises: WS24139, HG24062, WS24205). A probe trained
  without PACE can't use PACE-specific directions. For the record, the PACE-usable subset (n = 100)
  in `probes.csv` gives dropped → visible:

| 18S family | Window | raw | A/e1 | B/e1 | C/e1 |
|---|---|---|---|---|---|
| NMF k5 | causal | 0.334 → 0.352 | 0.557 → 0.561 | 0.462 → 0.457 | 0.522 → 0.525 |
| NMF k10 | causal | 0.091 → 0.118 | 0.392 → 0.395 | 0.180 → 0.180 | 0.172 → 0.171 |
| PCs | causal | 0.028 → −0.033 | 0.271 → 0.297 | 0.072 → 0.072 | 0.089 → 0.089 |
| NMF k5 | retro | 0.355 → 0.286 | 0.567 → 0.557 | 0.466 → 0.452 | 0.536 → 0.525 |

  Maximum |Δ| over all 16S/18S families and both windows: raw 0.07, A/e1 0.036, B/e1 0.014,
  C/e1 0.011. `pace_era` is the informative version. It trains and scores only on samples with
  PACE, PACE visible vs dropped in both, using leave-one-cruise-out for eDNA (3 cruises, so a
  small test) and 2024 → 2025 for SFER. It includes paired bootstrap intervals for the difference.
- **Uncertainty.**
  - The bootstrap resamples within cruises: the 2024 eDNA test year has only 4 cruises, so
    resampling whole cruises would be degenerate.
  - Rotating the test year covers year-to-year variation: eDNA 2020–2024, SFER 2018–2025, MaxN
    2018–2024, CRCP 2018/2021/2022/2024. Pretraining used no labels, so any label year can be the
    test year.
  - Caveats: training years after the test year are used, and the eDNA NMF/PCA bases were fitted
    on 2018–2023 labels. Both are shared by every feature set.

**Arm D design choices:**
- C (not B) is the JEPA base, because C ≥ B on most probes.
- The decoder has its own mask-token embeddings, so the predictor and E2 are built exactly as in
  arm C.
- ν = 0.1 (design-document default). The loss scales differ (embedding MSE vs MSE of transformed
  reflectance), so the logged `recon` against `pred` will show the actual balance.

**What reopens 2.11:** D (at ν = 0.1 or 0.3) matches A on Chl and PACE retention while keeping
B/C's NMF k5 and Si, and is not behind A on the focused tests: PACE-era phytoplankton groups (E1
visible vs dropped, vs MOANA) and fish added value over habitat. If both D runs trail A on those, A stands.

### Stage 1 results (job 116376, 2026-11-22): arm A (masked reconstruction) leads on every measure
Frozen final weights (step 30,000) of all three arms. CSVs on the laptop in
`multimodal_seusmbol/eval_results/stage1/` (outside the repo). The SFER probes are missing (bug,
rerun F2), so the main PACE on/off test is still to come.

**Embeddings.** 10,725 unique evaluation places/times. 10,021 (causal window) and 9,996 (retro)
have a valid observation in the 3 × 3 cells; 2,006 and 2,098 of those include PACE. Effective
rank at the evaluation samples (causal, PACE visible; other settings within ±3):

| Arm | E1 (512) | E2 (512) | E3 (1024) |
|---|---|---|---|
| A | 115.9 | – | 280.0 |
| B | 103.2 | 243.1 | 119.4 |
| C | 127.1 | 244.2 | 151.4 |

No collapse in any arm. In training, arm C's E1 rank was 212 on training batches.

**Probes, median test R² per target family.** Settings:
- causal window, PACE visible;
- eDNA with the sequencing covariates;
- targets with test spread < 0.25 × training spread left out (three of them).

Test sets: eDNA 183 samples (2024), CRCP 1,057, MaxN 616.

| Dataset | Family | meta | raw | A/e1 | A/e3 | B/e1 | B/e2 | C/e1 | C/e2 |
|---|---|---|---|---|---|---|---|---|---|
| 16S | NMF k5 | 0.269 | 0.376 | 0.711 | 0.679 | 0.645 | 0.680 | **0.745** | 0.678 |
| 16S | NMF k10 | 0.188 | 0.171 | **0.423** | 0.411 | 0.186 | 0.176 | 0.196 | 0.171 |
| 16S | NMF k15 | 0.151 | 0.125 | **0.389** | 0.354 | 0.137 | 0.144 | 0.056 | 0.141 |
| 16S | NMF k20 | 0.073 | 0.075 | **0.297** | 0.288 | 0.062 | 0.070 | 0.094 | 0.066 |
| 16S | PCs | 0.104 | 0.141 | **0.377** | 0.305 | 0.076 | 0.074 | 0.053 | 0.070 |
| 16S | genera | 0.084 | 0.141 | **0.304** | 0.274 | 0.167 | 0.147 | 0.144 | 0.141 |
| 18S | NMF k5 | 0.181 | 0.181 | 0.489 | 0.419 | 0.400 | 0.573 | 0.460 | **0.578** |
| 18S | NMF k10 | 0.151 | 0.071 | 0.417 | **0.418** | 0.165 | 0.203 | 0.183 | 0.203 |
| 18S | NMF k15 | 0.132 | 0.176 | 0.345 | **0.359** | 0.121 | 0.124 | 0.110 | 0.126 |
| 18S | NMF k20 | 0.093 | 0.140 | 0.284 | **0.285** | 0.069 | 0.106 | 0.107 | 0.109 |
| 18S | PCs | 0.057 | 0.070 | **0.220** | 0.215 | 0.103 | 0.123 | 0.115 | 0.122 |
| 18S | genera | −0.030 | 0.020 | **0.096** | 0.062 | 0.007 | 0.023 | 0.027 | 0.034 |
| CRCP | log1p count | 0.016 | 0.040 | −0.568 | −0.067 | 0.040 | 0.060 | 0.041 | **0.067** |
| MaxN | log1p MaxN | 0.050 | 0.009 | 0.069 | **0.076** | 0.068 | 0.027 | 0.055 | 0.034 |

The same comparison as mean Spearman: A/e1 is highest for every eDNA family.

| Dataset | Family | meta | raw | A/e1 | best of B or C |
|---|---|---|---|---|---|
| 16S | NMF k5 | 0.597 | 0.726 | 0.775 | 0.709 |
| 16S | NMF k20 | 0.383 | 0.396 | 0.543 | 0.450 |
| 16S | PCs | 0.483 | 0.465 | 0.626 | 0.476 |
| 18S | NMF k5 | 0.572 | 0.593 | 0.717 | 0.683 |
| 18S | NMF k20 | 0.446 | 0.501 | 0.574 | 0.476 |
| 18S | PCs | 0.352 | 0.393 | 0.588 | 0.443 |
| CRCP | log1p count | 0.254 | 0.281 | 0.290 | 0.318 |
| MaxN | log1p MaxN | 0.270 | 0.221 | 0.329 | 0.298 |

CRCP presence, mean AUC (causal):

| meta | raw | A/e1 | A/e3 | B/e1 | B/e2 | C/e1 | C/e2 |
|---|---|---|---|---|---|---|---|
| 0.542 | 0.612 | 0.642 | 0.628 | 0.626 | 0.624 | 0.631 | 0.624 |

Retro vs causal windows: within a few hundredths for A. B and C are lower in the retro window for
NMF k5.

**PACE in the probes (visible − dropped).** Mean change in R² on the test samples with usable PACE:

| Dataset | raw | A/e1 | A/e3 | B/e1 | B/e2 | C/e1 | C/e2 |
|---|---|---|---|---|---|---|---|
| 16S | −0.049 | −0.003 | +0.005 | −0.001 | 0.000 | −0.002 | 0.000 |
| 18S | −0.056 | +0.025 | +0.042 | −0.001 | 0.000 | −0.003 | 0.000 |
| CRCP | +0.004 | −0.235 | −0.038 | −0.003 | 0.000 | −0.005 | 0.000 |
| MaxN | +0.002 | −0.005 | −0.006 | −0.001 | −0.002 | −0.001 | −0.005 |

Why the eDNA and fish probes can't show a PACE benefit:
- Their training labels have no PACE (all pre-2024), so a probe never sees PACE-influenced
  embeddings until the test.
- A's embeddings move when PACE is present: E1 R² differs by up to 0.72 between settings. For A,
  PACE-visible is therefore partly a distribution shift (CRCP −0.235).
- B's and C's E2 barely change: mean |ΔR²| is 0.0001–0.0002.

SFER (346 training samples with PACE) is the fair test of PACE: rerun F2.

**PACE spectral retention.** Ridge decoder from E1 to the normalised PACE spectrum (172 bands,
346–719 nm). Trained on 1,959 anchors (61,845 cells); tested on 777 anchors each:
- test_temporal: Apr–Oct 2026, 24,354 cells;
- test_geo: geographic blocks, 24,512 cells.

R² by band group:

| Model | Setting | geo all | geo UV | geo blue | geo green | geo red | temp all | temp UV | temp blue | temp green | temp red |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heritage ridge* | – | 0.967 | 0.940 | 1.000 | 0.999 | 0.942 | 0.968 | 0.946 | 1.000 | 0.998 | 0.943 |
| A | visible | **0.738** | 0.673 | 0.889 | 0.939 | 0.578 | **0.637** | 0.535 | 0.579 | 0.855 | 0.592 |
| A | hidden | 0.687 | 0.626 | 0.873 | 0.929 | 0.487 | 0.572 | 0.451 | 0.538 | 0.836 | 0.497 |
| B | visible | −0.126 | −0.449 | −0.287 | 0.500 | −0.248 | 0.188 | 0.077 | 0.003 | 0.565 | 0.139 |
| B | hidden | −0.119 | −0.437 | −0.282 | 0.507 | −0.240 | 0.190 | 0.076 | 0.005 | 0.570 | 0.138 |
| C | visible | −0.033 | −0.335 | −0.208 | 0.579 | −0.147 | 0.204 | 0.060 | −0.011 | 0.598 | 0.171 |
| C | hidden | −0.030 | −0.334 | −0.212 | 0.580 | −0.137 | 0.204 | 0.056 | −0.010 | 0.602 | 0.170 |
| train mean | – | −0.074 | −0.083 | −0.131 | −0.015 | −0.069 | −0.092 | −0.144 | −0.110 | −0.013 | −0.104 |

RMSE (all bands, normalised units), test_geo / test_temporal: heritage 0.101 / 0.126; A visible
0.393 / 0.598; A hidden 0.428 / 0.650; B 0.859 / 0.895; C 0.822 / 0.885; train mean
0.862 / 1.046.

\* The heritage ridge maps the same observation's own heritage bands to its PACE spectrum. It is a
ceiling for "spectrum given the heritage bands", not a competitor for an embedding.

Reading:
- A's E1 keeps most of the spectrum.
- B's and C's E1 keep almost none outside the green, and hiding PACE changes nothing. Their
  embeddings ignore the PACE observation.
- C's PACE-residual term did not create retention (C − B: +0.09 geo, +0.02 temporal).
- A is worse in Apr–Oct 2026 than on the geographic blocks (blue 0.58 vs 0.89). That period has
  no OLCI (OB.DAAC ends April 2026), and it is outside the training years.

**Sequencing covariates (eDNA).** They raise every feature set. The mean R² of 16S NMF k10 (causal,
PACE dropped) goes:
- meta 0.023 → 0.187;
- A/e1 0.204 → 0.442;
- B/e2 0.126 → 0.273.

For the strongest target the gain is small (16S NMF k5 A/e1: 0.738 → 0.744).

**Fish.** Weak for all arms. MaxN R² ≤ 0.08, CRCP presence AUC ≤ 0.64.

CRCP counts with A/e1 have a strongly negative R² (−0.57) but the best Spearman and AUC. So the
ranking carries over but the level does not: the training years are 2018/2021/2022 and the test
year is 2024. A/e3 (−0.07) is far less affected.

**SFER (F2, 2026-11-23).** Surface samples, test 2025 (464 samples, 459 with PACE). Training
2018–2024: 2,433 samples (causal) / 2,358 (retro), 346 of them with PACE. Chl has 2,301 training
labels. Test R² with PACE visible (one target per family, so median = value):

| Target | Window | meta | raw | A/e1 | A/e3 | B/e1 | B/e2 | C/e1 | C/e2 |
|---|---|---|---|---|---|---|---|---|---|
| Chl | causal | −0.080 | 0.740 | **0.800** | 0.787 | 0.573 | 0.556 | 0.601 | 0.543 |
| Chl | retro | −0.080 | 0.771 | **0.838** | 0.808 | 0.610 | 0.555 | 0.647 | 0.539 |
| Si | causal | −0.012 | 0.338 | 0.378 | 0.404 | 0.406 | **0.411** | 0.404 | 0.400 |
| Si | retro | −0.011 | 0.360 | 0.387 | 0.387 | **0.408** | 0.405 | 0.407 | 0.395 |
| NO₃ | causal | −0.016 | −0.079 | 0.013 | 0.016 | −0.044 | 0.022 | −0.044 | **0.026** |
| NO₃ | retro | 0.003 | −0.049 | 0.022 | 0.022 | 0.023 | 0.040 | 0.039 | **0.044** |
| PO₄ | causal | −0.178 | −0.049 | 0.012 | −0.029 | 0.008 | **0.073** | 0.029 | 0.066 |
| PO₄ | retro | −0.176 | −0.064 | 0.026 | −0.042 | **0.107** | 0.082 | 0.083 | 0.076 |

E3: B 0.425 / 0.426 and C 0.516 / 0.530 for Chl; B 0.231 / 0.248 and C 0.319 / 0.334 for Si.

Reading:
- Chl: A is best. It beats the raw reflectance baseline by +0.06 / +0.07, while B and C fall
  0.14–0.20 below raw.
- Si: all arms ≈ 0.38–0.41, raw 0.34–0.36.
- NO₃ and PO₄: no skill from any feature set (≤ 0.11). Surface nutrients at these stations are
  mostly not optically expressed. They are the targets Stage 2 (PISCES supervision) is meant to
  help.
- PACE on/off for SFER, the fair test (346 training samples with PACE): **no benefit for any arm**.
  The table shows test R² on the 459 test samples with usable PACE, as dropped → visible (Δ).
  Training labels: 2,431 (causal) / 2,357 (retro); Chl 2,028 / 1,956.

| Target | Window | raw | A/e1 | A/e3 | B/e1 | B/e2 | C/e1 | C/e2 |
|---|---|---|---|---|---|---|---|---|
| Chl | causal | 0.728 → 0.735 (+0.007) | 0.787 → 0.796 (+0.009) | 0.780 → 0.783 (+0.003) | 0.561 → 0.564 (+0.003) | 0.546 → 0.545 (0.000) | 0.589 → 0.591 (+0.002) | 0.531 → 0.531 (0.000) |
| Chl | retro | 0.769 → 0.766 (−0.003) | 0.839 → 0.833 (−0.005) | 0.811 → 0.804 (−0.007) | 0.602 → 0.600 (−0.002) | 0.544 → 0.544 (0.000) | 0.639 → 0.639 (0.000) | 0.526 → 0.527 (0.000) |
| Si | causal | 0.315 → 0.326 (+0.011) | 0.382 → 0.365 (−0.017) | 0.393 → 0.392 (−0.001) | 0.398 → 0.394 (−0.004) | 0.399 → 0.399 (0.000) | 0.396 → 0.392 (−0.003) | 0.388 → 0.388 (0.000) |
| Si | retro | 0.322 → 0.348 (+0.027) | 0.357 → 0.374 (+0.017) | 0.396 → 0.375 (−0.021) | 0.406 → 0.396 (−0.010) | 0.393 → 0.393 (0.000) | 0.414 → 0.395 (−0.019) | 0.383 → 0.383 (0.000) |
| PO₄ | causal | −0.098 → −0.094 | −0.017 → −0.051 (−0.034) | −0.027 → −0.096 (−0.069) | −0.076 → −0.074 | −0.011 → −0.011 | −0.049 → −0.047 | −0.019 → −0.019 |
| NO₃ | causal | −0.088 → −0.088 | 0.009 → 0.004 | 0.013 → 0.006 | −0.054 → −0.054 | 0.009 → 0.009 | −0.055 → −0.055 | 0.015 → 0.015 |

  The changes are within ±0.03 everywhere (PO₄ A/e3 −0.07, on a target with no skill).

  Why PACE adds nothing here:
  - PACE presence is not where the information gain is: the embeddings already reach R² 0.80–0.84
    on Chl from the heritage sensors.
  - Only 14% of the probe's training samples carry PACE.
  - B/E2 and C/E2 do not respond to PACE at all (Δ 0.000), consistent with the retention result.

  The PACE question for this phase: hyperspectral PACE does not improve these frozen-embedding
  probes in Stage 1.

**Summary for 2.11:**
- A wins the eDNA probes in 10 of 12 families (median R²).
- B or C wins 16S NMF k5 (C/e1 0.745 vs A 0.711) and 18S NMF k5 (C/e2 0.578 vs A 0.489), by small
  margins.
- A is the only arm that retains PACE spectra (0.74 / 0.64 vs ≤ 0.20).
- SFER Chl: A 0.80 / 0.84 vs B/C ≤ 0.65 (raw 0.74 / 0.77). Si: tie. NO₃/PO₄: no skill.
- PACE visible vs dropped: no gain on SFER (|Δ| ≤ 0.03 for Chl and Si, every arm), eDNA or fish.
- Recommendation: A. It is best or tied on every probe family except 16S/18S NMF k5, and it is the
  only arm that retains PACE spectra.

Each entry gives the decision, the evidence, the reasoning, and what would reopen it. Newest first.

### Stage 1 model: design document v1, with your adjustments (2026-11-12)
**Source.** "Ocean Foundation Model Design Decisions.md" (repo root), "Implementation specification
(v1)", edited on 2026-11-12 to the settled pilot decisions. "Follow fork 1": read as *build the v1
specification* (fork #1 in the table, released embedding E2 or E3 instead of E1, is inference-only
and evaluated in the pilot anyway: E1, E2 and E3 are all implemented). Other forks stay available.
**Built as specified:** SRF-integrated spectral tokenizer; 6 spatial + 4 temporal blocks at D = 512;
predictor 2 + 2 blocks at 384; targets = spatial-stage outputs of hidden observations encoded alone,
no EMA, gradients through targets; SIGReg on targets only; masking modes 40/30/20/10; spectral dropout
20–100% of PACE bands; arm A band-query decoder; arm C ridge-residual decoder; AdamW / cosine / bf16 /
effective batch 128.
**SIGReg** follows the official LeJEPA library code (galilai-group/lejepa, fetched 2026-11-12): 17
points on [0, 5] with doubled trapezoid weights (= [−5, 5]), window exp(−t²/2), × N × world size,
cos/sin means all-reduced separately, slice seed from a step counter synchronised across GPUs,
1,024 slices, λ = 0.05. (The paper's text, its listing and the README differ slightly; the library
code was used.) Unit test: isotropic Gaussian ≈ 1, collapsed > 600.
**Your adjustments (2026-11-12).** You edited the design document in parallel (it arrived as
"Ocean Foundation Model Design Decisions 2.md", a sync-conflict copy); where your text and my
first pass differed, your version was adopted and the code changed to match. The merged document
is the main file; the "2" copy is left untouched and uncommitted.
1. Wavelength generator 340–2300 nm for every view (SFREFL SWIR to ~2258 nm), conditioned on the
   product type (Rrs or rhos); separate normalisation curves per view.
2. Modality balancing: per view group (MODIS Rrs, VIIRS Rrs over all platforms, OLCI Rrs, PACE Rrs,
   PACE rhos; stage 2 adds nutrients and physics), prediction MSE averaged within each group, then
   equal-weight mean over groups; **SIGReg per view group** (groups under 256 tokens pooled), equal-weight
   mean; whole-view masking draws uniformly over the groups present; per-group losses logged. To do:
   inverse-frequency / temperature-scaled weights. Multi-GPU per-group SIGReg uses global sums over a
   fixed group list (exactly the reference statistic when GPUs hold equal token counts; avoids
   mismatched collective calls).
3. Amplitude: **no LayerNorm on the token** (it divided out amplitude: spectrum vs ×2 gave cosine
   0.99995); two amplitude features (mean transformed value, asinh of mean raw reflectance) through
   a small MLP. Unit test at initialisation: 0.5× / 2× copies distinguishable, Rrs vs rhos
   distinguishable, linear probe recovers log amplitude with R² ≥ 0.95 (passes).
4. Quality channels: 9 (HISATZEN plus the eight soft flags).
5. E2: reference = MODIS-Aqua ocean bands; in 10% of sequences a hidden PACE or OLCI observation is
   also a synthetic target SRF-convolved onto those bands (encoded alone at its own Δt); native hidden
   MODIS observations count too. E3 = 1,024 dimensions.
6. PACE ridge from 2024–2025 only (the training split; never 2026).
**Deviations from the v1 text, with reasons:** Δt is added at the temporal stage, not in the
tokenizer (a target token must not depend on t0); SRFs are Gaussian at nominal FWHM (the files carry
no band widths; measured response tables are an easy later swap); the monitoring probe uses SFER
chlorophyll (no matched HPLC; for now fitted at checkpoints, not online); the transform's noise scale s(λ) is the robust spread of
neighbouring-cell differences in training data, fitted smoothly in λ.
**GPU environment:** Python 3.12 + PyTorch (your call: whatever works best). Everything after training
(embedding extraction, probes) also runs in that environment, so the 3.14 environment is not needed
for testing; checkpoints are plain PyTorch state dicts.
**Open:** headline NMF k (all of 5/10/15/20 are built); you mentioned possibly switching the eDNA
target to taxonomy prediction (dominant genera are already built as targets).

### Flag policy v2: soft flags become inputs, not masks (2026-11-12)
The design document's v1 flag policy keeps ATMWARN, MAXAERITER, LOWLW, ABSAER, COCCOLITH, TURBIDW,
COASTZ and OPSHAL as per-cell **quality channels** (model inputs) and does not mask CHLFAIL/CHLWARN;
the tiles built so far (v1) used NASA's L3 mask minus HISATZEN, which removes several of those pixels
(blooms, coccolithophores, low water-leaving radiance). Masks are applied while gridding, so this
needs a re-tile: `processing.version: v2`, hard masks LAND, CLDICE, SEAICE, ATMFAIL, NAVFAIL, HILT,
HIGLINT, HISOLZEN, BOWTIEDEL, STRAYLIGHT (+ HISATZEN for MODIS only); SFREFL: LAND and NAVFAIL only
(floating mats can trip cloud and high-radiance flags; all flags stay stored). The strict
spectral-evaluation subset is unchanged (NASA L3 default). Physics coarse grid set to the document's
4 × 4 per tile (the 3 × 3-tile context is a config fork). Cost: one ~4 h re-run of `prep.sbatch`.

### PACE era vs the 2024 hold-out year. *Approved (2026-11-12) with your adjustments; implemented in 2.5.*
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

**Decision (your approval, with adjustments).** Config `training.partitions`; enforced by `oefm/loader.py`.
1. **Imagery vs labels.** Self-supervised pretraining uses unlabeled imagery from 2018–2025,
   including all 2024 and 2025 PACE. The 2024 hold-out applies to downstream biological *labels*,
   not to self-supervised imagery: no label, and no target derived from labels, enters pretraining.
2. **PACE spectral retention.** Train March 2024 – December 2025; validate January – February 2026;
   March 2026 is a separation period used by nothing; test April – October 2026. Every split is
   formed by first keeping only its own observations and then building windows, so no sequence
   (causal or retrospective) can contain an observation from another split: a training window near
   2026-01-01 simply ends at the last 2025 observation. The same holds for all sensors, so heritage
   observations from the test months are not seen in pretraining either.
   **Independent geographic test:** tiles grouped in 4 × 4-tile blocks (153.6 km); 20% of blocks
   (seeded) are held out of pretraining at all times, with a 1-tile buffer used by nothing:
   193 train tiles (71%), 43 geo-test tiles (16%), 36 buffer tiles (13%). Evaluated on the training
   period, so it is purely spatial. (3 × 3 blocks would have cost 41% of tiles with the buffer.)
3. **eDNA.** All 2024 labels held out; probes trained on 2018–2023. Target preprocessing
   (Hellinger, NMF, feature selection) is refitted without the 2024 samples (step 2.7). PACE-visible
   vs PACE-hidden embeddings are compared on the identical 2024 test samples
   (`TileSequences(pace="visible"|"hidden")`); the PACE-specific difference is exploratory.
4. **PACE coverage reporting.** No complete 30-day PACE history is required. Step 2.4b reports per
   dataset and year, from actual pixel validity (valid weight in the sample's cell or 3 × 3
   neighbourhood under the training mask): ≥ 1 valid PACE observation, ≥ 2, whole window inside the
   PACE era, and whole window plus ≥ 1 valid; same-day PACE; and ≥ 1 for VIIRS, MODIS, OLCI.
5. **SFER is the primary biological PACE-on/off comparison:** probes trained on labels before 2025
   (including 2024), tested on 2025 (465 samples, 6 cruises), identical samples and identical heads
   for PACE visible and hidden.
6. **Fish:** 2024 labels held out; PACE-specific comparisons exploratory. **Video MaxN dates are
   month precision only** (checked: all 5,112 events are dated the 15th at 00:00 UTC; the Granite
   matches were 0.3–3.7 days from that date). They are flagged `time_precision: month`: coverage
   uses the calendar month, and no exact-date histories are built unless exact survey dates are found
   in the original GF records.
7. **Objective comparison (Stage 1):** reconstruction, JEPA and JEPA + PACE guard use the same
   splits, samples, grouping and preprocessing; ecological prediction and PACE spectral retention are
   reported separately.

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

### eDNA: sequencing platform and depth are covariates in every downstream model (2026-11-20)
Your instruction: platform and sequencing depth influence the eDNA results a lot, so they go into
the downstream models. Both are in the SEUS MBON metadata (`platform`, `input_read_count`,
`seq_run_id`) for every replicate of all 913 evaluation samples; none is missing.

Platform is almost entirely confounded with time, and the test set is all on one platform:

| Collection year | Illumina | Element AVITI | Both (replicates split) |
|---|---|---|---|
| 2018 | 0 | 5 | 0 |
| 2020 | 82 | 0 | 0 |
| 2021 | 127 | 0 | 30 |
| 2022 | 215 | 0 | 10 |
| 2023 | 0 | 261 | 0 |
| 2024 (test) | 0 | 183 | 0 |

Training labels: 424 Illumina, 266 Element, 40 mixed. Depth follows platform. The median of the
input reads summed over replicates is:

| Marker | Element | Illumina | Ratio |
|---|---|---|---|
| 16S | 715k | 41k | ~18× |
| 18S | 1.10M | 34k | ~32× |

The 2024 test samples were sequenced in two 2025 runs that also hold 22 and 81 training replicates
(same numbers for 18S).

Implementation:
- `make_eval_targets.py` adds per-sample `seq_*` columns, computed over the sample's replicates.
- `evaluate.py` gives every eDNA probe two **unpenalised** covariates: Element share and
  log10(1 + mean input reads per replicate). They are partialled out of features and targets, the
  ridge is fitted on the residuals, and then least squares is fitted on [1, covariates], so the
  ridge penalty cannot shrink them away.
- An extra baseline `seq` uses the covariates only.
- Probes are reported both with and without the covariates.
- `seq_*` columns are never targets.
- The run ID is kept for reference but is not a covariate: within training it is mostly year, so a
  run offset would absorb interannual change.

Choice of covariates, by grouped CV over 20 cruises on the **training labels only**, with the meta
features (lat, lon, season). Mean CV R² per target family:

| | none | all 4 seq numbers | Element share + input depth |
|---|---|---|---|
| 16S NMF k5 | 0.152 | 0.208 | 0.239 |
| 16S NMF k10 | 0.038 | 0.088 | 0.111 |
| 16S NMF k20 | −0.059 | −0.065 | −0.038 |
| 16S PCs | 0.038 | 0.112 | 0.130 |
| 16S genera | −0.003 | 0.018 | 0.032 |
| 18S NMF k5 | 0.163 | 0.182 | 0.196 |
| 18S NMF k10 | 0.084 | 0.084 | 0.110 |
| 18S NMF k20 | −0.016 | −0.070 | −0.034 |
| 18S PCs | 0.072 | 0.089 | 0.102 |
| 18S genera | −0.022 | −0.052 | −0.026 |

Single covariates were similar: platform alone 16S k5 0.235, depth alone 0.245. Covariates alone
score below zero (e.g. 16S k5 −0.085), because platform is constant within a cruise and the CV
holds out whole cruises.

Caveats:
- The targets themselves (Hellinger NMF, PCA) were fitted across both platforms. A
  platform-driven component would therefore be a target to explain, not noise. In the test year
  (all Element) the covariates only shift the intercept.
- The replicate-mean count tables average Illumina and Element replicates for the 40 mixed
  samples; Element dominates these by depth.
- If platform effects turn out to be large in the targets, the options are platform-specific
  normalisation, refitting NMF on Element only, or rarefying. That is your call with the
  taxonomy/NMF decision.

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
| 2.8b GPU env + check | `tail -10 logs/oefm_gpuenv_<id>.out`; `tail -30 logs/oefm_gpucheck_<id>.out` | torch/CUDA versions, all-reduce = 10, each arm runs |
| 2.8c benchmark | the summary table at the end of `logs/oefm_bench_<id>.out` | batch, length and steps to set for the runs |
| 2.9 training | `tail -5 data/runs/stage1_<ARM>/log.jsonl` | losses per modality, SIGReg, effective rank, no collapse |
| G0 targets + MOANA | `tail -20 logs/oefm_targets_<id>.out`; the `phyto_`/`hab_` column counts (5, 9) | MOANA files found, samples with a matchup, new columns joined |
| 2.10 / 2.10b / 2.10c evaluation, robustness, focused | `tail -250 logs/oefm_eval_<id>.out` (I: `tail -400`); copy `data/runs/eval_<TAG>/*.csv` to `multimodal_seusmbol/eval_results/<TAG>/` | probe tables, intervals, rotation, PACE-era test, retention, amplitude, phytoplankton vs MOANA, fish over habitat, window spans |
| 2.9b arm D (ν 0.1, 0.3) | first lines of `logs/oefm_train_<id>.out` and `tail -3 data/runs/<run>/log.jsonl` (`stage1_D`, `stage1_D3`) | `recon` next to `pred`/`sigreg`/`pace_residual`, ν, memory, speed |
| 2.6–2.7 targets | `tail -40 logs/oefm_targets_<id>.out` | PISCES days per source, tile coverage, the matched-evaluation summary |
| 2.0–2.4b trial (`LIMIT=40`) | `tail -40 logs/oefm_prep_<id>.out` | speed per granule, no failures |
| 2.0–2.5a full | `tail -80 logs/oefm_prep_<id>.out` (includes the sequence summary and physics coverage); `cat data/processed/regional/tiles/v2/index/summary_by_product.csv data/processed/regional/tiles/v2/index/eval_coverage_summary.csv`; copy back `data/processed/regional/tiles/v2/examples/` | tile counts, the PACE-usable evaluation counts, the visual check |
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
| 2026-11-18 | 116167 | targets rerun after the xarray pin: 109/109 PISCES files; 3,295 of 3,301 days (3,218 reanalysis, 77 forecast), median tile coverage 1.00; evaluation sets re-matched (same counts as 116161). Stage 1 training A, B, C submitted (user) |
| 2026-11-22 | 116376 | evaluate (all three arms, final weights) ✅ 1 h 30 min: embeddings 4 × ~815 s, probes, PACE retention. A leads (decision log, Stage 1 results). Bugs: SFER 0 targets (missing Chl dropped every target), family means dominated by 3 near-constant test targets; fixed, rerun probes (F2) |
| 2026-11-21 | 116240 | train arm C ✅ 30,000 steps in 25.6 h (41.7 seq/s, 27.1 GB peak, loader wait 0.1%); follow-on 116366 cancelled itself. Last steps: prediction loss 0.003-0.007 per sensor, SIGReg ~2.0 per group (from 1,456 at the start), PACE residual 0.02-0.17, total loss ~0.11; effective rank of E1 212 (no collapse). Next: confirm weights_final.pt for A, B, C, then step F |
| 2026-11-20 | 116353 | targets `STEPS=evalsets` with the refreshed eDNA files (sequencing covariates): all sets matched in 4 s; counts identical to 116161/116167 (eDNA test 183, 100/92 with PACE; SFER test 465, 459/457; CRCP 1,057, 1,027/995; MaxN 616, all). PISCES 109/109 verified |
| 2026-11-18 | 116200, 116201 | benchmark B/C (fix works) and A at batch 16/32. L16 b32: A 42.2 seq/s 26.0 GB 25.3 h; B 40.1, 26.0 GB, 26.6 h; C 39.9, 26.7 GB, 26.7 h (30k steps × 128). L8 b32 ~74 seq/s (~14.5 h); L24 b32 ~27 seq/s (~39 h), 38-39 GB. Loader wait ≤ 0.14. Over 200 steps SIGReg fell 1,456 → ~34, effective rank of E1 ~150-170 (no collapse), per-sensor losses all falling. Chosen: L16, b32, 30k steps for all arms; train.sbatch cancels its follow-on when an arm finishes |
| 2026-11-18 | 116191 | benchmark: arm A ✅ all 9 (seq/s, peak GB, projected h for 30k steps × 128: L8 b2/4/8 42.4/60.1/83.6, 1.8/2.6/4.4 GB, 25.1/17.7/12.8 h; L16 33.8/45.5/53.6, 2.5/4.2/7.4 GB, 31.6/23.5/19.9 h; L24 29.0/34.5/38.3, 3.3/5.7/10.5 GB, 36.8/30.9/27.9 h; loader wait ≤ 1%). Arms B and C: all 18 died at ~10.5 min each = NCCL 10-min timeout. Cause: GroupSIGReg's autograd all_reduce ran inside backward and interleaved with DDP's gradient all-reduces in a different order per rank (ranks hold different view groups). Reproduced with 2-process gloo DDP; fixed (all-reduce detached sums in forward, gradient through local tokens × world size: identical loss and gradient to 1e-9); new tests/test_ddp.py |
| 2026-11-17 | 116190 | benchmark: 932,992 training anchors (date fix works; normstats rebuilt in 116188), but the first configuration failed under bf16 autocast: `Index put requires the source and destination dtypes match` (predictor input grid float32, projection output bf16). Fixed in `oefm/model/net.py`; new test runs all arms under bf16 autocast. Rerun benchmark |
| 2026-11-17 | 116176 | benchmark: all 27 configurations failed with `anchors 0` (reported as "probably out of memory"). Cause: pandas 3 in the GPU env parses times at µs resolution; `astype(int64)/1e9` in the loader gave 1970 dates. Fixed (resolution-independent epoch seconds); empty training set now errors; benchmark stops if the first configuration fails |
| 2026-11-16 | 116161 | targets: evalsets ✅ (eDNA test 183, 100 with ≥1 PACE / 92 with ≥2; SFER test 465, 459/457; CRCP test 1,057, 1,027/995; MaxN test 616, all with PACE). PISCES download: 0 of 109 (xarray 2026.4+ passes `zarr_format`, unsupported by zarr 2). Fixed: `xarray<2026.4`; rebuild env, rerun download build evalsets |
| 2026-11-16 | 116124 | prep v2 finished successfully (user report): tiles, index, coverage, physics, sequences, normstats. Next: targets + GPU env |
| 2026-11-14 | 116124 | prep v2 resubmitted after cleanup: tiles/v1 and data_smoke deleted (~233 GB); `processed --prune-version v1` removed 1,986,998 tile records and 24,335 outputs (database 964 MB → 40 MB). `mmlsquota` is not available to users, so the quota cause stays unconfirmed. All 3,290 MUR days are cropped (so the 8 md5 failures of step 1.7 are resolved) |
| 2026-11-14 | 116106 | prep v2 re-tile stopped after ~800 of 24,335 granules: `sqlite3.OperationalError: database or disk is full` when writing processing.sqlite (0.3 s per granule before that). Filesystem 21% full, database intact, `quota`/`mmlsquota` printed nothing (errors hidden): a user/group quota (EDQUOT) is the likely cause; usage ~6.15 TB. Also found: index rows of v1 could leak into the v2 index (fixed, export filtered by version) and v1 records made the database 964 MB (new `processed --prune-version v1`). Cleanup + resubmit |
| 2026-11-12 | 116101 | prep trial again after the fix: 36 granules tiled, 4 empty (no valid cell in any tile), 0.6 s per granule on 12 workers; 6 example figures; 2.4b ran (all zero: only Feb 2018 tiled) |
| 2026-11-12 | 116100 | prep trial (`LIMIT=40`): layout, grid (272 tiles), 40 CCMP crops OK; all 40 tile jobs failed on 2018 files without `scan_line_attributes/time`. Fixed: fall back to year/day/msec, then to the file's start/end time. Failed granules are retried automatically on the next run |
| 2026-11-12 | qc.sbatch | QC run on all data, reviewed (1.6). 34,000+ files verified across 15 products, 6.44 TB; 8 MUR files failed md5 (step 1.7). 21 eDNA windows loaded. HISATZEN settled per sensor (MODIS keeps the hard mask). 7,245 region-days with 2+ instrument families, 2,610 with PACE |
| 2026-11-12 | download (1.4, 1.4b) | PACE, physical products and the eDNA-window days downloaded; ~220–270 MB/s aggregate |
| 2026-11-11 | 116027–116029 | Three download jobs ran at once and found nothing to do; the six satellite products are fully verified (1.79 TB) |
| 2026-11-11 | 116018 | Satellite download crashed at 9103/9129 (thread-unsafe manifest writes; fixed in `01d3ed4`); follow-on 116019 finished the rest |
| 2026-11-11 | 116016 | Smoke test, 23/23 verified; ~25 MB/s per NASA connection |
| 2026-11-11 | (inventory) | First inventory stopped before PACE; re-run by the user |
