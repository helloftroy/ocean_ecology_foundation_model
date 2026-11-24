#!/bin/bash
#SBATCH --job-name=oefm_prep
#SBATCH --partition=service
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase 2 data preparation, steps 2.0-2.4 (cluster_instructions.md). Reads data/raw/, writes only
# data/processed/. Resumable: every step skips outputs already recorded in
# data/processed/regional/processing.sqlite, so just resubmit after a crash or the 48 h limit.
#
#   cd /scratch/ptolemy/users/$USER/ocean_ecology_foundation_model
#   sbatch --account=191001-366746 cluster/prep.sbatch
#   # a quick trial first (the first LIMIT files of crop and of tiles):
#   sbatch --account=191001-366746 --export=ALL,LIMIT=40 cluster/prep.sbatch
#
# STEPS (default "layout grid crop tiles examples evalcover physics sequences normstats"), WORKERS (default 12), LIMIT (default 0 = all).
# Tiling v3 (catalog_v1, raw kept): STEPS=tiles3 (PER_PRODUCT=N for a trial on N granules per product,
# PRODUCTS="..." to restrict, FORCE=1 to re-tile granules already done); STEPS=index3 re-exports the v3 indexes.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
STEPS="${STEPS:-layout grid crop tiles examples evalcover physics sequences normstats}"
WORKERS="${WORKERS:-12}"
LIMIT="${LIMIT:-0}"
echo "Job started: $(date) | steps: ${STEPS} | workers: ${WORKERS} | limit: ${LIMIT}"
for step in ${STEPS}; do
  echo "=== ${step} $(date)"
  case "${step}" in
    layout)   python -m oefm layout ;;
    grid)     python -m oefm grid ;;
    crop)     python -m oefm crop --workers "${WORKERS}" --limit "${LIMIT}" ;;
    tiles)    python -m oefm tiles --workers "${WORKERS}" --limit "${LIMIT}" ;;
    examples) python -m oefm examples --n 6 ;;
    evalcover) python -m oefm evalcover ;;
    physics)  python -m oefm physics --workers "${WORKERS}" ;;
    sequences) python -m oefm sequences ;;
    normstats) python -m oefm normstats ;;
    tiles3)   python -m oefm tiles3 --workers "${WORKERS}" --limit "${LIMIT}" ${PER_PRODUCT:+--per-product ${PER_PRODUCT}} ${PRODUCTS:+--products ${PRODUCTS}} ${FORCE:+--force} ;;
    index3)   python -m oefm index3 ;;
    *) echo "unknown step ${step}" >&2; exit 1 ;;
  esac
done
python -m oefm processed
du -sh data/processed/regional/* 2>/dev/null || true
echo "Job finished: $(date)"
