#!/bin/bash
#SBATCH --job-name=oefm_inventory
#SBATCH --partition=service
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Dry-run inventory: check the archives, search every product over the regions and windows,
# and fill data/manifest/manifest.sqlite. Downloads nothing. Safe to re-run: new files are
# added, known ones refreshed, download state kept.
#
#   cd /scratch/ptolemy/users/$USER/ocean_ecology_foundation_model
#   sbatch --account=191001-366746 cluster/inventory.sbatch
#   sbatch --account=191001-366746 --export=ALL,PRODUCTS="PACE OLCI" cluster/inventory.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
echo "Job started: $(date)"
python -m oefm regions
python -m oefm windows
python -m oefm catalog
# shellcheck disable=SC2086
python -m oefm inventory ${PRODUCTS:+--products ${PRODUCTS}}
python -m oefm report
echo "Job finished: $(date)"
