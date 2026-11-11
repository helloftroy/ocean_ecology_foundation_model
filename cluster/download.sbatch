#!/bin/bash
#SBATCH --job-name=oefm_download
#SBATCH --partition=service
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Download planned files from the manifest (run cluster/inventory.sbatch first). Resumable:
# verified files are skipped, partial files resumed, failures retried on the next run.
# Run one download job at a time (one SQLite manifest, one writer).
#
#   cd /scratch/ptolemy/users/$USER/ocean_ecology_foundation_model
#   sbatch --account=191001-366746 cluster/download.sbatch                                  # everything planned
#   sbatch --account=191001-366746 --export=ALL,PRODUCTS="PACE OLCI",START=2024-01-01 cluster/download.sbatch
#   sbatch --account=191001-366746 --export=ALL,CHAIN=5 cluster/download.sbatch            # auto-resubmit up to 5 times
#
# Variables: PRODUCTS (keys/families/satellite/physical), START, END, REGION, PER_PRODUCT,
# MAX_FILES, CONNECTIONS (default 8), CHAIN (re-queue itself after the 48 h limit, N more times).

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"

# Queue the follow-on job first, so it runs even if this one is killed at the time limit.
if [ "${CHAIN:-0}" -gt 0 ]; then
  next=$(sbatch --parsable --account="${SLURM_JOB_ACCOUNT}" --dependency=afterany:"${SLURM_JOB_ID}" \
         --export=ALL,CHAIN=$((CHAIN - 1)) "${REPO_ROOT}/cluster/download.sbatch")
  echo "Queued follow-on download job ${next} (CHAIN=$((CHAIN - 1)))"
fi

echo "Job started: $(date)"
df -h "${OEFM_DATA_ROOT}" | tail -1
ARGS=(--connections "${CONNECTIONS:-8}")
[ -n "${PRODUCTS:-}" ] && ARGS+=(--products ${PRODUCTS})
[ -n "${START:-}" ] && ARGS+=(--start "${START}")
[ -n "${END:-}" ] && ARGS+=(--end "${END}")
[ -n "${REGION:-}" ] && ARGS+=(--region "${REGION}")
[ -n "${PER_PRODUCT:-}" ] && ARGS+=(--per-product "${PER_PRODUCT}")
[ -n "${MAX_FILES:-}" ] && ARGS+=(--max-files "${MAX_FILES}")
python -m oefm download "${ARGS[@]}"
python -m oefm status
python -m oefm report
echo "Disk used by raw data: $(du -sh "${OEFM_DATA_ROOT}/raw" 2>/dev/null | cut -f1)"
echo "Job finished: $(date)"
