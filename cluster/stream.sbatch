#!/bin/bash
#SBATCH --job-name=oefm_stream
#SBATCH --partition=service
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Download, tile (v3, catalog_v1), delete raw: batch after batch (oefm/stream.py). Only tiles are kept;
# the manifest keeps every granule (status 'tiled') so any file can be fetched again. Needs the
# internet ('service') and Earthdata credentials (source credentials.sh first). One download job per
# manifest at a time. Resumable: CHAIN=N re-queues itself after the 48 h limit.
#   sbatch --account=191001-366746 --export=ALL,OEFM_CONFIG=config/biogoship.yaml,OEFM_DATA_ROOT=$PWD/data_biogoship,CHAIN=3 cluster/stream.sbatch
#   # one product only, e.g. PACE BGC for Florida:  --export=ALL,PRODUCTS=pace_oci_bgc
# Variables: PRODUCTS, BATCH (default 200 granules), CONNECTIONS (8), WORKERS (24), MAX_BATCHES, CHAIN.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
if [ "${CHAIN:-0}" -gt 0 ]; then
  next=$(sbatch --parsable --account="${SLURM_JOB_ACCOUNT}" --dependency=afterany:"${SLURM_JOB_ID}" \
         --export=ALL,CHAIN=$((CHAIN - 1)) "${REPO_ROOT}/cluster/stream.sbatch")
  echo "Queued follow-on stream job ${next} (CHAIN=$((CHAIN - 1)))"
fi
echo "Job started: $(date) | config ${OEFM_CONFIG:-config/pilot.yaml} | data ${OEFM_DATA_ROOT}"
df -h "${OEFM_DATA_ROOT}" | tail -1
ARGS=(--batch "${BATCH:-200}" --connections "${CONNECTIONS:-8}" --workers "${WORKERS:-24}")
[ -n "${PRODUCTS:-}" ] && ARGS+=(--products ${PRODUCTS})
[ -n "${MAX_BATCHES:-}" ] && ARGS+=(--max-batches "${MAX_BATCHES}")
python -m oefm stream "${ARGS[@]}"
python -m oefm status
echo "Job finished: $(date)"
