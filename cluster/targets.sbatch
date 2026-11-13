#!/bin/bash
#SBATCH --job-name=oefm_targets
#SBATCH --partition=service
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Steps 2.6-2.7 (cluster_instructions.md): download PISCES nutrients (Copernicus Marine; needs the
# internet, so it runs on 'service' and counts as a download job: no other download at the same
# time), build the per-tile nutrient targets, and join the evaluation targets copied from the
# laptop with the usable-imagery counts. Needs the full prep.sbatch run (tiles, evalcover) first.
# Resumable: resubmit after a failure.
#
#   sbatch --account=191001-366746 cluster/targets.sbatch
#   # only some steps: --export=ALL,STEPS="build evalsets"

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
STEPS="${STEPS:-inventory download build evalsets}"
echo "Job started: $(date) | steps: ${STEPS}"
for step in ${STEPS}; do
  echo "=== ${step} $(date)"
  case "${step}" in
    inventory) python -m oefm inventory --products pisces_nutrients ;;
    download)  python -m oefm download --products pisces_nutrients --connections 2 ;;
    build)     python -m oefm targets --workers "${SLURM_CPUS_PER_TASK:-8}" ;;
    evalsets)  python -m oefm evalsets ;;
    *) echo "unknown step ${step}" >&2; exit 1 ;;
  esac
done
python -m oefm status | grep -E "product|pisces" || true
echo "Job finished: $(date)"
