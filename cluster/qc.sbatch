#!/bin/bash
#SBATCH --job-name=oefm_qc
#SBATCH --partition=gpu-a100-mig7
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Quality checks and indexing on whatever has been downloaded: inspect sample files of every
# product (structure, flags, mask scenarios, maps), build the observation index, compare
# overlapping sensors, and rewrite the report. No internet needed.
#
# No internet: runs on gpu-a100-mig7 (CPU only, no GPU requested); 'service' is for downloads only.
#
#   cd /scratch/ptolemy/users/$USER/ocean_ecology_foundation_model
#   sbatch --account=191001-366746 cluster/qc.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
echo "Job started: $(date)"
python -m oefm verify --products physical          # cheap: size/signature/checksum of physical files
python -m oefm inspect --per-product "${PER_PRODUCT:-3}"
python -m oefm index --workers "${SLURM_CPUS_PER_TASK:-8}"
python -m oefm overlap
python -m oefm zenith --workers "${SLURM_CPUS_PER_TASK:-8}"   # HISATZEN diagnostic
python -m oefm report
echo "Job finished: $(date)"
