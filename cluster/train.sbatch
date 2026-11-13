#!/bin/bash
#SBATCH --job-name=oefm_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 1 training of one arm on 4 A100s (step 2.9). Saves before the 48 h limit and resubmits
# itself up to CHAIN more times; each run resumes from data/runs/stage1_<ARM>/checkpoint.pt.
#   sbatch --account=191001-366746 --export=ALL,ARM=B,CHAIN=3 cluster/train.sbatch
# Extra settings: OVERRIDES="optim.per_gpu_batch=4 optim.steps=30000" (from the benchmark).

set -euo pipefail
export OEFM_ENV="${OEFM_GPU_ENV:-/scratch/ptolemy/users/${USER}/envs/oefm_gpu}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
ARM="${ARM:?set ARM=A, B or C}"
if [ "${CHAIN:-0}" -gt 0 ]; then
  next=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} --account="${SLURM_JOB_ACCOUNT}" \
         --export=ALL,ARM=${ARM},CHAIN=$((CHAIN - 1)) "${REPO_ROOT}/cluster/train.sbatch")
  echo "Queued follow-on job ${next} (CHAIN=$((CHAIN - 1)))"
fi
echo "Job started: $(date) | arm ${ARM}"
torchrun --standalone --nproc_per_node=4 -m oefm.model.train --arm "${ARM}" ${OVERRIDES:+--override ${OVERRIDES}}
echo "Job finished: $(date)"
