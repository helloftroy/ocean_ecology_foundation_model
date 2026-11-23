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
# itself up to CHAIN more times; each run resumes from data/runs/<NAME>/checkpoint.pt.
#   sbatch --account=191001-366746 --export=ALL,ARM=B,CHAIN=3 cluster/train.sbatch
# Extra settings: OVERRIDES="optim.per_gpu_batch=4 optim.steps=30000" (from the benchmark).
# NAME: run name (default stage1_<ARM>), e.g. the second stage-1.5 reconstruction weight:
#   sbatch --account=191001-366746 --export=ALL,ARM=D,NAME=stage1_D3,OVERRIDES="loss.nu=0.3",CHAIN=1 cluster/train.sbatch

set -euo pipefail
export OEFM_ENV="${OEFM_GPU_ENV:-/scratch/ptolemy/users/${USER}/envs/oefm_gpu}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
ARM="${ARM:?set ARM=A, B, C or D}"
NAME="${NAME:-stage1_${ARM}}"
if [ "${CHAIN:-0}" -gt 0 ]; then
  next=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} --account="${SLURM_JOB_ACCOUNT}" \
         --export=ALL,ARM=${ARM},NAME=${NAME},CHAIN=$((CHAIN - 1)) "${REPO_ROOT}/cluster/train.sbatch")
  echo "Queued follow-on job ${next} (CHAIN=$((CHAIN - 1)))"
fi
echo "Job started: $(date) | arm ${ARM} | run ${NAME} | overrides ${OVERRIDES:-none}"
torchrun --standalone --nproc_per_node=4 -m oefm.model.train --arm "${ARM}" --name "${NAME}" ${OVERRIDES:+--override ${OVERRIDES}}
# finished all steps: cancel the queued follow-on instead of letting it start and do nothing
if [ -n "${next:-}" ] && [ -f "${OEFM_DATA_ROOT}/runs/${NAME}/weights_final.pt" ]; then
  scancel "${next}" && echo "Training complete: cancelled follow-on job ${next}"
fi
echo "Job finished: $(date)"
