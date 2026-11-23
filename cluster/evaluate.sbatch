#!/bin/bash
#SBATCH --job-name=oefm_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 1 evaluation (steps 2.10, 2.10b): frozen embeddings at the evaluation samples, ridge probes, PACE
# spectral retention, for the runs in RUNS (default: the three Stage 1 arms, final weights).
# Results in data/runs/eval_<TAG>/ (summary tables are printed at the end of the log).
#   sbatch --account=191001-366746 cluster/evaluate.sbatch
#   # robustness + amplitude test on the existing embeddings (2.10b):
#   sbatch --account=191001-366746 --export=ALL,EXTRA="--steps robust amplitude" cluster/evaluate.sbatch
#   # quick check on 300 samples and few PACE anchors, from the latest checkpoints:
#   sbatch --account=191001-366746 --export=ALL,TAG=check,WEIGHTS=checkpoint,EXTRA="--limit 300 --pace-anchors 100 50" cluster/evaluate.sbatch

set -euo pipefail
export OEFM_ENV="${OEFM_GPU_ENV:-/scratch/ptolemy/users/${USER}/envs/oefm_gpu}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
RUNS="${RUNS:-stage1_A stage1_B stage1_C}"; TAG="${TAG:-stage1}"; WEIGHTS="${WEIGHTS:-final}"
echo "Job started: $(date) | runs: ${RUNS} | weights: ${WEIGHTS} | tag: ${TAG}"
python -m oefm.model.evaluate --runs ${RUNS} --weights "${WEIGHTS}" --tag "${TAG}" --workers 14 ${EXTRA:-}
echo "Job finished: $(date)"
