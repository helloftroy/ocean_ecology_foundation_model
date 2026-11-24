#!/bin/bash
#SBATCH --job-name=oefm_bgs_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Bio-GO-SHIP benchmark (step 3.4), part 1: frozen embeddings at every sample (1,977), windows causal
# and retro, PACE visible and dropped, for the runs in RUNS (trained on Florida: runs_root and
# stats_dir in config/biogoship.yaml point at data/). Writes data_biogoship/runs/eval_<TAG>/.
# Part 2 (scoring, CPU) is cluster/biogoship_bench.sbatch.
#   sbatch --account=191001-366746 --export=ALL,RUNS="stage1_A stage1_B stage1_C",TAG=stage1 cluster/biogoship_embed.sbatch

set -euo pipefail
export OEFM_CONFIG="${OEFM_CONFIG:-config/biogoship.yaml}"
export OEFM_DATA_ROOT="${OEFM_DATA_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}/data_biogoship}"
export OEFM_ENV="${OEFM_GPU_ENV:-/scratch/ptolemy/users/${USER}/envs/oefm_gpu}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
RUNS="${RUNS:-stage1_A stage1_B stage1_C}"; TAG="${TAG:-stage1}"
echo "Job started: $(date) | config ${OEFM_CONFIG} | data ${OEFM_DATA_ROOT} | runs: ${RUNS} | tag: ${TAG}"
python -m oefm.model.biogoship --runs ${RUNS} --tag "${TAG}" --steps embed --workers 14 ${EXTRA:-}
echo "Job finished: $(date)"
