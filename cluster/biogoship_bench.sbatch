#!/bin/bash
#SBATCH --job-name=oefm_bgs_bench
#SBATCH --partition=service
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Bio-GO-SHIP benchmark (step 3.4), part 2: the Granite protocol (the bundle's benchmark_eval.run) on
# the embeddings from biogoship_embed.sbatch: Granite reproduction check, paired benchmark (Granite
# rerun on the same tiles), PACE-era subset, extended set, report. CPU only (~10-20 h).
#   sbatch --account=191001-366746 --export=ALL,TAG=stage1 cluster/biogoship_bench.sbatch
#   # only the report from finished CSVs: --export=ALL,TAG=stage1,STEPS=report

set -euo pipefail
export OEFM_CONFIG="${OEFM_CONFIG:-config/biogoship.yaml}"
export OEFM_DATA_ROOT="${OEFM_DATA_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}/data_biogoship}"
export OEFM_ENV="${OEFM_GPU_ENV:-/scratch/ptolemy/users/${USER}/envs/oefm_gpu}"   # has scikit-learn (checked at setup)
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
TAG="${TAG:-stage1}"; STEPS="${STEPS:-paired pace extended report}"
echo "Job started: $(date) | config ${OEFM_CONFIG} | data ${OEFM_DATA_ROOT} | tag ${TAG} | steps ${STEPS}"
python -m oefm.model.biogoship --tag "${TAG}" --steps ${STEPS} ${EXTRA:-}
echo "Job finished: $(date)"
