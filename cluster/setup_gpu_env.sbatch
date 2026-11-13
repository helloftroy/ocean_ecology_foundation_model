#!/bin/bash
#SBATCH --job-name=oefm_gpuenv
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One-off: the GPU training environment (step 2.8). Python 3.12 (PyTorch wheels are mature there;
# the CPU environment's Python 3.14 is not used for training) from the scratch Miniforge, a venv at
# /scratch/ptolemy/users/$USER/envs/oefm_gpu, PyTorch with CUDA 12.4 wheels, then requirements.txt,
# so the same environment also runs the data code, embedding extraction and the probes.
# Runs on 'service' (internet). Then check the GPUs with cluster/gpu_check.sbatch.
#
#   sbatch --account=191001-366746 cluster/setup_gpu_env.sbatch
#
# Overrides: TORCH_VERSION (default 2.6.0), TORCH_INDEX (default .../whl/cu124).

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
SCRATCH_USER_DIR="${SCRATCH_USER_DIR:-/scratch/ptolemy/users/${USER}}"
MINIFORGE="${MINIFORGE:-${SCRATCH_USER_DIR}/miniforge3}"
PY312="${SCRATCH_USER_DIR}/envs/py312"
ENV_PATH="${OEFM_GPU_ENV:-${SCRATCH_USER_DIR}/envs/oefm_gpu}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
export PIP_CACHE_DIR="${SCRATCH_USER_DIR}/envs/pip_cache"
export CONDA_PKGS_DIRS="${SCRATCH_USER_DIR}/envs/conda_pkgs"
export TMPDIR="${REPO_ROOT}/.cache/tmp" PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}" logs
echo "Job started: $(date) on $(hostname)"
if [ ! -x "${MINIFORGE}/bin/conda" ]; then
  echo "Miniforge not found at ${MINIFORGE}: run cluster/setup_env.sbatch first." >&2; exit 2
fi
if [ ! -x "${PY312}/bin/python3.12" ]; then
  HOME="${SCRATCH_USER_DIR}/.home_shim" "${MINIFORGE}/bin/conda" create -y -p "${PY312}" python=3.12
fi
if [ ! -f "${ENV_PATH}/bin/activate" ]; then
  "${PY312}/bin/python3.12" -m venv "${ENV_PATH}"
fi
"${ENV_PATH}/bin/python" -m pip install --upgrade pip
"${ENV_PATH}/bin/python" -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
"${ENV_PATH}/bin/python" -m pip install -r requirements.txt
"${ENV_PATH}/bin/python" - <<'PY'
import sys, torch, numpy, zarr, sklearn
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| numpy", numpy.__version__, "| zarr", zarr.__version__, "| sklearn", sklearn.__version__)
PY
echo "Job finished: $(date)"
