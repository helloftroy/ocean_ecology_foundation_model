#!/bin/bash
#SBATCH --job-name=oefm_env
#SBATCH --partition=service
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One-off: build the project's Python environment on Ptolemy scratch.
#
# Ptolemy scratch had no Python of its own (checked 2026-10-30), so this first installs a
# Miniforge base interpreter at /scratch/ptolemy/users/$USER/miniforge3 (only if it is not
# there), then creates a plain venv from it at /scratch/ptolemy/users/$USER/envs/oefm and
# pip-installs requirements.txt. No conda environments are created, and the installer runs with
# HOME pointed at scratch so nothing is written to the real $HOME.
#
#   cd /scratch/ptolemy/users/$USER/ocean_ecology_foundation_model
#   sbatch --account=191001-366746 cluster/setup_env.sbatch
#
# Overrides: OEFM_ENV (venv path), OEFM_BASE_PYTHON (use an existing Python >= 3.10 instead).

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
SCRATCH_USER_DIR="${SCRATCH_USER_DIR:-/scratch/ptolemy/users/${USER}}"
MINIFORGE="${MINIFORGE:-${SCRATCH_USER_DIR}/miniforge3}"
ENV_PATH="${OEFM_ENV:-${SCRATCH_USER_DIR}/envs/oefm}"
BASE_PYTHON="${OEFM_BASE_PYTHON:-${MINIFORGE}/bin/python3}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_USER_DIR}/envs/pip_cache}"
export XDG_CACHE_HOME="${REPO_ROOT}/.cache/xdg"
export TMPDIR="${REPO_ROOT}/.cache/tmp"
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" "$(dirname "${ENV_PATH}")" logs

case "${ENV_PATH}" in
  /scratch/*) ;;
  *) echo "Refusing to build an environment outside /scratch: ${ENV_PATH}" >&2; exit 2 ;;
esac

echo "Job started: $(date) on $(hostname)"
df -h "${SCRATCH_USER_DIR}" | tail -1

if [ ! -x "${BASE_PYTHON}" ]; then
  echo "Installing Miniforge into ${MINIFORGE}"
  installer="${TMPDIR}/Miniforge3-Linux-x86_64.sh"
  curl -fsSL --retry 5 -o "${installer}" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh"
  fake_home="${SCRATCH_USER_DIR}/.home_shim"
  mkdir -p "${fake_home}"
  HOME="${fake_home}" CONDA_PKGS_DIRS="${SCRATCH_USER_DIR}/envs/conda_pkgs" bash "${installer}" -b -p "${MINIFORGE}"
  rm -f "${installer}"
fi
echo "Base Python: ${BASE_PYTHON} ($("${BASE_PYTHON}" --version 2>&1))"

if [ -d "${ENV_PATH}" ] && [ ! -f "${ENV_PATH}/pyvenv.cfg" ]; then
  echo "Removing incomplete environment at ${ENV_PATH}"
  rm -rf "${ENV_PATH}"
fi
[ -f "${ENV_PATH}/pyvenv.cfg" ] || "${BASE_PYTHON}" -m venv "${ENV_PATH}"
"${ENV_PATH}/bin/python" -m pip install --upgrade pip
"${ENV_PATH}/bin/python" -m pip install -r "${REPO_ROOT}/requirements.txt"

# Check the pieces work together, not only that they import.
"${ENV_PATH}/bin/python" - <<'PY'
import sys, tempfile
import numpy, netCDF4, h5py, xarray, pandas, matplotlib, yaml, earthaccess, copernicusmarine
print("python", sys.version.split()[0], "| prefix", sys.prefix)
print("numpy", numpy.__version__, "| xarray", xarray.__version__, "| netCDF4", netCDF4.__version__,
      "| earthaccess", earthaccess.__version__, "| copernicusmarine", copernicusmarine.__version__)
with tempfile.TemporaryDirectory() as tmp:
    ds = xarray.Dataset({"x": (("lat", "lon"), numpy.ones((2, 3), "f4"))})
    ds.to_netcdf(f"{tmp}/t.nc", engine="netcdf4")
    assert float(xarray.open_dataset(f"{tmp}/t.nc")["x"].sum()) == 6.0
print("netCDF write/read check passed")
PY
cd "${REPO_ROOT}" && "${ENV_PATH}/bin/python" -m unittest discover -s tests -q
echo "Environment size: $(du -sh "${ENV_PATH}" | cut -f1)"
echo "Environment ready: ${ENV_PATH}"
echo "Job finished: $(date)"
