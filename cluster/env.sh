#!/usr/bin/env bash
# Shared setup for every Ptolemy job. Source it from the repo root:  source cluster/env.sh
#
# Override at submit time, e.g.  sbatch --export=ALL,OEFM_ENV=/path/to/env cluster/download.sbatch

# SLURM runs a spooled copy of the job script, so $0 cannot locate the repo.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "${REPO_ROOT}/oefm" ] || [ ! -d "${REPO_ROOT}/cluster" ]; then
  echo "Run this from the ocean_ecology_foundation_model repo root (cd there before sbatch)." >&2
  exit 1
fi
cd "${REPO_ROOT}"

# Ptolemy scratch (GPFS, 300 TB). Data, caches and the Python environment all live here,
# never in $HOME.
SCRATCH_USER_DIR="${SCRATCH_USER_DIR:-/scratch/ptolemy/users/${USER}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${REPO_ROOT}}"
export OEFM_DATA_ROOT="${OEFM_DATA_ROOT:-${SCRATCH_ROOT}/data}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/.cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRATCH_ROOT}/.cache/mpl}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_USER_DIR}/envs/pip_cache}"
export TMPDIR="${TMPDIR_OVERRIDE:-${SCRATCH_ROOT}/.cache/tmp}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${TMPDIR}" "${REPO_ROOT}/logs" "${OEFM_DATA_ROOT}"

# Python environment, activated by full path (built by cluster/setup_env.sbatch).
OEFM_ENV="${OEFM_ENV:-${SCRATCH_USER_DIR}/envs/oefm}"
if [ ! -f "${OEFM_ENV}/bin/activate" ]; then
  echo "Python environment not found at ${OEFM_ENV}; run: sbatch --account=191001-366746 cluster/setup_env.sbatch" >&2
  exit 2
fi
set +u
source "${OEFM_ENV}/bin/activate"
set -u
_actual_prefix="$(python -c 'import sys; print(sys.prefix)')"
if [ "$(cd "${_actual_prefix}" && pwd -P)" != "$(cd "${OEFM_ENV}" && pwd -P)" ]; then
  echo "Environment activation did not take effect: python is ${_actual_prefix}, expected ${OEFM_ENV}." >&2
  exit 2
fi

# Archive credentials (gitignored). See credentials.sh.example.
if [ -f "${REPO_ROOT}/credentials.sh" ]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/credentials.sh"
fi

echo "Host: $(hostname) | repo: ${REPO_ROOT} | data: ${OEFM_DATA_ROOT} | python: $(command -v python) ($(python --version 2>&1))"
