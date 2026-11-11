#!/bin/bash
#SBATCH --job-name=oefm_smoke
#SBATCH --partition=service
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# End-to-end acquisition test for every satellite and physical product, on a few dates:
# inventory (one month in the core period) -> download N files per product -> verify ->
# inspect + maps -> report. Uses its own data root (data_smoke/) so the real manifest is
# untouched.
#
#   cd /scratch/ptolemy/users/$USER/ocean_ecology_foundation_model
#   sbatch --account=191001-366746 cluster/smoke_test.sbatch
#   cat logs/oefm_smoke_*.out; less data_smoke/reports/pilot_acquisition_report.md
#
# Variables: PER_PRODUCT (default 2), SMOKE_START / SMOKE_END (default 2025-06-08..2025-06-14,
# a week when PACE, OLCI-A, MODIS and all three VIIRS have data).

set -euo pipefail
export OEFM_DATA_ROOT="${OEFM_DATA_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}/data_smoke}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
S="${SMOKE_START:-2025-06-08}"
E="${SMOKE_END:-2025-06-14}"

# A small config: same regions and products, one short core window, no historical windows,
# and a 3-day physical lead so the physical files stay few.
SMOKE_CFG="${OEFM_DATA_ROOT}/smoke_config.yaml"
python - "$S" "$E" "${SMOKE_CFG}" <<'PY'
import sys, yaml
start, end, dest = sys.argv[1:]
cfg = yaml.safe_load(open("config/pilot.yaml"))
cfg["time"]["core"] = {"start": start, "end": end}
cfg["time"]["historical"]["years"] = []
cfg["time"]["events"] = []
cfg["time"]["physical_lead_days"] = 1
yaml.safe_dump(cfg, open(dest, "w"), sort_keys=False)
PY
export OEFM_CONFIG="${SMOKE_CFG}"

echo "Job started: $(date) | smoke window ${S}..${E}"
python -m oefm catalog
python -m oefm inventory
python -m oefm download --per-product "${PER_PRODUCT:-2}" --connections 8
python -m oefm status
python -m oefm inspect --per-product "${PER_PRODUCT:-2}"
python -m oefm index
python -m oefm overlap
python -m oefm report
echo "Report: ${OEFM_DATA_ROOT}/reports/pilot_acquisition_report.md"
echo "Job finished: $(date)"
