#!/bin/bash
#SBATCH --job-name=oefm_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Required benchmarks before the full runs (design document, "Compute and engineering"): for each
# arm, sequence length 8 / 16 / 24 and per-GPU batch, BENCH_STEPS steps each on 4 A100s:
# throughput, peak memory, data-loader wait, projected wall-clock for the configured steps.
# Results: data/runs/bench_*/benchmark_*.json and a summary at the end of the log.
#   sbatch --account=191001-366746 cluster/benchmark.sbatch
#   # subset: --export=ALL,ARMS="B",LENGTHS="16",BATCHES="2 4 8"

set -uo pipefail
export OEFM_ENV="${OEFM_GPU_ENV:-/scratch/ptolemy/users/${USER}/envs/oefm_gpu}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
ARMS="${ARMS:-A B C}"; LENGTHS="${LENGTHS:-8 16 24}"; BATCHES="${BATCHES:-2 4 8}"; BENCH_STEPS="${BENCH_STEPS:-200}"
for arm in ${ARMS}; do for L in ${LENGTHS}; do for b in ${BATCHES}; do
  echo "=== arm ${arm} length ${L} per-GPU batch ${b} $(date)"
  torchrun --standalone --nproc_per_node=4 -m oefm.model.train --arm "${arm}" --name "bench_${arm}" \
    --benchmark "${BENCH_STEPS}" --override data.length="${L}" optim.per_gpu_batch="${b}" data.num_workers=6 \
    || echo "FAILED (probably out of memory): arm ${arm} L ${L} batch ${b}"
done; done; done
python - <<'PY'
import glob, json, os
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(os.environ["OEFM_DATA_ROOT"], "runs", "bench_*", "benchmark_*.json")))]
print(f"{'arm':4s} {'L':>3s} {'b':>3s} {'seq/s':>8s} {'mem GB':>7s} {'wait':>6s} {'proj h':>7s}")
for r in rows:
    print(f"{r['arm']:4s} {r['length']:3d} {r['per_gpu_batch']:3d} {r['seq_per_s_global']:8.1f} {r['peak_mem_gb'] or 0:7.1f} "
          f"{r['loader_wait_share']:6.2f} {r['projected_hours_for_run']:7.1f}")
PY
echo "Job finished: $(date)"
