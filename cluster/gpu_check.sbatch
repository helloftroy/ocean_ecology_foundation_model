#!/bin/bash
#SBATCH --job-name=oefm_gpucheck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Short check of the GPU environment on 4 A100s: driver, CUDA visibility, bf16 attention, NCCL
# all-reduce across the 4 GPUs, and one tiny training step of each arm. Ends in a few minutes.
#   sbatch --account=191001-366746 cluster/gpu_check.sbatch

set -euo pipefail
export OEFM_ENV="${OEFM_GPU_ENV:-/scratch/ptolemy/users/${USER}/envs/oefm_gpu}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/cluster/env.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'PY'
import torch, torch.nn.functional as F
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
q = torch.randn(8, 8, 1024, 64, device="cuda", dtype=torch.bfloat16)
m = torch.rand(8, 1, 1, 1024, device="cuda") > 0.3
print("bf16 SDPA with key mask OK:", F.scaled_dot_product_attention(q, q, q, attn_mask=m).shape)
PY
torchrun --standalone --nproc_per_node=4 - <<'PY'
import torch, torch.distributed as dist, os
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
x = torch.ones(1, device="cuda") * (r + 1); dist.all_reduce(x)
print(f"rank {r}: all_reduce sum = {x.item()} (expect 10)"); dist.destroy_process_group()
PY
TINY="model.dim=64 model.heads=4 model.pred_dim=48 model.pred_heads=4 model.spatial_depth=1 model.temporal_depth=1 model.pred_temporal_depth=1 model.pred_spatial_depth=1 model.decoder_depth=1 data.num_workers=2 optim.per_gpu_batch=2 optim.effective_batch=8 optim.steps=4 run.log_every=2 run.rank_every=2"
for arm in A B C; do
  torchrun --standalone --nproc_per_node=4 -m oefm.model.train --arm "${arm}" --name gpucheck_${arm} --benchmark 12 --override ${TINY}
done
echo "Job finished: $(date)"
