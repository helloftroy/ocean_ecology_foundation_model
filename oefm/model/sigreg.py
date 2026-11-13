"""SIGReg: sliced Epps-Pulley test against an isotropic Gaussian (LeJEPA, arXiv 2511.08544).

Follows the official library (github.com/galilai-group/lejepa, files lejepa/univariate/epps_pulley.py
and lejepa/multivariate/slicing.py, fetched 2026-11-12) line by line:
- Epps-Pulley on t = linspace(0, t_max, n_points) with trapezoid weights doubled to cover the
  mirrored negative half-line (the integrand is even), Gaussian window phi(t) = exp(-t^2 / 2), which
  is both the N(0,1) characteristic function and the weight; err = (mean cos - phi)^2 + (mean sin)^2;
  statistic = (err @ weights) * N * world_size (N = local batch).
- Multi-GPU: mean cos and mean sin are each averaged across ranks with a differentiable all_reduce.
- Slicing: directions A ~ N(0, I) of shape (D, num_slices), columns normalised to unit length,
  drawn from a generator reseeded every call with a step counter that is synchronised across ranks
  (all_reduce MAX) and then incremented, so every rank draws the same directions; mean over slices.
Design-document settings: 17 points on [-5, 5] (t_max = 5 with the doubled weights is the same
integral), 1,024 slices, <= 8,192 tokens per step across GPUs.
"""

from __future__ import annotations

import torch
from torch import distributed as dist


def _world() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _all_reduce_avg(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn import all_reduce as functional_all_reduce

        return functional_all_reduce(x, op=dist.ReduceOp.AVG)
    return x


class EppsPulley(torch.nn.Module):
    def __init__(self, t_max: float = 5.0, n_points: int = 17) -> None:
        super().__init__()
        assert n_points % 2 == 1
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = t.square().mul(0.5).neg().exp()
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (*, N, K) -> (*, K)
        n = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = _all_reduce_avg(torch.cos(x_t).mean(-3))
        sin_mean = _all_reduce_avg(torch.sin(x_t).mean(-3))
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * n * _world()


class SIGReg(torch.nn.Module):
    def __init__(self, num_slices: int = 1024, t_max: float = 5.0, n_points: int = 17) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.test = EppsPulley(t_max, n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._gen = None
        self._gen_dev = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (N, D), float32
        with torch.no_grad():
            step = self.global_step.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(step, op=dist.ReduceOp.MAX)
            if self._gen is None or self._gen_dev != x.device:
                self._gen = torch.Generator(device=x.device)
                self._gen_dev = x.device
            self._gen.manual_seed(int(step.item()))
            a = torch.randn((x.size(-1), self.num_slices), device=x.device, generator=self._gen)
            a /= a.norm(p=2, dim=0)
            self.global_step.add_(1)
        return self.test(x.float() @ a).mean()


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn import all_reduce as functional_all_reduce

        return functional_all_reduce(x, op=dist.ReduceOp.SUM)
    return x


class GroupSIGReg(torch.nn.Module):
    """SIGReg per view group (design document, "Modality balancing"; Mol-JEPA applies it per
    modality), averaged with equal weight over the groups present; groups with fewer than
    `min_tokens` tokens in a step (counted across GPUs) are pooled together for that step.

    Multi-GPU: different GPUs may hold different groups, so every rank walks the same fixed group
    list and all-reduces SUMS of cos/sin over its tokens (zeros if it has none) plus the token
    counts; the empirical characteristic function is then the global mean. With equal token counts
    per rank this equals the reference implementation (mean of per-rank means, x N x world size);
    with unequal counts it is the exact global statistic. The slicing directions are drawn once per
    step from the synchronised step counter, as in the reference.
    """

    def __init__(self, groups: list[str], num_slices: int = 1024, t_max: float = 5.0, n_points: int = 17,
                 min_tokens: int = 256) -> None:
        super().__init__()
        self.groups = list(groups)
        self.min_tokens = min_tokens
        self.slicer = SIGReg(num_slices, t_max, n_points)  # holds the step counter and generator

    def _directions(self, d: int, device) -> torch.Tensor:
        s = self.slicer
        with torch.no_grad():
            step = s.global_step.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(step, op=dist.ReduceOp.MAX)
            if s._gen is None or s._gen_dev != device:
                s._gen = torch.Generator(device=device)
                s._gen_dev = device
            s._gen.manual_seed(int(step.item()))
            a = torch.randn((d, s.num_slices), device=device, generator=s._gen)
            a /= a.norm(p=2, dim=0)
            s.global_step.add_(1)
        return a

    def forward(self, per: dict[str, torch.Tensor], dim: int, device) -> tuple[torch.Tensor, dict]:
        ep = self.slicer.test
        a = self._directions(dim, device)
        local = torch.tensor([float(per[g].shape[0]) if g in per else 0.0 for g in self.groups], device=device)
        counts = local.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        own = [g for g, c in zip(self.groups, counts.tolist()) if c >= self.min_tokens]
        small = [g for g, c in zip(self.groups, counts.tolist()) if 0 < c < self.min_tokens]
        effective = [(g, [g]) for g in own] + ([("pooled", small)] if small else [])
        stats, logs = [], {}
        for name, members in effective:
            xs = [per[g] for g in members if g in per and per[g].shape[0] > 0]
            x = torch.cat(xs).float() if xs else torch.zeros(0, dim, device=device)
            x_t = (x @ a).unsqueeze(-1) * ep.t                                   # (N, K, T)
            cs = _all_reduce_sum(torch.cos(x_t).sum(0))
            sn = _all_reduce_sum(torch.sin(x_t).sum(0))
            n = float(sum(counts[self.groups.index(g)].item() for g in members))
            err = (cs / n - ep.phi).square() + (sn / n).square()
            st = ((err @ ep.weights) * n).mean()
            stats.append(st)
            logs[name] = st.detach()
        total = torch.stack(stats).mean() if stats else torch.zeros((), device=device)
        return total, logs
