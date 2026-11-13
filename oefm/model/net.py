"""Encoder, predictor and decoders: design document "Implementation specification (v1)".

Shapes: a batch holds B sequences of up to T observations; every observation is a 32 x 32 grid of
1.2 km cells (N = 1,024 tokens). Observations arrive grouped by product (same band set), see
oefm/model/data.py. Token width D (512), predictor width Dp (384).

- SpectralTokenizer: an MLP on Fourier features of wavelength, conditioned on the view type
  (Rrs or surface rhos, so the same wavelength can get different kernels in the two views), gives
  w(lambda) on one 1 nm grid, 340-2300 nm (PACE SFREFL reaches ~2258 nm); each band's kernel is w
  integrated against the band's SRF, so a heritage band and the PACE bands under it get consistent
  kernels. Token = sum over valid bands of kernel x value / number of valid bands, with NO
  LayerNorm on the token (design document: LayerNorm would divide out overall amplitude, making a
  spectrum and a scaled copy identical; normalisation happens only inside the pre-LN transformer
  branches, so the residual stream keeps magnitude). Two amplitude features are added through a
  small MLP: the mean transformed value over valid bands and asinh of the mean raw reflectance.
  Two heads on one trunk: D (1 x 1 patch at 1.2 km) and 16 D (4 x 4 OLCI patch at 300 m). Added:
  quality channels (linear 9 -> D: HISATZEN and the eight soft flags), view-type embedding, 2-D
  sine-cosine cell position (local, not geographic).
- Spatial stage: 6 pre-LN transformer blocks per observation, global attention over its visible
  cells (missing / hidden cells are excluded from attention).
- Temporal stage: 4 blocks per cell across that cell's visible observations, with Fourier-encoded
  dt (days relative to t0). dt is added here, not in the tokenizer, so that a target token (the
  spatial-stage output of an observation encoded alone) does not depend on the chosen t0.
- Predictor (arms B, C): context tokens projected to Dp plus one mask token per hidden
  (observation, cell) carrying view type, band signature (mean Fourier features of the hidden
  observation's bands), dt and cell position; 2 temporal then 2 spatial blocks; projection to D.
- Arm A decoder: the same mask-token construction, 2 blocks per cell, then a dot product with a
  band query MLP(Fourier(lambda)) per band (16 outputs per band for OLCI 300 m pixels).
- PACE residual decoder (arm C): MLP D -> 512 -> number of PACE bands on temporal-stage tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

VIEW_TYPES = {"rrs": 0, "surface": 1, "physics": 2, "static": 3, "nutrient": 4}
N_QUALITY = 9          # HISATZEN + ATMWARN, MAXAERITER, LOWLW, ABSAER, COCCOLITH, TURBIDW, COASTZ, OPSHAL
AMP_SCALE = 1e-3       # sr^-1: asinh(mean raw Rrs / 1e-3) ~ linear below 1e-3, logarithmic above


def fourier(x: torch.Tensor, n: int, lo: float, hi: float) -> torch.Tensor:
    """[..., 2n] sin/cos features at n log-spaced periods between lo and hi (units of x)."""
    periods = torch.logspace(math.log10(lo), math.log10(hi), n, device=x.device, dtype=torch.float32)
    a = 2 * math.pi * x.float().unsqueeze(-1) / periods
    return torch.cat([a.sin(), a.cos()], dim=-1)


def sincos_2d(n: int, dim: int) -> torch.Tensor:
    """Fixed [n*n, dim] position code for an n x n grid (row-major)."""
    q = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(q, dtype=torch.float32) / q))
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    out = [torch.sin(yy.reshape(-1, 1) * omega), torch.cos(yy.reshape(-1, 1) * omega),
           torch.sin(xx.reshape(-1, 1) * omega), torch.cos(xx.reshape(-1, 1) * omega)]
    pe = torch.cat(out, dim=1)
    return F.pad(pe, (0, dim - pe.shape[1]))


class DropPath(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class Block(nn.Module):
    """Pre-LN transformer block; attention restricted by a key mask [rows, L] (True = attend)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        r, length, d = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(r, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        x = x + self.dp(self.proj(a.transpose(1, 2).reshape(r, length, d)))
        return x + self.dp(self.mlp(self.n2(x)))


class Stack(nn.Module):
    def __init__(self, depth: int, dim: int, heads: int, mlp_ratio: float, drop_path: float, ckpt: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, drop_path) for _ in range(depth)])
        self.ckpt = ckpt

    def forward(self, x, key_mask):
        for b in self.blocks:
            x = checkpoint(b, x, key_mask, use_reentrant=False) if (self.ckpt and self.training) else b(x, key_mask)
        return x


PER_OBS = ("values", "band_mask", "band_mask_ctx", "quality", "cell_valid", "ctx_cells", "tgt_cells", "seq", "slot",
           "dt", "pace_resid", "pace_resid_mask", "raw_mean")


def subset(g: dict, sel: torch.Tensor) -> dict:
    """Select observations of a product group (per-observation fields only; band metadata kept)."""
    return {k: (v[sel] if k in PER_OBS else v) for k, v in g.items()}


def _rows(x: torch.Tensor, mask: torch.Tensor, stack: Stack) -> torch.Tensor:
    """Run a stack only on rows with at least one valid key; others pass through unchanged."""
    keep = mask.any(dim=1)
    if keep.all():
        return stack(x, mask)
    out = x.clone()
    if keep.any():
        out[keep] = stack(x[keep], mask[keep])
    return out


class SpectralTokenizer(nn.Module):
    def __init__(self, dim: int, n_fourier: int = 32, hidden: int = 256, olci_patch: int = 4) -> None:
        super().__init__()
        self.dim, self.n_fourier, self.p2 = dim, n_fourier, olci_patch * olci_patch
        self.trunk = nn.Sequential(nn.Linear(2 * n_fourier + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.amplitude = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, dim))
        self.head1 = nn.Linear(hidden, dim)
        self.head4 = nn.Linear(hidden, dim * self.p2)
        self.quality = nn.Linear(N_QUALITY, dim)
        self.view = nn.Embedding(len(VIEW_TYPES), dim)
        self.register_buffer("pos", sincos_2d(32, dim), persistent=False)

    def kernels(self, srf: torch.Tensor, grid: torch.Tensor, patch: bool, view: int = 0) -> torch.Tensor:
        """srf [C, G] on grid [G] nm -> [C, D] or [C, 16, D]; w(lambda) is evaluated only where
        some band responds."""
        cols = srf.sum(0) > 0
        g, r = grid[cols], srf[:, cols].float()
        onehot = torch.zeros(g.shape[0], 2, device=g.device)
        onehot[:, min(view, 1)] = 1.0
        h = self.trunk(torch.cat([fourier(g, self.n_fourier, 2.0, 5000.0), onehot], dim=-1))
        w = self.head4(h).reshape(g.shape[0], self.p2, self.dim) if patch else self.head1(h)
        return torch.einsum("cg,g...->c...", r, w)

    @staticmethod
    def _amp(x: torch.Tensor, m: torch.Tensor, raw_mean: torch.Tensor, dims) -> torch.Tensor:
        """[..., 2]: mean transformed value over valid bands; asinh of the mean raw reflectance
        (in units of AMP_SCALE, so typical ocean values span the asinh's log range)."""
        mean = (x * m).sum(dims) / m.sum(dims).clamp(min=1)
        return torch.stack([mean, torch.asinh(torch.nan_to_num(raw_mean.float()) / AMP_SCALE)], dim=-1)

    def forward(self, g: dict, band_mask: torch.Tensor) -> torch.Tensor:
        """Tokens [n, 1024, D] for one product group (band_mask may differ from the stored one,
        e.g. after spectral dropout)."""
        x = torch.nan_to_num(g["values"].float()) * band_mask
        if g["olci"]:
            n, c = x.shape[:2]
            k = self.kernels(g["srf"], g["grid"], True, g["view"])               # [C, 16, D]
            xp = x.reshape(n, c, 32, 4, 32, 4).permute(0, 2, 4, 1, 3, 5).reshape(n, 32, 32, c, 16)
            mp = band_mask.reshape(n, c, 32, 4, 32, 4).permute(0, 2, 4, 1, 3, 5).reshape(n, 32, 32, c, 16).float()
            pre = torch.einsum("nijcp,cpd->nijd", xp, k) / mp.sum((-1, -2)).clamp(min=1).unsqueeze(-1)
            amp = self._amp(xp, mp, g["raw_mean"].reshape(n, 32, 32), (-1, -2))
            tok = (pre + self.amplitude(amp)).reshape(n, 1024, self.dim)
        else:
            k = self.kernels(g["srf"], g["grid"], False, g["view"])              # [C, D]
            pre = torch.einsum("nchw,cd->nhwd", x, k) / band_mask.sum(1).clamp(min=1).unsqueeze(-1)
            n = x.shape[0]
            amp = self._amp(x.permute(0, 2, 3, 1), band_mask.permute(0, 2, 3, 1).float(), g["raw_mean"].reshape(n, 32, 32), -1)
            tok = (pre + self.amplitude(amp)).reshape(n, 1024, self.dim)
        q = g["quality"].float().reshape(x.shape[0], N_QUALITY, 1024).transpose(1, 2)
        return tok + self.quality(q) + self.view.weight[g["view"]] + self.pos


class Model(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        m = cfg["model"]
        d, dp = m["dim"], m["pred_dim"]
        self.arm = cfg["arm"]
        self.d, self.dp = d, dp
        self.tok = SpectralTokenizer(d, m["n_fourier"])
        self.spatial = Stack(m["spatial_depth"], d, m["heads"], m["mlp_ratio"], m["drop_path"], m["checkpoint"])
        self.temporal = Stack(m["temporal_depth"], d, m["heads"], m["mlp_ratio"], m["drop_path"], m["checkpoint"])
        self.dt_ctx = nn.Linear(32, d)
        if self.arm in ("B", "C"):
            self.p_in = nn.Linear(d, dp)
            self.p_temporal = Stack(m["pred_temporal_depth"], dp, m["pred_heads"], m["mlp_ratio"], 0.0, m["checkpoint"])
            self.p_spatial = Stack(m["pred_spatial_depth"], dp, m["pred_heads"], m["mlp_ratio"], 0.0, m["checkpoint"])
            self.p_out = nn.Linear(dp, d)
        if self.arm == "A":
            self.p_in = nn.Linear(d, dp)
            self.dec = Stack(m["decoder_depth"], dp, m["pred_heads"], m["mlp_ratio"], 0.0, m["checkpoint"])
            self.band_query = nn.Sequential(nn.Linear(2 * m["n_fourier"], 256), nn.GELU(), nn.Linear(256, dp))
            self.sub16 = nn.Linear(dp, 16 * dp)
        if self.arm in ("A", "B", "C"):
            self.mask_token = nn.Parameter(torch.zeros(dp))
            self.m_view = nn.Embedding(len(VIEW_TYPES), dp)
            self.m_band = nn.Linear(2 * m["n_fourier"], dp)
            self.m_dt = nn.Linear(32, dp)
            self.register_buffer("m_pos", sincos_2d(32, dp), persistent=False)
        if self.arm == "C":
            self.resid = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, m["pace_bands"]))
        self.n_fourier = m["n_fourier"]

    # ----------------------------------------------------------------- pieces
    def _dt(self, dt):
        return fourier(dt, 16, 0.1, 64.0)

    def encode_context(self, batch: dict):
        """Temporal-stage tokens on the [B, T, 1024] grid, the grid mask, and spatial outputs."""
        B, T = batch["B"], batch["T"]
        d = self.d
        grid = torch.zeros(B, T, 1024, d, device=batch["device"], dtype=batch["dtype"])
        gmask = torch.zeros(B, T, 1024, dtype=torch.bool, device=batch["device"])
        dts = torch.zeros(B, T, device=batch["device"])
        for g in batch["groups"]:
            sel = g["ctx_cells"].any(1)
            if not sel.any():
                continue
            gg = subset(g, sel)
            x = self.tok(gg, gg["band_mask_ctx"])
            x = _rows(x, gg["ctx_cells"], self.spatial)
            grid[gg["seq"], gg["slot"]] = x.to(grid.dtype)
            gmask[gg["seq"], gg["slot"]] = gg["ctx_cells"]
            dts[gg["seq"], gg["slot"]] = gg["dt"].float()
        # temporal: per cell across observations
        h = grid + self.dt_ctx(self._dt(dts))[:, :, None, :].to(grid.dtype)
        h = h.permute(0, 2, 1, 3).reshape(B * 1024, T, d)
        km = gmask.permute(0, 2, 1).reshape(B * 1024, T)
        h = _rows(h, km, self.temporal)
        z = h.reshape(B, 1024, T, d).permute(0, 2, 1, 3)
        return z, gmask, dts

    def encode_targets(self, batch: dict):
        """Spatial-stage outputs of hidden observations, each encoded alone with all valid cells."""
        outs = []
        for g in batch["groups"]:
            sel = g["tgt_cells"].any(1)
            if not sel.any():
                continue
            gg = subset(g, sel)
            x = self.tok(gg, gg["band_mask"])
            x = _rows(x, gg["cell_valid"], self.spatial)
            outs.append((gg, x))
        return outs

    def _mask_grid(self, batch, z, gmask, dts):
        """Predictor/decoder input grid [B, T, 1024, Dp] with mask tokens at target positions."""
        B, T = batch["B"], batch["T"]
        p = torch.zeros(B, T, 1024, self.dp, device=z.device, dtype=z.dtype)
        p[gmask] = self.p_in(z[gmask])
        tmask = torch.zeros_like(gmask)
        for g in batch["groups"]:
            sel = g["tgt_cells"].any(1)
            if not sel.any():
                continue
            seq, slot, cells = g["seq"][sel], g["slot"][sel], g["tgt_cells"][sel]
            band_sig = self.m_band(fourier(g["wl"], self.n_fourier, 2.0, 2000.0).mean(0))
            base = (self.mask_token + self.m_view.weight[g["view"]] + band_sig)[None, None, :] \
                + self.m_dt(self._dt(g["dt"][sel]))[:, None, :] + self.m_pos[None]
            cur = p[seq, slot]
            p[seq, slot] = torch.where(cells[..., None], base.to(p.dtype), cur)
            tmask[seq, slot] |= cells
        return p, tmask

    def predict(self, batch, z, gmask, dts):
        p, tmask = self._mask_grid(batch, z, gmask, dts)
        B, T = batch["B"], batch["T"]
        valid = gmask | tmask
        h = p.permute(0, 2, 1, 3).reshape(B * 1024, T, self.dp)
        h = _rows(h, valid.permute(0, 2, 1).reshape(B * 1024, T), self.p_temporal)
        h = h.reshape(B, 1024, T, self.dp).permute(0, 2, 1, 3).reshape(B * T, 1024, self.dp)
        h = _rows(h, valid.reshape(B * T, 1024), self.p_spatial)
        return self.p_out(h.reshape(B, T, 1024, self.dp)), tmask

    def decode_values(self, batch, z, gmask, dts):
        """Arm A: predicted transformed values for every hidden (observation, cell)."""
        p, tmask = self._mask_grid(batch, z, gmask, dts)
        B, T = batch["B"], batch["T"]
        valid = gmask | tmask
        h = p.permute(0, 2, 1, 3).reshape(B * 1024, T, self.dp)
        h = _rows(h, valid.permute(0, 2, 1).reshape(B * 1024, T), self.dec)
        return h.reshape(B, 1024, T, self.dp).permute(0, 2, 1, 3), tmask

    # ----------------------------------------------------------------- losses
    @staticmethod
    def balance(per: dict, mode: str, weights: dict | None) -> tuple[torch.Tensor, dict]:
        """Combine per-modality squared errors. per: family -> 1-D tensor of errors (one value per
        valid target token or value). 'modality': each family's mean (normalised by its own valid
        targets), then a weighted mean over the families present, weights from config
        (default 1). 'token': one mean over all targets (large modalities dominate)."""
        logs = {f"{k}": e.mean().detach() for k, e in per.items() if e.numel()}
        if mode == "token":
            return torch.cat(list(per.values())).mean(), logs
        num, den = 0.0, 0.0
        for k, e in per.items():
            if e.numel() == 0:
                continue
            w = float((weights or {}).get(k, 1.0))
            num = num + w * e.mean()
            den += w
        return num / max(den, 1e-12), logs

    def forward(self, batch: dict, sigreg=None, lam: float = 0.05, mu: float = 0.1,
                sigreg_tokens: int = 8192, balance: str = "modality", weights: dict | None = None) -> dict:
        z, gmask, dts = self.encode_context(batch)
        out: dict = {}
        if self.arm in ("B", "C"):
            zhat, tmask = self.predict(batch, z, gmask, dts)
            per: dict[str, list] = {}
            tgt_by: dict[str, list] = {}
            tgts = []
            for gg, x in self.encode_targets(batch):
                cells = gg["tgt_cells"]
                zp, zt_g = zhat[gg["seq"], gg["slot"]][cells].float(), x[cells].float()
                per.setdefault(gg["family"], []).append(((zp - zt_g) ** 2).mean(-1))
                tgt_by.setdefault(gg["family"], []).append(zt_g)
                tgts.append(zt_g)
            zt = torch.cat(tgts)
            l_pred, fam = self.balance({k: torch.cat(v) for k, v in per.items()}, balance, weights)
            out.update({f"pred_{k}": v for k, v in fam.items()})
            if hasattr(sigreg, "groups"):  # per view group (design document), token cap split over groups
                cap = max(1, sigreg_tokens // max(1, len(sigreg.groups)))
                per_t = {}
                for k, v in tgt_by.items():
                    v = torch.cat(v)
                    per_t[k] = v if v.shape[0] <= cap else v[torch.randperm(v.shape[0], device=v.device)[:cap]]
                l_sig, slog = sigreg(per_t, self.d, zt.device)
                out.update({f"sigreg_{k}": v for k, v in slog.items()})
            else:
                sub = zt if zt.shape[0] <= sigreg_tokens else zt[torch.randperm(zt.shape[0], device=zt.device)[:sigreg_tokens]]
                l_sig = sigreg(sub)
            loss = (1 - lam) * l_pred + lam * l_sig
            out.update(pred=l_pred.detach(), sigreg=l_sig.detach(), n_targets=zt.shape[0])
            if self.arm == "C":
                l_res = self.pace_residual_loss(batch, z)
                if l_res is not None:
                    loss = loss + mu * l_res
                    out["pace_residual"] = l_res.detach()
            out["loss"] = loss
        elif self.arm == "A":
            h, tmask = self.decode_values(batch, z, gmask, dts)
            per: dict[str, list] = {}
            n = 0
            for g in batch["groups"]:
                sel = g["tgt_cells"].any(1)
                if not sel.any():
                    continue
                q = self.band_query(fourier(g["wl"], self.n_fourier, 2.0, 2000.0))      # [C, Dp]
                tok = h[g["seq"][sel], g["slot"][sel]]                                  # [n, 1024, Dp]
                cells = g["tgt_cells"][sel]
                vals = g["values"][sel].float()
                bm = g["band_mask"][sel]
                if g["olci"]:
                    sub = self.sub16(tok).reshape(*tok.shape[:2], 16, self.dp)          # [n, 1024, 16, Dp]
                    pred = torch.einsum("nlpd,cd->nlcp", sub, q)                         # [n, 1024, C, 16]
                    c = vals.shape[1]
                    tv = vals.reshape(-1, c, 32, 4, 32, 4).permute(0, 2, 4, 1, 3, 5).reshape(-1, 1024, c, 16)
                    tm = bm.reshape(-1, c, 32, 4, 32, 4).permute(0, 2, 4, 1, 3, 5).reshape(-1, 1024, c, 16)
                else:
                    pred = torch.einsum("nld,cd->nlc", tok, q)                            # [n, 1024, C]
                    tv = vals.reshape(vals.shape[0], vals.shape[1], 1024).transpose(1, 2)
                    tm = bm.reshape(bm.shape[0], bm.shape[1], 1024).transpose(1, 2)
                m = tm & cells.reshape(cells.shape + (1,) * (tm.ndim - 2))
                e = ((pred.float() - torch.nan_to_num(tv)) ** 2)[m]
                per.setdefault(g["family"], []).append(e)
                n += e.numel()
            if per:
                l_rec, fam = self.balance({k: torch.cat(v) for k, v in per.items()}, balance, weights)
                out.update({f"recon_{k}": v for k, v in fam.items()})
            else:
                l_rec = z.sum() * 0
            out.update(loss=l_rec, recon=l_rec.detach(), n_targets=n)
        out["z"], out["gmask"] = z, gmask
        return out

    def pace_residual_loss(self, batch, z):
        errs = []
        for g in batch["groups"]:
            if not g.get("is_pace") or "pace_resid" not in g:
                continue
            sel = g["ctx_cells"].any(1)
            if not sel.any():
                continue
            tok = z[g["seq"][sel], g["slot"][sel]]                                       # [n, 1024, D]
            cells = g["ctx_cells"][sel]
            r = g["pace_resid"][sel].float().transpose(1, 2)                             # [n, 1024, P]
            rm = g["pace_resid_mask"][sel].transpose(1, 2) & cells[..., None]
            if not rm.any():
                continue
            pred = self.resid(tok[cells])                                                # [k, P]
            errs.append(((pred.float() - torch.nan_to_num(r[cells])) ** 2)[rm[cells]])
        return torch.cat(errs).mean() if errs else None

    @torch.no_grad()
    def embed_e2(self, batch, ref_wl: torch.Tensor | None = None, view: int = 0) -> torch.Tensor:
        """E2 (arms B, C): the predictor queried at dt = 0, view type `view`, with MODIS-Aqua's
        ocean-band signature (default), for every cell, even on days without an observation. The
        query is trained: synthetic MODIS-band targets from hidden PACE/OLCI observations plus native
        hidden MODIS observations (oefm/model/data.py). Returns [B, 1024, D]."""
        if ref_wl is None:
            from .data import MODIS_REF_WL

            ref_wl = torch.tensor(MODIS_REF_WL, dtype=torch.float32)
        z, gmask, dts = self.encode_context(batch)
        B, T = batch["B"], batch["T"]
        p = torch.zeros(B, T + 1, 1024, self.dp, device=z.device, dtype=z.dtype)
        valid = torch.zeros(B, T + 1, 1024, dtype=torch.bool, device=z.device)
        p[:, :T][gmask] = self.p_in(z[gmask])
        valid[:, :T] = gmask
        sig = self.m_band(fourier(ref_wl.to(z.device), self.n_fourier, 2.0, 2000.0).mean(0))
        q = (self.mask_token + self.m_view.weight[view] + sig)[None, :] + self.m_dt(self._dt(torch.zeros(1, device=z.device)))
        p[:, T] = (q + self.m_pos).to(p.dtype)
        valid[:, T] = True
        h = p.permute(0, 2, 1, 3).reshape(B * 1024, T + 1, self.dp)
        h = _rows(h, valid.permute(0, 2, 1).reshape(B * 1024, T + 1), self.p_temporal)
        h = h.reshape(B, 1024, T + 1, self.dp).permute(0, 2, 1, 3).reshape(B * (T + 1), 1024, self.dp)
        h = _rows(h, valid.reshape(B * (T + 1), 1024), self.p_spatial)
        return self.p_out(h.reshape(B, T + 1, 1024, self.dp)[:, T])

    @torch.no_grad()
    def embed_e1(self, batch, z=None, gmask=None) -> tuple[torch.Tensor, torch.Tensor]:
        """E1: per cell, the temporal-stage tokens averaged over observations, weighted by each
        observation's valid fraction. Returns [B, 1024, D] and a cell mask [B, 1024]."""
        if z is None:
            z, gmask, _ = self.encode_context(batch)
        w = gmask.float() * gmask.float().mean(-1, keepdim=True)                         # valid fraction per obs
        num = (z.float() * w[..., None]).sum(1)
        den = w.sum(1)
        return num / den.clamp(min=1e-6)[..., None], den > 0


@torch.no_grad()
def embed_e3(e1_mean: torch.Tensor, batch_z: torch.Tensor, gmask: torch.Tensor) -> torch.Tensor:
    """E3: E1 mean concatenated with the valid-fraction-weighted standard deviation over the
    window: [B, 1024, 2D] (1,024 dimensions for D = 512)."""
    w = gmask.float() * gmask.float().mean(-1, keepdim=True)
    den = w.sum(1).clamp(min=1e-6)
    var = (w[..., None] * (batch_z.float() - e1_mean[:, None]) ** 2).sum(1) / den[..., None]
    return torch.cat([e1_mean, var.clamp(min=0).sqrt()], dim=-1)


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of normalised singular values) of a [N, D] sample (RankMe)."""
    x = x.float() - x.float().mean(0)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum()))
