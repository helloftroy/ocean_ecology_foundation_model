# Ocean Foundation Model: Methods

Oct 27, 2026 · @helen · updated 2026-11-12 to the settled pilot decisions and the v1 implementation (`oefm/model/`, `config/train_stage1.yaml`); the running decision log is `cluster_instructions.md`.

## Overview

An open, multisensor ocean foundation model whose embeddings ocean scientists, coastal managers, and eDNA, fish and coral researchers can use directly with few labels.&#32;

**Approach.** A joint-embedding predictive architecture (JEPA). Each satellite product or physical field is a *view* of the same ocean state. Whole views, patches and time steps are hidden, and the model predicts their embeddings from what remains, adapting Mol-JEPA's modality masking, with LeJEPA's SIGReg regularizer against collapse. A small auxiliary decoder keeps PACE-only spectral detail in the embeddings.

| Decision | Choice | Why |
| --- | --- | --- |
| Objective | JEPA latent view prediction + SIGReg, plus a PACE residual decoder | Embeddings are the product; latent prediction ignores unpredictable noise; one loss across very different views; the loss tracks downstream quality |
| Views | Rrs per sensor (MODIS, VIIRS, OLCI, PACE); fluorescence; Rayleigh-corrected red/NIR; physical drivers (stage-2 option) | Covers phytoplankton through floating seaweed, plus the drivers of change |
| Physics policy | Observation-based preferred; physical reanalysis only where no observation exists, tagged as modeled; no modeled biology | Nutrient supply learned through physical proxies; nutrients used as a test |
| Location and dates | Not in the backbone; relative time only | Learn ocean processes, not places; avoid leakage downstream |
| Resolution | 1.2 km token grid; 300 m OLCI precision path | Coastal users |
| Outputs | `ocean_embedding_1200m`, `ocean_embedding_300m`; causal and retrospective modes | One trained model |
| Compute | 4× A100 (80 GB) on Ptolemy, 48-hour jobs with automatic resubmission | Existing resources; no fixed total wall-clock budget |
| First step | Two-stage Gulf pilot: objectives, then physics and nutrient views | Decide the objective cheaply before the global run |

## Views

Each view is one product of one instrument, or one physical field, over the same tile and time window. Views differ in bands, resolution and timing, and any subset can be present.

| Group | View | Source | Native resolution | Years | Notes |
| --- | --- | --- | --- | --- | --- |
| Ocean colour | MODIS-Aqua Rrs | OB.DAAC L2 OC | 1 km | 2002–2022 core | Later years tagged degraded; mission ends \~Sept 2027 |
| Ocean colour | VIIRS Rrs | OB.DAAC L2 OC (SNPP, NOAA-20, NOAA-21) | 750 m | 2012–present | SNPP delivery ends Nov 2026 |
| Ocean colour | OLCI Rrs | OB.DAAC L2 OC, full resolution (EFR) | 300 m | 2016–present | Feeds the 300 m path |
| Ocean colour | PACE OCI Rrs | OB.DAAC L2 AOP | \~1.2 km | 2024–present | 172–184 Rrs bands by product version |
| Physiology | Fluorescence line height | Same L2 files where provided | As sensor | As sensor | Not VIIRS; confirm per sensor |
| Surface | Rayleigh-corrected red/NIR reflectance | PACE: L2 SFREFL, variable rhos (provisional product). MODIS, VIIRS: l2gen from Level-1, selected scenes only. OLCI: distributed product to verify | As sensor | As sensor | Floating seaweed; keeps aerosol signal, removes Rayleigh and gas; own flag rules; optional view slot so the model never discards this signal |
| Physics | SST | GHRSST L4 MUR (PO.DAAC) | \~1 km daily | 2002–present | Observation-based analysis |
| Physics | Sea level and geostrophic currents | Copernicus Marine L4 altimetry | \~0.25°, product-dependent | 1993–present | Observation-based |
| Physics | Winds | CCMP (satellite winds blended with reanalysis) or ERA5 | \~0.25° | — | ERA5 tagged modeled |
| Physics | Precipitation | IMERG (GES DISC) | 0.1° | 2000–present | Runoff proxy |
| Static | Bathymetry | GEBCO grid | \~450 m | — | Needed for optically shallow water |
| Optional | Salinity | SMAP, SMOS | \~40–70 km | 2010–present | River plumes |
| Optional | Mixed-layer depth | Physical reanalysis | Coarse | — | Tagged modeled |

**Excluded:** modeled nutrients and chlorophyll as inputs (PISCES phosphate, nitrate and silicate are a target-only view in stage 2); USF SaWS Sargassum products (derived from the same MODIS/VIIRS bands plus HYCOM currents, so a benchmark target only; **blocked in the pilot** until numerical SaWS grids are obtained); coordinates and absolute dates as inputs. River discharge is well observed only in the US, so it belongs in regional downstream models.

**What the views can see, by trophic level:**

| Target | Seen through | Role |
| --- | --- | --- |
| Phytoplankton biomass and groups | Pigment absorption in Rrs; finer group detail from PACE | Input views |
| Phytoplankton physiology | Chlorophyll fluorescence near 680 nm | Input view |
| Dissolved organic matter, river plumes | UV–blue absorption, especially PACE UV | Input views |
| Bacteria and other non-pigmented microbes | Not visible; inferred through association with phytoplankton, organic matter and physics | Downstream eDNA heads |
| Floating seaweed | Red/NIR reflectance of surface mats | Rayleigh-corrected view |
| Fish, reefs | Not visible; respond to the above plus physics over weeks to months | Downstream heads, causal windows |
| Nutrients | Not visible; supply implied by mixing, upwelling, storm wakes, rain | Physics proxies as inputs; measured nutrients as a test |

## Architecture

&#91;embedded content: JEPA training · 3 view groups, 1 auxiliary decoder\]

Visible views pass through the context encoder and predictor; hidden views pass through the same encoders to give targets. Physics fits as additional views with a longer lookback, so the model can learn driver-to-response links such as a storm's cold wake followed by a bloom.

### Per-view encoders

**Optical views:** a Clay/DOFA-style dynamic spectral embedding (reuse the code, train from scratch) with four ocean-colour changes:

1. Encode bandwidth as well as centre wavelength (Copernicus-FM-style Fourier encodings). *v1:* bandwidth enters through the SRF integration (item 2); the L2 files carry no band widths, so SRFs are Gaussian at each sensor's nominal FWHM until measured response tables are added.
2. Generate weights through spectral response functions: a continuous weight function w(λ) integrated against each band's SRF. This pushes a heritage band and the PACE bands it spans toward similar contributions; it does not make them equal.
3. Normalize for band count by dividing by the number of valid bands. No LayerNorm on the token, so overall amplitude is kept (see Tokenizer, "Projection").
4. Signed reflectance transform (Preprocessing).

During training, PACE is randomly subsampled to 20–100% of its bands (spectral dropout).

**Patch sizes:** OLCI 4 × 4 at 300 m; MODIS, VIIRS and PACE 1 × 1 at 1.2 km, so every optical view lands on the same 1.2 km token grid. Per-view patch sizes need code changes: separate patch paths sharing one weight generator, or kernel resizing as in Copernicus-FM.

**Physical views:** a variable-aware embedding (each field tagged by its variable, like Copernicus-FM's variable hypernetwork) on a coarse token grid aligned to the tile. Bathymetry is a static view.

### Context encoder

- **Spatial:** a ViT per observation; benchmark windowed or hierarchical attention against global attention.
- **Temporal:** irregular attention over observations with Fourier-encoded Δt relative to the target time. Optics window: ±15 days (retrospective) or the preceding 30 days (causal). Physics stream: daily, with a longer lookback (*workshop:* 60–120 days).
- **No** coordinates, absolute dates or site identifiers. Sensor identity is implicit in wavelength and variable metadata.

### Predictor and targets

- **Predictor:** a transformer that receives context tokens plus mask tokens carrying the hidden view's identity (its wavelength set or variable), relative time and patch position, and outputs predicted embeddings.
- **Targets:** hidden views encoded by the same encoders; no EMA teacher network (LeJEPA).
- **Regularizer:** SIGReg on the target embeddings. In Mol-JEPA, applying it to predicted embeddings collapsed training.
- **No projection head** (Mol-JEPA: removing it improved downstream results).

### Outputs

- `ocean_embedding_1200m`: pooled context-encoder output per 1.2 km cell for a requested window.
- `ocean_embedding_300m`: from the OLCI precision path, where valid native-resolution OLCI observations exist.
- **Modes:** retrospective (window centred on the date) and causal (past observations only, for near-real-time use).
- **Size:** 512 dimensions internally (Mol-JEPA saw gains saturate at 512); released size *workshop*.

## Training objective and masking

**Loss** = (1 − λ) · latent prediction + λ · SIGReg, plus μ · PACE residual in arm C (LeJEPA's weighting; exact definitions in the implementation specification).

- **Latent prediction:** mean squared error between predicted and target embeddings for every hidden view, patch or time step.
- **SIGReg** on target embeddings: an Epps–Pulley test over random 1D projections. Starting values from LeJEPA: 1,024 projections, 17 integration points, integration range \[−5, 5\], λ = 0.05; sweep λ in the pilot.
- **PACE residual guard (μ, small, *workshop*):**
  1. For each PACE spectrum p, compute the values heritage bands would record (SRF-convolved MODIS, VIIRS and OLCI bands) and the best linear reconstruction of p from them, using one fixed regression fitted on PACE data.
  2. The residual r = p minus that reconstruction is the part of the spectrum heritage bands cannot explain.
  3. A small decoder predicts r from the embedding whenever PACE is among the visible views.

  This rewards keeping PACE-only spectral detail instead of only aligning the bands the sensors share.
- **Monitor, not a loss:** a full PACE-spectrum decoder trained on frozen embeddings, with per-wavelength error on the reserved 2026 PACE observations reported at each checkpoint.

**Masking recipe** (rates *workshop*; start with roughly 30–50% of tokens hidden, near Mol-JEPA's best of 0.3):

| Mask | What is hidden | What it teaches |
| --- | --- | --- |
| Whole view | One or more sensor or physics views across the window | Cross-sensor and driver-to-response links |
| PACE drop | PACE, with heritage sensors visible (2024-on samples) | PACE-informed embeddings from heritage sensors |
| Patch | Spatial patches within a view | Spatial structure, fronts, eddies |
| Time step | Whole observation dates | Temporal dynamics and gap filling |
| Future | Observations after a cut-off date | Lagged responses (causal mode) |
| Physics dropout | The whole physics stream | Robustness when only optics are available |

**Rules:** at least one optical observation stays visible, except in a small share of samples where optics are predicted from physics alone (*workshop*: 5–10%). The PACE residual loss applies only when PACE is visible.

## Implementation specification (v1)

First version of everything, for the pilot. Where this differs from Mol-JEPA, it is because our views are spatial, temporal and spectrally variable rather than one vector per molecule. Alternatives are in the next section. Notation: B = sequences per batch, T ≤ 16 optical observations per sequence, N = 32 × 32 = 1,024 cells per tile at 1.2 km, D = 512.

### Defaults

| Component | v1 value |
| --- | --- |
| Token width D; heads; MLP ratio | 512; 8; 4×; pre-LayerNorm; drop-path 0.1 |
| Spatial blocks (per observation) | 6, global attention over ≤ 1,024 cells |
| Temporal blocks (per cell, across observations) | 4 |
| Predictor | 4 blocks at width 384 (2 temporal, 2 spatial) |
| Spectral tokenizer | SRF-integrated weight function w(λ): MLP on 32 Fourier features of λ, 1 nm grid 340–2300 nm, conditioned on product (Rrs or rhos) |
| Patch sizes | OLCI 4 × 4 at 300 m; MODIS, VIIRS, PACE 1 × 1 at 1.2 km |
| Max observations per sequence | 16 |
| SIGReg | 1,024 random projections resampled each step; Epps–Pulley, 17 points on \[−5, 5\]; ≤ 8,192 tokens per step |
| Loss weighting | (1 − λ) · prediction + λ · SIGReg, λ = 0.05; both balanced over view groups (Modality balancing) |
| PACE residual weight μ (arm C) | 0.1 |
| Optimizer | AdamW, β = (0.9, 0.95), peak learning rate 5e-4, weight decay 0.05, gradient clip 1.0 |
| Schedule | Linear warmup 5% of steps, cosine decay to 1e-6 |
| Batch | 128 sequences effective (per-GPU batch set by benchmark, plus accumulation) |
| Pilot length | Same number of steps for every arm (*workshop:* \~30k, set from the benchmark) |
| Precision | bf16; checkpoints every 2 hours; automatic resubmission |

### Input sample schema

One training example = one tile, one target time t0, and a set of observations. t0 is used only to compute relative times; it is never an input.

- **Optical observation** (one per sensor overpass and product):
  - `view_type`: `rrs` or `surface` (PACE SFREFL); `sensor` kept for bookkeeping and sampling only, never fed to the model.
  - `values`: float16 `[C, H, W]`, transformed and normalized (H = W = 128 for OLCI, 32 otherwise).
  - `band_mask`: bool `[C, H, W]`.
  - `wavelength_nm`, `bandwidth_nm`: `[C]`; `srf_index`: `[C]` into a table of SRFs sampled at 1 nm from 340 to 2300 nm (PACE SFREFL rhos includes SWIR bands out to \~2258 nm).
  - `quality`: uint8 `[9, H, W]`, the soft-flag channels.
  - `dt_days`: observation time minus t0.
- **Physics** (stage 2): float `[V, L, 4, 4]` with V variables, L = 90 daily steps; `var_id`: `[V]`; mask.
- **Nutrient targets** (stage 2, arm E): PISCES phosphate, nitrate and silicate at t0, one tile-level value each (0.25° model cells; log + z-score from the training partition); never an input.
- **Static**: bathymetry `[1, 32, 32]`.
- **Storage (implemented):** one zipped Zarr store per satellite overpass holding every pilot tile it covers (one tile per chunk, which keeps the GPFS file count near one per granule); a Parquet index lists (tile, observation, product, family, time, valid fraction, zenith). The loader samples t0, queries the index for the window, and assembles the sequence; values are transformed on the fly with frozen training statistics.

### Tokenizer

- **Spectral weights:** an MLP maps Fourier features of each 1 nm wavelength (340–2300 nm), together with a product embedding (Rrs or rhos), to a weight vector, so the same wavelength gets different kernels for water-leaving and surface reflectance; each band's kernel is that function integrated against the band's SRF. A heritage band and the PACE bands under it therefore get consistent kernels. There are two output heads on a shared trunk: D values (1 × 1 patch) and D × 16 values (4 × 4 OLCI patch).
- **Projection:** token = sum over valid bands of kernel × value, divided by the number of valid bands. This handles band count and partial spectra directly. No LayerNorm on the token: it would divide out overall amplitude, making a spectrum and a scaled copy of it identical. Normalization happens only inside the pre-LayerNorm transformer branches, so the residual stream keeps magnitude. Two amplitude features are also added explicitly through a small MLP: the mean transformed value over valid bands, and asinh of the mean raw reflectance. Unit test: synthetic spectra with the same shape scaled by 0.5, 1 and 2 must give distinguishable tokens, and a linear probe on tokens must recover log amplitude (R² ≥ 0.95) at initialization and after training. No transformer runs over bands, which avoids attention cost growing with PACE's band count.
- **Added embeddings:** quality channels (linear 9 → D); view-type embedding (`rrs`, `surface`, `physics`, `static`, `nutrient`); 2D sine-cosine position of the cell within the tile (local, not geographic). *v1 change:* Fourier features of `dt_days` are added at the temporal stage, not in the tokenizer, so a target token (an observation encoded alone) does not depend on the chosen t0.
- *Implemented amplitude features:* the mean transformed value over valid bands and asinh(mean raw Rrs / 10⁻³ sr⁻¹). The unit test passes at initialization (log-amplitude probe R² ≥ 0.95; 0.5× and 2× copies distinguishable; Rrs vs rhos at the same wavelengths distinguishable); the after-training check runs at checkpoints.
- **Missing cells** (no valid band) are excluded from attention.
- **Spectral dropout:** in context only, keep a random 20–100% of PACE bands.
- **Physics** (stage 2): per variable and coarse cell, a small temporal MLP over the 90 daily values gives one token, plus a learned variable embedding: V × 16 tokens.

### Context encoder

1. **Spatial stage:** each observation is encoded alone by the 6 spatial blocks.
2. **Temporal stage:** for each cell, the 4 temporal blocks attend across that cell's tokens from every visible observation (all sensors and products together), using the Δt embedding.
3. **Stage 2 only:** 2 fusion blocks in which cell tokens cross-attend to the physics and static tokens.

The output is one token per visible (observation, cell).

### Targets

- A target is the **spatial-stage output** of a hidden observation, encoded alone with all its valid cells visible. There is no temporal context, so a target can't leak its neighbours' information.
- Gradients flow through targets; there is no EMA teacher and no stop-gradient (LeJEPA).
- **SIGReg** is applied to the target tokens. Mol-JEPA's training collapsed when SIGReg was on predictions, and was stable on targets.
- **Nutrient targets** (arm E): a 2-layer MLP encodes each coarse cell's phosphate and nitrate into D, as a view the predictor must predict and the context never contains.
- **Why not Mol-JEPA exactly:** its targets were whole-molecule vectors from pretrained models. Ours are per-cell tokens from our own encoder, so SIGReg must keep them spread out; this is why it applies to targets and not to predictions.

### Predictor

- **Inputs:** the context tokens, projected to 384, plus one mask token per hidden (observation, cell).
- **Each mask token carries:** view-type embedding; a band-signature embedding, the mean of the Fourier features of the hidden observation's bands (telling the predictor which sensor's bands to predict); Δt embedding; cell position.
- **Blocks:** 2 temporal (per cell), then 2 spatial (per observation), then projection back to D.

### Masking sampler

Each sequence draws one mode:

| Mode | Share | Hidden |
| --- | --- | --- |
| Whole view | 40% | All observations of one view group, drawn uniformly over the groups present (Modality balancing); PACE is one of the groups |
| Time step | 30% | 25–50% of observations |
| Spatial blocks | 20% | 4–6 rectangles per observation covering \~40% of cells |
| Future | 10% | All observations with Δt > 0 |

At least one optical observation always stays visible. Separately, 10% of sequences carry a synthetic reference-band target for E2 (Released embeddings). Rectangles rather than random cells, because ocean fields are spatially smooth; random cells would be trivial to interpolate.

### Losses

- **Prediction:** mean over hidden valid (observation, cell) of the squared error between predicted and target tokens, divided by D, computed within each view group (Modality balancing, below).
- **SIGReg:** sample up to 8,192 target tokens per step across all GPUs and project onto 1,024 random directions shared across GPUs by seeding on the global step. Then apply the Epps–Pulley statistic with 17 integration points on \[−5, 5\], per view group (Modality balancing). *Implementation:* the official LeJEPA library (galilai-group/lejepa: `epps_pulley.py`, `slicing.py`): 17 points on \[0, 5\] with doubled trapezoid weights (the same integral as \[−5, 5\]), window exp(−t²/2), statistic × N, slice seed from a step counter synchronized across GPUs.
- **Total (arms B, C):** (1 − λ) · prediction + λ · SIGReg; arm C adds μ · PACE residual.
- **PACE residual (arm C):**
  1. Fit a ridge regression W offline, mapping SRF-convolved heritage-band values (MODIS, VIIRS, OLCI) to the full PACE spectrum. Use ≥ 100k PACE spectra from 2024–2025 only (never 2026) and store W. *Implemented* in `oefm/normstats.py` from the training split (2024-03 – 2025-12, training tiles only); the penalty is chosen on a 10% hold-out.
  2. Residual r = PACE spectrum − W · heritage values, in transformed units.
  3. Decoder: MLP (D → 512 → number of PACE bands) on the temporal-stage token of a visible PACE observation's cell.
  4. Loss: mean squared error over valid bands.

**Modality balancing.** Hidden tokens are split unevenly across views (three VIIRS platforms, many more Rrs than rhos observations, nutrients only in arm E), so plain averaging would let the most frequent view dominate.

- **View groups:** MODIS Rrs, VIIRS Rrs, OLCI Rrs, PACE Rrs, PACE rhos, nutrients (arm E). Sensor is used here for bookkeeping only; it is still never a model input.
- **Prediction loss:** mean squared error within each view group, then the group means averaged with equal weight: L\_pred = (1/G) Σ\_g mean\_g.
- **SIGReg:** computed per view group on that group's target tokens (Mol-JEPA applies it per modality), then averaged with equal weight. A group with fewer than 256 tokens in a step is pooled with the others for that step.
- **Sampling:** in whole-view masking, the hidden view is drawn uniformly over the view groups present, not over observations.
- **Logging:** per-group prediction and SIGReg losses every step.
- **TODO:** compare equal group weights with inverse-frequency and temperature-scaled weights; check that no group's loss stalls.
- *Implementation:* `loss.modality_weights` (default 1 per group), `loss.balance` (`modality` or `token` for the pooled mean), `loss.sigreg_mode` (`group` or `pooled`), `loss.sigreg_min_group_tokens` (256). Across GPUs, different ranks can hold different groups, so the per-group SIGReg walks one fixed group list on every rank and all-reduces the sums of cos/sin and the token counts. The characteristic function is then the exact global mean (equal to the reference implementation when every GPU has the same number of tokens), and every rank makes the same collective calls.

### Arm A (original plan)

- Same tokenizer, context encoder and masking sampler. No SIGReg, no predictor.
- **Decoder:** 2 blocks per cell over query tokens for the hidden observations (the same mask-token construction as the predictor).
- **Output:** the dot product of the decoder token with a band query MLP(Fourier(λ)) gives the transformed value per band. For OLCI, 16 outputs give the 4 × 4 pixels.
- **Loss:** mean squared error on valid bands of hidden observations.

### Released embeddings: three candidates tested in the pilot

All three are inference-only, so the pilot evaluates every one on the same trained model.

- **E1 (default):** run the encoder on every observation in the window (retrospective ±15 days; causal −30 to 0 days) and average temporal-stage tokens per cell, weighted by valid fraction.
- **E2:** a predictor query at Δt = 0 with view type `rrs` and a fixed reference band signature. This produces an embedding even on days without an observation, similar in spirit to AlphaEarth's time-conditioned embeddings. Trained explicitly: the reference band set is MODIS-Aqua's ocean bands. In 10% of sequences the sampler adds a synthetic reference target, a hidden PACE or OLCI observation SRF-convolved onto those bands and encoded alone at its own Δt; native hidden MODIS observations also count. Without this, the reference query would be one the predictor never learned to answer.
- **E3:** E1 mean concatenated with E1 standard deviation over the window, capturing variability.

E1 and E2 are 512 dimensions; E3 is 1,024 (mean and standard deviation concatenated). All are float16; compression is a fork.

### The 300 m path: after stage 1

Not trained in stage 1, so the objective comparison stays clean. After the objective is chosen:

- A pixel encoder embeds each OLCI 300 m pixel with the 1 × 1 spectral head, at 128 dimensions.
- **Training task:** hide half of the 16 sub-pixels in a cell. Predict their pixel embeddings from the cell's temporal-stage token plus the visible sub-pixels, with SIGReg on the pixel embeddings.
- **Release:** `ocean_embedding_300m` = MLP(\[cell token projected to 128, pixel embedding\]).

### Monitoring

Log prediction loss, SIGReg, and the effective rank of released embeddings. Every few thousand steps, fit an online linear probe on SFER fluorometric chlorophyll. At each checkpoint, fit the frozen PACE-spectrum decoder.

### One training step (arms B and C)

```python
seq = loader.next()                          # observations, physics, nutrients
ctx, hid = sample_mask(seq)                  # one masking mode; >=1 optical obs visible
hid = add_reference_target(hid, p=0.1)       # synthetic MODIS-band target for E2
ctx = pace_spectral_dropout(ctx)
z_ctx = temporal(spatial(tokenize(ctx)))     # context encoder
z_tgt = spatial(tokenize(hid))               # targets: each hidden obs encoded alone
z_hat = predictor(z_ctx, mask_tokens(hid))   # view type, band signature, dt, position
g = hid.view_group                           # MODIS-Rrs, PACE-rhos, nutrients, ...
l_pred = mean_over_groups(mse(z_hat, z_tgt), g)
l_sig = mean_over_groups(sigreg(z_tgt, g, n_proj=1024, step=global_step))
loss = (1 - lam) * l_pred + lam * l_sig
if arm == "C" and pace_visible(ctx):
    loss += mu * pace_residual_loss(z_ctx, ctx)
loss.backward(); clip_grad(1.0); opt.step(); sched.step()
```

## Forks: alternatives to try

Ranked by predicted usefulness. Each has a trigger: the pilot result that would make it worth trying. If a later conversation changes direction, record here which option was selected.

| # | Fork | Try when | Change from v1 |
| --- | --- | --- | --- |
| 1 | Released embedding E2 or E3 instead of E1 | Always; tested in the pilot | Inference only; no retraining |
| 2 | Hybrid JEPA + light reconstruction | B or C trails A on SFER chlorophyll or PACE decoder error | Add arm A's band decoder at weight \~0.1 beside the JEPA loss |
| 3 | SIGReg also on temporal-stage tokens | Low effective rank of released embeddings, or weak probes despite low prediction loss | Apply SIGReg to subsampled context tokens too |
| 4 | Targets with temporal context | Prediction loss very low early (task too easy) with weak probes | Targets = spatial + temporal stages on all observations, with stop-gradient on targets |
| 5 | Masking mix | Some tasks lag (e.g. Sargassum next-week, cross-sensor) | Shift shares toward whole-view (Mol-JEPA style) or future masking |
| 6 | Windowed spatial attention (8 × 8 windows with shifts) | Memory or throughput fails the benchmark | Replace global spatial attention |
| 7 | Interleaved spatial and temporal blocks | Weak time-step and future tasks | Alternate spatial and temporal blocks instead of spatial-then-temporal |
| 8 | Wider predictor (6 blocks, width 512) | Prediction loss plateaus high | Increase predictor capacity |
| 9 | DOFA-style transformer over bands | PACE decoder error high even in arm C | Generator attends over the band set instead of the SRF-integrated MLP |
| 10 | EMA teacher (I-JEPA style) | Training unstable under SIGReg | Targets from an EMA copy of the encoder, with stop-gradient |
| 11 | Physics as daily tokens | Stage 2 shows physics helps but plateaus | Keep daily resolution instead of one pooled token per variable |
| 12 | Variable names embedded by a language model (Copernicus-FM style) | Adding new physics variables later | Replace the learned variable table |
| 13 | Compressed release (PCA to 64 dimensions, int8) | Storage of global layers is too large | Post-hoc compression; check probe loss |
| 14 | Projection head | Collapse despite SIGReg | Add a 2-layer projector before SIGReg (Mol-JEPA found it hurt, so last resort) |

## Gulf pilot: choosing the objective

The pilot tests the full pipeline, decides whether JEPA beats the original reconstruction plan, and then tests physics and nutrients.

- **Region:** three pilot boxes (SW Florida shelf, Florida Keys, offshore eastern Gulf) in a domain of 22.8–29.3°N, 92–78.8°W; 272 tiles.
- **Years:** 2018–2025 for training (2024–2025 has all four sensors), including all unlabeled 2024–2025 PACE; the 2024 hold-out applies to downstream biological labels only. 2026 PACE observations are reserved as the PACE spectral test set: never used in training or to fit the PACE residual regression. Partitions (approved 2026-11-12): January–February 2026 validation; March 2026 a separation period used by nothing; **April–October 2026 the PACE spectral-retention test** (all sensors in those months are kept out of pretraining). An independent geographically blocked test holds out 20% of 4 × 4-tile blocks (with a 1-tile buffer) at all times. A sequence only ever contains observations of its own partition.
- **Equal budget per arm:** same tokenizer, encoder size, data and number of steps.

**Stage 1: objective (optics only)**

| Arm | Objective |
| --- | --- |
| A | Masked reconstruction of actual values (the original plan; no SIGReg) |
| B | JEPA latent view prediction + SIGReg |
| C | B + PACE residual guard |

**Stage 2: on the stage-1 winner**

| Arm | Adds |
| --- | --- |
| D | Physical drivers as an input view |
| E | PISCES phosphate, nitrate and silicate as a target-only view (predicted, never an input) |
| F | D + E |

**Why stage 2 is worth testing:** Helen's R24–R27 tests (Granite PCs + temperature, 3,921 KOs, one cruise held out) found that adding ocean-model fields made 414 genes (11%) usefully predicted. PISCES phosphate and nitrate made 70 and 67 genes useful, mostly nutrient-acquisition genes (pho regulon, urea uptake, urease), even though PISCES surface phosphate is about 8× too low. Physical fields helped smaller, less consistent sets; the 30-day maximum mixed layer was the most consistent (164 genes in ≥ 75% of cruises). Testing in the denser Gulf, and inside pretraining rather than as extra regression inputs, may behave differently.

**Metrics (both stages):**

- Main benchmark: Florida 16S/18S community composition (SEUS MBON; major taxa and community components), with cruise-grouped holdouts and cross-cruise consistency. NMF and all target preprocessing are refitted on training labels only; 2024 labels held out; shelf and Keys only. Measured SFER nutrients at the eDNA samples are an independent target.
- SFER fluorometric chlorophyll and nutrients (labels from 2025 held out; the primary PACE-on/PACE-off comparison); reef fish (CRCP, and video MaxN at month precision; 2024 held out). HPLC is too sparse in the pilot area.
- Gulf Sargassum (SaWS): same-week and next-week tasks. Blocked until numerical SaWS grids are obtained; image products alone are not enough.
- PACE full-spectrum decoder error by wavelength, from frozen embeddings.
- Residual gain from adding PISCES phosphate on top of the embeddings for the Florida communities, and for pho/pst genes where gene data exist; it should shrink if arm E works.
- Correlation between training loss and probe scores; throughput and memory.

**Decision rules (*workshop* margins):**

- Stage 1: choose B or C if probes match or beat A and PACE decoder error stays within the margin of A; choose C over B if B loses PACE detail.
- If JEPA wins clearly, tune it on the Gulf before the global run: λ, masking mix, predictor depth, and the released-embedding method (E1–E3 in the spec).
- Stage 2: keep D or E only if gains beat shuffled-variable controls and hold in ≥ 75% of held-out cruises.

## Data preparation: start now

Downloading raw Level-2 ocean colour and the physics fields needs no further design decisions. Processing (regridding, chip building) does; those decisions are listed at the end of this section, with recommendations.

### Set-up

- **Accounts:** NASA Earthdata (OB.DAAC, PO.DAAC, GES DISC); Copernicus Marine (altimetry); Copernicus Climate Data Store only if ERA5 is used.
- **Tools:** `earthaccess` (Python) for NASA search and download; the Copernicus Marine toolbox; OCSSW/SeaDAS `l2gen` only if Level-1 processing is needed.
- **Layout on Ptolemy scratch:** `raw/<sensor>/<product>/<version>/`, never modified after download. Code, manifests and normalization statistics live in storage that is not purged.
- **Manifest per download:** product short name, processing version, granule list with checksums, download date.

### Download order

1. **Pilot ocean colour, Gulf, 2018–2026:** MODIS-Aqua L2 OC; VIIRS L2 OC (NOAA-20, NOAA-21; SNPP until Oct 2026); OLCI Sentinel-3A/3B L2 OC full resolution (EFR); PACE OCI L2 AOP; PACE OCI L2 SFREFL (surface reflectance, rhos; record its provisional version).
2. **Pilot physics, same window:** GHRSST MUR L4 SST; Copernicus Marine L4 altimetry (sea level and geostrophic currents); CCMP winds, with ERA5 as fallback; IMERG precipitation; GEBCO bathymetry. These are coarse and small.
3. **Pilot evaluation data:** Florida 16S/18S communities; SFER fluorometric chlorophyll and nutrients; reef fish (CRCP, video MaxN) — in hand. HPLC (SeaBASS, BCO-DMO) too sparse so far; BGC-Argo nitrate in the Gulf not yet checked; SaWS Sargassum numerical grids (blocked until obtained). See `docs/evaluation_inventory.md`.
4. **Global:** only after the pilot, as a sampled set of tiles and dates, never the full archive.

### Checks during download

- Open each file and confirm Rrs, `l2_flags` and the wavelength variables are present.
- Log band lists per file (PACE `wavelength_3d`; it differs by product version).
- Record the processing version from file metadata, not only from the file name.
- Spot-check OLCI full-resolution coverage over the Gulf (Open checks).

### Decisions needed before processing (not before downloading)

| Decision | Blocks | Recommendation |
| --- | --- | --- |
| Floating-seaweed view in v1? | Historical MODIS/VIIRS Level-1 processing | Pilot uses PACE L2 SFREFL (no Level-1 work); test against SaWS; process selected MODIS/VIIRS scenes with l2gen only if it works. Keep the view slot in the architecture either way |
| PACE processing version | Which PACE files to download | **Decided:** AOP 3.2, SFREFL 3.1 (tracked separately) |
| Projection and tile size | Regridding | **Decided:** Lambert azimuthal equal-area (pilot: one projection centred on the domain), 38.4 km tiles, fixed IDs |
| STRAYLIGHT flag | Regridding | **Decided:** hard mask in v1 |
| HISATZEN flag | Regridding | **Decided 2026-11-12:** not in the default hard mask; a quality channel; tag for PACE and VIIRS, hard for MODIS (full QC run) |
| Physics lookback | Chip building | 90 days to start |
| Global sampling plan | Global download | Set after the pilot |

## Preprocessing specification

### Data unit

One sample element is one overpass of one sensor over one tile, with its UTC acquisition time. Physics fields are daily values on the same tiles. *Workshop alternative:* daily per-sensor composites as a faster pilot shortcut (loses exact timing and mixes passes).

### Georeferencing

- Fixed global tile catalog; tiles are 38.4 × 38.4 km on a per-tile Lambert azimuthal equal-area projection (*workshop:* UTM or EASE-Grid 2.0).
- The 1.2 km grid (32 × 32 cells per tile) is exactly nested by the 300 m grid (128 × 128): one 1.2 km cell = 4 × 4 300 m cells.
- MODIS, VIIRS, PACE → 1.2 km: **implemented with footprint-area weighting** (each pixel's footprint sub-sampled; cell value = area-weighted mean of valid pixels), with per-cell valid weight, pixel count and OR of all flags. Pixel-centre binning left most 1.2 km cells empty at the PACE swath edge (8% vs 38% filled).
- OLCI → 300 m and → 1.2 km: the same footprint weighting at both resolutions (the OLCI EFR geolocation repeats each longitude over 2–3 columns and is de-stair-stepped first; nearest-centre gridding left stripes of empty 300 m cells).
- Physics → coarse tile grid, 4 × 4 per tile (9.6 km cells), each cell the mean of 6 × 6 bilinear sub-samples (an area average for fine fields; a larger context window is a config fork).
- Stored, not v1 inputs: solar zenith, sensor zenith and relative azimuth per cell.

### Flags

NASA's Level-3 default mask includes chlorophyll-algorithm flags ([SeaDAS l2bin](https://seadas.gsfc.nasa.gov/help-8.2.0/processors/ProcessL2bin.html), [flag table](https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/)). We train on reflectance, so:

- **Hard mask (pixel → missing):** LAND, CLDICE, SEAICE, ATMFAIL, NAVFAIL, HILT, HIGLINT, HISOLZEN (HISATZEN is not hard-masked for PACE and VIIRS; it is kept for QC diagnostics (`oefm zenith`) and as a quality channel, with sensor zenith stored per cell and per-sensor red-band angle weights. MODIS keeps it as a hard mask: its 412 nm band failed the high-angle test in the full QC run. The strict spectral-evaluation subset uses NASA's full Level-3 mask); BOWTIEDEL for VIIRS; STRAYLIGHT in v1 (it removes a large share of near-cloud and coastal data, e.g. 17% of bins and 32% of samples in one SeaWiFS test, [OB.DAAC](https://oceancolor.gsfc.nasa.gov/data/reprocessing/r2002/seawifs/masks_n_flags/mchg_gen)).
- **Not masked:** CHLFAIL, CHLWARN (chlorophyll algorithm, not reflectance).
- **Kept as a per-pixel quality channel:** HISATZEN, ATMWARN, MAXAERITER, LOWLW, ABSAER, COCCOLITH, TURBIDW, COASTZ, OPSHAL. Blooms, turbid and optically shallow water are real signal.
- **Surface-reflectance view:** its own rules. Mats can trip cloud, high-radiance and atmospheric-correction flags, so those are not applied there; use LAND, NAVFAIL and a floating-algae-aware cloud test instead (*workshop*). NASA rhos is not identical to the Rayleigh-corrected reflectance (Rrc) USF uses for its floating-algae index; validate any index against SaWS before substituting one for the other.

### Negative and missing reflectance

1. Fill values → missing, with an explicit mask; never zero-filled.
2. Negative Rrs values are kept as measured; nothing is clipped.
3. Transform `asinh(Rrs / s_λ)`, with `s_λ` a smooth function of wavelength near the typical noise level. *Workshop alternative:* `log(Rrs)` with values ≤ ε masked, logging how much data that removes.
4. Implausible values (Rrs < −k·s\_λ with k ≈ 5, or outside the file's valid range) → missing, counted in a QC log.
5. Masks are per band and per pixel; the spectral embedding receives the band mask.
6. Losses use valid values only.

### Normalization

After the transform, standardize optics with mean and standard-deviation curves fitted as smooth functions of wavelength on pooled data from all sensors, so the same wavelength gets the same normalization whatever the sensor. Rrs and rhos get separate curves; they are different quantities. Physics: per-variable standardization. All statistics come from the training split and are frozen in the config.

### Sampling

One training example = (tile, target time, optics window, physics lookback). Windows with fewer than 2 valid optical observations are skipped.

- **Regimes:** stratify tiles by Longhurst province.
- **Coast:** stratify by distance to coast (0–20, 20–100, >100 km), upweighting coastal bins.
- **US focus:** *workshop* share from the US EEZ; default 30–40%.
- **Season:** balanced by month.
- **Sensor eras:** MODIS only (2002–2011), + VIIRS (2012–2015), + OLCI (2016–2023), + PACE (2024 on); oversample the PACE era to 25–30% (*workshop*).
- **Validation holdout:** in the pilot, the approved partitions (see "Gulf pilot": 2026 reserved for validation and the PACE spectral test; geographic blocks). Province, coast-distance and EEZ stratification apply to the global run; the pilot balances months and raises the PACE era to 30% of sampled sequences.

### The 300 m path

- **Alignment:** OLCI pixels go to 300 m cells by pixel-centre nearest neighbour on the nested grid; check geolocation against shorelines before training.
- **Valid mask:** a 300 m pixel is valid when no hard-mask flag is set. `ocean_embedding_300m` is produced only where at least one valid OLCI observation falls in the window (*workshop:* at least 2), with per-pixel observation count and Δt to the nearest observation.
- **Effective resolution:** 300 m is the grid spacing, not guaranteed resolving power (off-nadir pixel growth, regridding, coastal adjacency effects, 1.2 km context). Measure it (power spectra of decoded versus held-out OLCI; edge response across fronts and coastlines) and report the number.

### Freezing versions

One OB.DAAC processing version per sensor for the whole run; manifest with granule list and checksums, flag policy, transform constants, normalization curves and code commit, stored beside the Zarr training chips.

## Compute and engineering

Training runs on Ptolemy (MSU HPC). Targets below are provisional; a short end-to-end benchmark must pass before the full run.

| Parameter | Provisional target | Notes |
| --- | --- | --- |
| Total parameters | 50–80M | Spectral and physics embeddings 2–5M; spatial ViT 512-d, 8 layers; temporal transformer 512-d, 4 layers; predictor 2–4 layers; decoders small |
| Input tile | 38.4 × 38.4 km | OLCI 128 × 128 pixels with 4 × 4 patches; other optics 32 × 32 with 1 × 1 patches; physics coarse |
| Max spatial tokens per observation | 1,024 | Visible share after masking about 50–70% |
| Max optical observations per sequence | 16 | Benchmark 8, 16, 24 |
| Physics lookback | 90 daily steps on a coarse grid | Few tokens |
| Batch | Set by benchmark | Gradient accumulation to an effective 128+ (LeJEPA works from 128) |

JEPA adds a predictor pass and encodes hidden views as targets, so expect more compute per sample than plain masked reconstruction; the benchmark measures how much.

**Rough estimate, to be replaced by the benchmark:** about 0.6–1.5 × 10²⁰ FLOPs for 20–50M sequence passes; at \~100 TFLOP/s per A100, 2–4 days of pure compute, or about 1–3 weeks of wall-clock time on 4 GPUs with overhead. Storage and CPU for preprocessing, queue wait and room for ablations are the likelier constraints. The v1 model is \~41 M parameters (arms A/B/C: 40.1 / 41.4 / 41.8 M).

**Required benchmarks** (200–500 end-to-end steps each):

- [ ] Throughput, peak memory, GPU utilization and data-loader wait for the default configuration
- [ ] Dynamic spectral embedding at the actual PACE band count (172–184) versus 5, 11 and 21 bands
- [ ] OLCI 4 × 4 versus 1 × 1 patches in the same model and batch; Clay's code will need changes for per-view patch sizes
- [ ] Global versus windowed spatial attention versus a hierarchical backbone
- [ ] Sequence length 8, 16, 24 versus memory
- [ ] Cost of the predictor and target encoding
- [ ] Data loading of Zarr chips from local scratch versus the shared filesystem
- [ ] Projected wall-clock for the full run

**Planning, not a cap:** the benchmark's projected wall-clock is used to plan job chaining; there is no fixed time budget. Shrink observations per sequence, then tile size, then model width only if memory or throughput fail the benchmark.

**Engineering defaults:** bf16 mixed precision, gradient checkpointing, multi-GPU data parallelism, checkpoints every few hours with automatic resubmission (48-hour job limit).

**Ptolemy setup:**

- `gpu-a100` partition: 4 nodes, each with 8 whole A100s and 1 TB RAM. MIG slices (40 GB on `gpu-a100-mig2`, 10 GB on the default `gpu-a100-mig7`) imply 80 GB GPUs; confirm with `nvidia-smi`. Always request `gpu-a100` for training and benchmarks; use `development` and MIG slices only for debugging.
- A full 8-GPU node would roughly halve wall-clock time, subject to per-user limits.
- Scratch: 300 TB shared, 248 TB free; treat as a rebuildable cache.
- Software: own conda environment (Python 3.11–3.12) with PyTorch CUDA wheels; check the GPU driver version.
- GitHub and NASA Earthdata are reachable from the login node.

## Evaluation and release

### Pretraining checks

- Training-loss trajectory, and its correlation with probe scores (LeJEPA reports close tracking, which enables label-free model selection).
- PACE full-spectrum decoder error by wavelength on the reserved 2026 PACE observations and on the geographic test blocks.
- Embedding agreement for the same place and time computed from different view subsets.
- Causal versus retrospective quality; with versus without physics.

### Downstream protocol

- Frozen embeddings into a ridge or logistic linear probe and a small gradient-boosted model; no backbone fine-tuning in headline numbers.
- Retrospective and causal matchups reported separately. Covariates (e.g. PISCES phosphate and nitrate for nutrient-acquisition genes) go into the same model as the embeddings, never fitted first; a gain counts only if it beats shuffled-variable controls and holds in ≥ 75% of held-out cruises.
- Baselines with the same downstream models: standard products at the same pixel (chlorophyll, Kd490, SST), raw Rrs, Granite embeddings, AlphaEarth embeddings where coverage exists.
- Repeated grouped splits (cruise, site, region, year), mean and spread reported; never random-row splits. Null results reported too.

### Suggested benchmarks

Collaborators can edit this list.

| Benchmark | Target | Depth and time relationship | Holdout | Realistic expectation |
| --- | --- | --- | --- | --- |
| HPLC pigments | Chlorophyll a, diagnostic pigments, pigment-based groups | Within the first optical depth (default ≤ 10 m); ±1 day | Cruise or campaign; years | Strongest; chlorophyll should match or beat standard algorithms |
| Florida 16S/18S (pilot); Bio-GO-SHIP (global phase) | Surface microbial community composition (major taxa, NMF components) | \~3–5 m rosette or \~7 m underway, mostly every 4–6 hours ([Larkin et al. 2021](https://link.springer.com/10.1038/s41597-021-00889-9)) | Whole cruises (stations \~26 km apart are autocorrelated); years | Moderate: good on broad gradients, weaker within a regime |
| Nutrients | Nitrate (BGC-Argo, ship), surface and mixed layer | Within the mixed layer; ±1–3 days; satellite era only | Float or cruise; years | Moderate; tests whether physics proxies carry nutrient information |
| Sargassum | SaWS density maps; field observations where available | Surface; same day to a few days | Years; regions | PACE era first; depends on the surface-reflectance view; blocked pending numerical SaWS grids |
| Fish | Presence/absence or abundance from eDNA or surveys | Indirect; causal windows over the preceding 30–90 days; depth added downstream | Spatial blocks of sites; years | Modest; show gain over standard products; pelagic likely better than demersal |
| Coral reefs | Site metrics such as coral cover or bleaching | Benthos at \~1–30 m; optically shallow water; annual surveys versus prior season or year | Reef region; years | Exploratory; SST heat stress is a strong baseline |

### Released artifacts

| Artifact | Contents |
| --- | --- |
| Model weights | Encoders, context encoder, predictor, PACE decoders; model card with data, versions and known limits |
| Preprocessing pipeline | Download, flag policy, transform, gridding, manifest; same code for training and inference |
| `ocean_embedding_1200m` | *Workshop:* weekly for the US EEZ, 2002–present; monthly globally |
| `ocean_embedding_300m` | *Workshop:* weekly for US coasts where valid OLCI exists, 2016–present, with count and Δt layers |
| Inference modes | Causal and retrospective |
| Example notebooks | Embeddings at sample points; a probe from \~50 labels; causal versus retrospective; inference on new granules |
| Benchmarks | Data or access scripts, fixed splits, evaluation code, baseline results |
| Licence and hosting | *Workshop:* Apache-2.0 for code, CC-BY for embeddings; NOAA Open Data Dissemination, Source Cooperative or Hugging Face |

## Open decisions and checks

**Decided by the pilot:**

- [ ] Objective: arm A, B or C
- [ ] λ (SIGReg weight), μ (PACE residual weight), masking rates
- [ ] Physics in or out, and lookback length
- [ ] Released embedding size

**Decisions for Helen and Andre:**

- [x] PACE processing version (AOP 3.2, SFREFL 3.1)
- [x] Projection and tile size; STRAYLIGHT policy (LAEA, 38.4 km; hard mask)
- [x] HISATZEN policy (quality channel; tag for PACE/VIIRS, hard for MODIS)
- [ ] Decision-rule margin for PACE decoder error
- [ ] Global sampling plan and US EEZ share

**Data checks:**

- [ ] OLCI full-resolution Level-2: global coverage, temporal completeness, granule dimensions, EFR versus ERR in catalog searches, valid 300 m coverage in priority coastal regions ([2 Jan 2026 archive](https://oceandata.sci.gsfc.nasa.gov/directdataaccess/Level-2/S3A-OLCI/2026/02-Jan-2026/))
- [ ] Cross-sensor radiometric consistency: compare Rrs per band across MODIS, VIIRS, OLCI and PACE matchups in overlap years; decide whether to correct biases in preprocessing or leave them to the model
- [ ] Fluorescence line height availability per sensor
- [ ] OLCI surface-reflectance product availability
- [ ] NASA rhos versus USF Rrc for floating-algae indices; obtain numerical SaWS grids (the Sargassum benchmark is blocked without them)

**Ptolemy, to ask HPC2 support:**

- [ ] Whether storing 10+ TB on scratch is allowed, and the purge policy
- [ ] Per-user GPU limits: can one user hold a full 8-GPU node?
- [ ] Where bulk downloads should run: a data-transfer node, or compute nodes with outbound access

## Sources

- [LeJEPA: provable and scalable self-supervised learning (arXiv 2511.08544)](https://arxiv.org/abs/2511.08544)
- Mol-JEPA: a multimodal joint embedding predictive architecture for molecules (NeurIPS 2026 ML4Molecules workshop; shared by Andre)
- [Sargassum Watch System — USF Optical Oceanography Lab](https://optics.marine.usf.edu/projects/saws.html)
- [OLCI water quality products — NASA Earthdata](https://www.earthdata.nasa.gov/data/projects/nsite/solutions/water-quality-products)
- [Sentinel-3B OLCI L2 EFR — data.nasa.gov](https://data.nasa.gov/dataset/sentinel-3b-olci-level-2-regional-earth-observation-full-resolution-efr-ocean-color-oc-nea)
- [S3A OLCI Level-2 archive, 2 Jan 2026 — OB.DAAC](https://oceandata.sci.gsfc.nasa.gov/directdataaccess/Level-2/S3A-OLCI/2026/02-Jan-2026/)
- [PACE Rrs wavelength count — Earthdata Forum](https://forum.earthdata.nasa.gov/viewtopic.php?p=23685)
- [Level-2 flag definitions — NASA Ocean Color](https://oceancolor.gsfc.nasa.gov/resources/atbd/ocl2flags/)
- [l2bin default flags — SeaDAS](https://seadas.gsfc.nasa.gov/help-8.2.0/processors/ProcessL2bin.html)
- [Mask impact test, SeaWiFS — OB.DAAC](https://oceancolor.gsfc.nasa.gov/data/reprocessing/r2002/seawifs/masks_n_flags/mchg_gen)
- [Bio-GO-SHIP metagenomes — Larkin et al. 2021, Scientific Data](https://link.springer.com/10.1038/s41597-021-00889-9)
- [MODIS to VIIRS transition — NASA Earthdata](https://www.earthdata.nasa.gov/data/alerts-outages/transition-from-modis-viirs)
- [From Aqua MODIS to PACE OCI hyperspectral reconstruction — Journal of Remote Sensing (news release)](https://www.alphagalileo.org/en-gb/Item-Display/ItemId/270665)
- [DOFA-CLIP (arXiv 2503.06312)](https://arxiv.org/pdf/2503.06312)
- [DOFA source — TorchGeo](https://mytorchgeo.readthedocs.io/en/latest/_modules/torchgeo/models/dofa.html)
- [Copernicus-FM source — TorchGeo](https://mytorchgeo.readthedocs.io/en/latest/_modules/torchgeo/models/copernicusfm.html)
