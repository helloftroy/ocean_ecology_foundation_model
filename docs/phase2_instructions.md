# Ocean Foundation Model — Phase 2: Preprocessing and Initial Training Experiments

_Received 2026-11-12. Saved verbatim; the storage revision that followed it is appended at the end
and takes precedence over Objective 1's "SST storage" paragraph._

## Overall objective

Build the first working version of the ocean foundation model using the Florida/Gulf pilot dataset acquired in Phase 1.

Proceed through geospatial preprocessing, multisensor alignment, construction of training sequences, and small-scale training experiments comparing masked reconstruction and JEPA-style objectives.

This is an **engineering and architecture-validation pilot**, not the final model. Keep architectural choices configurable because the JEPA objective, modality balancing, and native-resolution OLCI branch will be reviewed with Andre.

The primary questions are:

1. Can the four optical satellite families and physical variables be processed into a consistent, reusable training dataset?
2. Can the model learn from irregular observations with missing sensors?
3. Does JEPA train stably and produce non-collapsed representations?
4. How much PACE hyperspectral information is retained in the learned embeddings?
5. Does including physical observations improve the representations?
6. Can we preserve meaningful 300 m OLCI information for a future fine-resolution output?

## Objective 1 — Finalize quality-control policies

Update the preprocessing pipeline using the Phase 1 findings.

**Swath-edge masking**

Remove `HISATZEN` from the default hard mask during the pilot, but preserve the original flags and sensor-zenith angles.

Before accepting these pixels, generate per-sensor diagnostics comparing coverage and reflectance distributions with and without `HISATZEN` masking.

Check for spatial artifacts, implausible reflectance values, and systematic spectral changes at high angles.

Maintain a stricter quality subset for independent spectral evaluation.

Retain hard rejection for genuinely invalid observations, including failed atmospheric correction, navigation failures, missing data, and saturation. Keep sensor-specific flags and surface-reflectance quality rules separate where necessary.

**SST storage**

Use regionally cropped MUR SST data, including sufficient geographic margins for physical context. Preserve timestamps, units, original metadata, and provenance.

Do not store the entire global daily MUR archive unless needed for another purpose.

## Objective 2 — Build a consistent geospatial dataset

Implement the proposed nested spatial grid:

- **Optical tile:** 38.4 × 38.4 km.
- **Shared representation grid:** 32 × 32 cells at 1.2 km.
- **Native OLCI grid:** 128 × 128 cells at 300 m.
- **Physical fields:** Retain their original resolution and provide a separate coarse representation, including optional larger geographic context.

Use fixed, reproducible tile identifiers and a suitable local equal-area projection.

Preserve individual satellite overpasses and UTC acquisition times rather than immediately combining all observations into daily composites.

Regrid MODIS, VIIRS, and PACE onto the shared 1.2 km grid. Retain native-resolution OLCI data alongside its aggregated 1.2 km representation.

For every observation, preserve:

- Sensor and product identification.
- Observation timestamp.
- Available wavelengths and band metadata.
- Reflectance values and valid-data masks.
- Quality flags and observation geometry.
- Spatial coverage and contributing-pixel counts.

Treat PACE SFREFL (`rhos`) as a separate optional optical view. Do not mix its values with ocean-colour \(R_{rs}\) during normalization.

Use a chunked, efficiently readable format such as Zarr for the processed training data.

**Deliverable:** Reproducible tile-building pipeline and a small collection of visually inspected multisensor examples.

## Objective 3 — Construct temporal training examples

Create a dataset loader that assembles irregular multisensor observations into training sequences.

Initial configuration:

- Target region: Florida/Gulf pilot.
- Optical context: preceding 30 days for causal mode.
- Retrospective mode: observations within ±15 days.
- Physical lookback: configurable, starting at 90 days.
- Optical sequence lengths: benchmark 8, 16, and 24 observations.
- Missing sensors are permitted.

Retain the exact observation times and relative-time offsets. Do not insert artificial observations to fill gaps.

Support multiple types of masking:

1. Spatial patches.
2. Entire sensor views.
3. Entire observation times.
4. PACE observations while heritage sensors remain available.
5. Physical fields or the complete physics stream.

Ensure sufficient real observations remain visible to provide learning context.

Create grouped training and validation partitions by geographic tiles and time blocks. Prevent overlapping temporal windows from leaking observations across partitions.

**Deliverable:** A data loader that returns correctly aligned optical and physical observations, masks, timestamps, wavelength metadata, and spatial coordinates.

## Objective 4 — Implement a configurable experimental encoder

Build a modest shared encoder compatible with the proposed architecture.

**Optical encoder**

Use a wavelength-aware spectral embedding inspired by Clay/DOFA, with spatial attention over the 1.2 km token grid.

Support different band counts and wavelengths across sensors, including the full available PACE spectrum.

Retain masks for missing bands and pixels. Use a signed reflectance transform such as `asinh` rather than clipping all negative values.

**Physical encoder**

Use a small variable-aware encoder for SST, surface currents, winds, precipitation, bathymetry, and other available physical fields.

Keep the physical branch separate before fusion with the optical representation.

**Temporal encoder**

Use relative-time information to learn relationships across irregular observations.

Do not require absolute date or geographic coordinates as direct backbone inputs.

**Important:** Keep encoder outputs, masking policies, loss functions, and auxiliary decoders modular. We should be able to change the objective after discussing it with Andre without rebuilding the dataset or entire model.

Start with the 1.2 km backbone. Preserve OLCI 300 m inputs and build a small prototype of the native-resolution branch, but defer finalizing its architecture and loss until the shared backbone is working.

## Objective 5 — Compare three training objectives

Run small-scale experiments with a shared dataset and comparable encoder capacity.

| Experiment | Training objective |
|---|---|
| A | Masked reflectance reconstruction, with cross-view prediction where applicable |
| B | JEPA latent prediction + SIGReg |
| C | JEPA + SIGReg + PACE spectral-residual reconstruction |

Use the same underlying observations and fixed validation partitions.

Start with short debugging runs and then train each candidate enough to assess stable learning. Record the actual computation and number of training examples processed so comparisons remain interpretable.

For JEPA, record separate prediction losses for optical sensors and physical-variable groups. Implement normalization by valid target tokens and make modality-loss weights configurable.

Avoid assuming that averaging every view equally is optimal.

For SIGReg and the target encoder, follow the reference implementation carefully and explicitly test for collapsed representations.

**Do not commit to final loss weights or objective selection yet.**

## Objective 6 — Test whether PACE spectral information survives training

This is a central experiment.

The goal is to determine whether JEPA preserves information specific to PACE's hyperspectral observations, rather than only information predictable from broadband heritage sensors.

On held-out PACE observations:

1. Freeze each pretrained encoder.
2. Train the same lightweight spectral decoder to reconstruct PACE reflectance from its embeddings.
3. Compare reconstruction error across wavelengths.
4. Pay particular attention to narrow spectral features and spectral information not represented by MODIS/VIIRS bands.
5. Compare embeddings generated with PACE visible against embeddings generated using heritage sensors alone.
6. Test whether adding the PACE residual objective improves retention of hyperspectral detail.

Calculate more than overall spectral RMSE. Include wavelength-specific error, spectral-derivative error, and error on the portion of the spectrum not explained by heritage-band reconstruction.

Distinguish between information that is present but not retained in a PACE-visible embedding and information that cannot be inferred from heritage sensors.

**Deliverable:** A comparative spectral-retention report for experiments A, B, and C, with plots by wavelength.

## Objective 7 — Evaluate embedding quality and physical-data contributions

For each trained model, assess:

**Representation stability**

- Embedding variance and effective rank.
- Training versus validation losses.
- Evidence of representation collapse.
- Similarity of embeddings generated from different available sensor combinations.
- Sensitivity to removal of individual modalities.

**Physics ablation**

Compare frozen embeddings with and without physical inputs. The optical branch should also be tested independently so we can assess how much ecological information comes from the satellite observations themselves.

**Initial biological probes**

Where existing matched observations are available, use small frozen-embedding prediction heads for Gulf eDNA community components, phytoplankton pigments, or other established targets.

Compare against simple metadata and physical-variable baselines, and Granite where practical.

These are preliminary indicators of representation usefulness, not definitive downstream benchmarks.

**High-resolution proof of concept**

Test whether a lightweight OLCI branch can reconstruct held-out native-resolution spatial features better than simply upsampling the 1.2 km representation.

Do not claim an effective resolution of 300 m without this evidence.

## Objective 8 — Deliver experimental results and a recommendation

At the end of this phase, provide:

1. **Preprocessing summary:** Grid alignment, quality masking, valid coverage, and representative sample images.
2. **Training pipeline:** Reusable loaders, model components, and configuration files.
3. **Objective comparison:** Reconstruction versus JEPA versus JEPA + PACE residual.
4. **PACE spectral-retention results:** Errors by wavelength and comparison of hyperspectral information retained.
5. **Representation diagnostics:** Collapse checks, modality sensitivity, and initial probes.
6. **Performance benchmarks:** Memory, throughput, and training stability on the available GPUs.
7. **Open architecture questions:** Concrete issues to discuss with Andre before global training.

### Success criteria

This phase is successful if we produce a reproducible dataset and working training framework, demonstrate stable learning with at least one objective, and quantify whether JEPA retains enough information to justify further investment.

JEPA does not need to win. If masked reconstruction performs better, document that result and retain it as the preferred baseline.

**Do not begin global training until the initial objectives have been compared and the remaining architecture decisions reviewed.**

---

**Revision to Phase 2 — Data storage and MUR SST**

Retain the complete global daily MUR SST archive (~2.2 TB) in the shared raw-data directory. We anticipate using these observations for global foundation-model training, so avoid discarding files that would need to be downloaded again.

Maintain separate processed-data directories:

- `processed/regional/` — Florida/Gulf pilot inputs, including regionally cropped SST and other physical fields, aligned satellite tiles, and training-ready datasets.
- `processed/global/` — Create this directory now but leave it empty until global processing begins.

Both datasets should reference the same immutable raw-data archive rather than duplicate original downloads. Maintain manifests documenting source files, processing versions, spatial/temporal subsets, and transformations so that datasets can be regenerated consistently.

Apply this principle to other downloaded satellite and physical products as well: preserve reusable originals, and keep regional and global training inputs separate.

---

# Phase 2 Revision — Physics and Nutrient-Aware Pilot Experiments

_Received 2026-11-12, together with three adjustments from the user that take precedence over it
(listed first, verbatim)._

**User adjustments (take precedence):**

1. Replace the Bio-GO-SHIP gene benchmark with Florida metabarcoding. Use your 16S/18S ASV tables, including NMF community components if available. Evaluate individual components, dominant taxa, and overall community composition. Use spatially and temporally grouped splits.

2. Retain the nutrient-supervision experiment. We can test whether PISCES phosphate, nitrate, and silicate used as training targets only improve the embeddings' ability to predict Florida microbial communities. If measured nutrients are available alongside the metabarcoding samples, they become an especially valuable independent evaluation target. This also lets us see whether different community components benefit from different environmental information.

3. Use Florida Sargassum observations for the seaweed benchmark. Start with same-period detection or density, then evaluate seven-day prediction using the physics-aware representation. Check that historical SaWS observations provide enough valid Sargassum-positive examples; the Gulf test area should include waters where mats are regularly observed.

**Revision as received:**

## Objective

Treat the contribution of physical variables and modeled nutrients as experimental questions, not predetermined requirements of the foundation model.

Our Bio-GO-SHIP experiments show that environmental variables improve the prediction of some microbial functions but not others. Adding all environmental variables indiscriminately can reduce median prediction performance.

The pilot should determine whether physical inputs and nutrient-supervised training produce better general-purpose embeddings, and which biological functions benefit.

## Stage 1 — Select the optical pretraining objective

Train and compare three optical-only models using matched data and architectures:

- **A:** Masked optical reconstruction.
- **B:** JEPA latent prediction + SIGReg.
- **C:** JEPA + SIGReg + PACE spectral-residual preservation.

Assess representation stability, PACE spectral retention, frozen-embedding probes, and training behavior.

Select the strongest approach for further experimentation. Do not automatically select JEPA if reconstruction performs better.

## Stage 2 — Evaluate environmental information

Starting from the winning optical objective, train three additional configurations:

- **Physics:** Add physical observations as inputs, including SST, surface currents, winds, precipitation, and bathymetry.
- **Nutrient targets:** Retain optical-only inputs, but add a small auxiliary head trained to predict PISCES nutrient estimates.
- **Physics + nutrient targets:** Combine physical inputs with auxiliary nutrient prediction.

Use identical validation splits and comparable model capacities.

Keep physics input optional through modality dropout. Models using physics should support evaluation with and without those inputs.

Nutrient estimates must never enter the encoder as inputs. Start with phosphate, nitrate, and silicate, selecting a documented PISCES product and version. Normalize nutrient targets, account for their coarse spatial resolution, and use a configurable auxiliary-loss weight.

Implement nutrient prediction as a simple auxiliary regression task initially. Leave open the possibility of converting it to a latent JEPA target after architectural review.

## Evaluation A — Bio-GO-SHIP microbial functions

_Replaced by user adjustment 1 (Florida metabarcoding); kept here for the record._

Use the existing panel of 414 newly predictable genes, supplemented by nutrient-uptake and transport genes, particularly phosphate-associated `pho` and `pst` functions.

Prepare sample-matched optical observations at available Bio-GO-SHIP locations and dates. Assess coverage before expanding acquisition, recognizing that these evaluation sites extend beyond the Gulf pilot.

For each model, calculate:

- Per-gene R² and improvement over the optical baseline.
- Mean and median improvement across genes.
- Fraction of genes that improve or worsen.
- Performance by gene function or pathway.
- Consistency across held-out cruises.

Compare frozen embeddings against direct environmental-variable baselines, including appropriately selected physical fields.

For phosphate-associated genes, separately test whether the incremental value of adding PISCES phosphate decreases after nutrient-supervised pretraining, and whether embedding-only prediction actually improves.

Where measured phosphate is available, retain it as an independent evaluation and downstream comparison.

Avoid feature selection using held-out test data.

## Evaluation B — Sargassum

Use available USF SaWS historical Sargassum observations to evaluate:

1. Same-period Sargassum occurrence or density.
2. Sargassum distribution approximately seven days later, using only past observations as inputs.

Compare optical-only and physics-aware embeddings.

Include persistence and simple current-driven transport baselines for the forecast task. Use geographically and temporally separated test periods.

Keep SaWS products out of foundation-model pretraining.

## Evaluation C — Phytoplankton and fish

Evaluate frozen embeddings against available HPLC pigment and Gulf fish observations.

Compare optical-only, physical-only, and combined representations where possible.

Treat gains as target-specific. A model that improves certain ecological targets without improving every target can still be valuable.

## Required outputs

Produce a consolidated report comparing the six training configurations, including:

- PACE spectral-information retention.
- Embedding stability and evidence of collapse.
- Per-gene and per-functional-group improvements.
- Contribution of physical inputs versus nutrient-supervised training.
- Sargassum detection and forecasting performance.
- HPLC pigment and fish prediction performance.
- Failure cases, negative results, and recommendations.

**Do not assume that the most complex model is the best model.** The final selection should prioritize transferable ecological information, preservation of the original optical signal, and performance across independent datasets.
