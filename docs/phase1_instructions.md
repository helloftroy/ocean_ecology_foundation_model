# Ocean Foundation Model — Phase 1: Pilot Data Acquisition

_Project brief as given on 2026-11-11 (verbatim)._

## Project objective

Prepare a multisensor satellite and physical-ocean dataset for the initial Florida/Gulf of Mexico pilot of a new ocean foundation model.

The model will eventually learn shared representations from four satellite instrument families (MODIS, VIIRS, Sentinel-3 OLCI, and PACE OCI), along with ocean physics and atmospheric forcing. It will incorporate spatial and temporal relationships and support ecological applications including plankton, eDNA, harmful algal blooms, and floating Sargassum.

**For now, focus on data discovery, acquisition, organization, and initial quality checks. Do not begin model training or commit to architecture-dependent preprocessing.**

## Objective 1 — Inventory available datasets

Identify and confirm the relevant data products, temporal coverage, geographic coverage, processing versions, available variables, and download methods.

**Primary satellite products**

| Instrument | Product | Source |
|---|---|---|
| MODIS Aqua | Level-2 ocean-colour \(R_{rs}\) | NASA OB.DAAC |
| VIIRS (SNPP, NOAA-20/21) | Level-2 ocean-colour \(R_{rs}\) | NASA OB.DAAC |
| Sentinel-3A/B OLCI | Level-2 ocean-colour \(R_{rs}\), 300 m full resolution | NASA OB.DAAC |
| PACE OCI | Level-2 OC_AOP \(R_{rs}\) | NASA OB.DAAC |
| PACE OCI | Level-2 SFREFL (`rhos`) | NASA OB.DAAC |

Retain the full available reflectance-band sets, particularly PACE's hyperspectral wavelengths and OLCI's native spatial resolution.

PACE SFREFL is a separate surface-reflectance product relevant to floating Sargassum. Record its processing version and quality flags separately from the ocean-colour products.

Do not download Level-1 data or generate custom Rayleigh-corrected reflectance for MODIS/VIIRS/OLCI yet.

**Physical and atmospheric products**

| Variable | Preferred source |
|---|---|
| Sea-surface temperature | GHRSST MUR L4 |
| Sea level and geostrophic currents | Copernicus Marine altimetry |
| Surface currents, including wind-driven components | Copernicus Marine observation-informed products, where coverage permits |
| Wind speed/direction or vector components | CCMP |
| Precipitation | NASA IMERG |
| Bathymetry | GEBCO |

Document whether each product is directly observed, observation-based analysis, or model/reanalysis-derived. Retain the original physical variables; don't calculate new nutrient or biological proxies yet.

Mixed-layer depth and a broader physical-data footprint remain under consideration. Identify the appropriate products but defer downloading them in bulk.

Exclude modeled chlorophyll, primary production, nitrate, phosphate and other ecosystem-model outputs from pretraining inputs.

## Objective 2 — Establish the pilot geographic and temporal coverage

Use three representative regions:

1. **Southwest Florida shelf:** Tampa Bay to the Florida Keys, emphasizing coastal waters relevant to red tide.
2. **Florida Keys and surrounding reef waters:** Coastal and offshore optical gradients.
3. **Offshore Gulf of Mexico:** Clearer oligotrophic waters, providing contrast to coastal environments.

Define precise geographic bounding boxes for each region and report their areas before downloading. Include enough surrounding ocean to support later tile extraction, rather than clipping exactly to coastlines.

**Temporal sampling**

- **2024–2026:** Retrieve all available days across the selected regions where practical. This is the highest priority because all four instrument families overlap.
- **2018–2023:** Start with three continuous 30–45-day windows per year, distributed across seasons. Expand to full-year coverage if estimated download size and throughput are manageable.
- Include extra temporal coverage around selected storms or bloom events where useful.
- Do not randomly select isolated dates; preserve continuous observation windows for temporal learning.
- Include up to 90 days of preceding physical data for each selected window.

Respect the actual availability and processing cutoffs of each instrument. Missing observations are acceptable and should be recorded rather than filled artificially.

## Objective 3 — Build a reproducible acquisition pipeline

Create Python scripts to discover, inventory, download, and validate data using appropriate archive APIs.

Use `earthaccess` for NASA-hosted observations and the Copernicus Marine Toolbox for ocean-current products where appropriate.

Requirements:

- Search by instrument, product, geographic bounds, and date range.
- Perform a dry-run inventory before bulk downloading.
- Report file counts, total expected bytes, and estimated transfer duration based on measured throughput.
- Download files using resumable, retry-safe operations.
- Avoid downloading the same granule multiple times when regions overlap.
- Preserve original files and product metadata.
- Maintain a manifest containing source, product, version, observation time, geographic coverage, file size, and download/checksum status.
- Allow the pipeline to resume after interruptions without re-downloading verified files.

Where NASA distributes whole satellite swaths, recognize that retrieved files may be considerably larger than the portion covering our regions.

Estimate both the raw-granule storage requirement and the later regional-extraction requirement.

## Objective 4 — Inspect products and perform initial quality checks

For representative files from every sensor, verify:

- Reflectance variable names and dimensions.
- Available wavelengths and units.
- Geographic coordinate variables and acquisition timestamps.
- Cloud, atmospheric-correction, navigation, and other quality flags.
- Fill values, negative reflectance, and missing bands.
- Actual OLCI 300 m resolution and geographic coverage.
- PACE SFREFL band coverage and its suitability for red/NIR floating-algae analysis.

Generate simple diagnostic maps for each instrument over the selected regions.

Also examine observations from overlapping dates to confirm the sensors can be geographically and temporally matched.

**Do not discard pixels based on a final quality-mask policy yet.** Preserve original flags and summarize how candidate masks would affect data availability, especially in coastal waters.

## Objective 5 — Prepare for temporal and multisensor training

Create an observation index linking:

- Geographic region
- Satellite/instrument
- Observation time
- Available wavelengths
- Valid data coverage
- Corresponding physical observations
- Original data-file locations

Identify dates and areas with observations from multiple instruments, especially PACE plus OLCI, MODIS, or VIIRS.

Report observation counts by sensor, year, month, and geographic region.

We will later use this index to construct irregular temporal sequences and masked-view training examples.

For now, avoid committing to the final 1.2 km grid, projection, chip format, or temporal sequence construction until the model architecture is reviewed.

## Objective 6 — Deliver a pilot acquisition report

Before expanding to the full dataset, produce:

1. **Dataset inventory:** Available products, versions, variables, wavelengths, and date ranges.
2. **Storage estimates:** Raw-download and projected processed-data sizes.
3. **Acquisition pipeline:** Reproducible scripts and manifests.
4. **Quality-control summary:** Example maps, missingness, cloud coverage, and sensor overlap.
5. **Transfer benchmarks:** Measured download rates and projected time for completing the pilot.
6. **Data-readiness assessment:** Any issues that may affect spectral harmonization, coastal 300 m coverage, or the eventual model architecture.

### Execution priorities

First complete a small end-to-end acquisition test for each satellite and physical product. Then inventory the complete requested time periods, confirm expected storage, and progressively acquire the 2024–2026 multisensor observations.

Expand the historical archive after the initial data pipeline and quality checks are working.

**Success criterion:** We have a documented, reproducible collection of real multisensor observations across representative coastal and offshore environments, sufficient to begin model pretraining experiments once the architecture is finalized.
