# Full reproduction (TB-scale)

This is the complete path from raw MAST data to the published catalog. It
requires a large download (raw `uncal` for all visits) and multi-day compute
(group-diff cubes are 16 GB per detector). Most users want the [demo](../demo)
or the reproduce-from-products path instead (see below).

## 0. Environment

```bash
conda env create -f environment.yml
conda activate gc-variables
export CRDS_PATH=$HOME/crds_cache
export CRDS_SERVER_URL=https://jwst-crds.stsci.edu
export CRDS_CONTEXT=jwst_1322.pmap      # the context used for the published reduction
cp config/pipeline.example.yaml config/pipeline.yaml   # edit paths; crds.context is pre-pinned
```

## 1. Download raw data from MAST

```bash
# preview what will be downloaded
python scripts/00_download_mast.py --config config/pipeline.yaml --list
# download everything (TB-scale; --all is required to confirm)
python scripts/00_download_mast.py --config config/pipeline.yaml --product uncal --all
```
Program GO-5381, NIRCam Module B. Four visits: Terzan 5 Seg 1 (dithered, second
epoch only) & Seg 2; Liller 1 Seg 3 & Seg 4. Files land in
`<data_root>/<target>/<segment>/` as configured in `config/pipeline.yaml`.

## 2. Calibrate (calwebb_detector1 + Image2)

```bash
python scripts/01_calibrate.py --config config/pipeline.yaml
```
Runs `Detector1Pipeline` with `save_calibrated_ramp=True` (the ramp files are the
essential product — MAST does not archive them) then `Image2Pipeline` for WCS.
**Determinism depends entirely on `CRDS_CONTEXT` being pinned.**

## 3. Build group-diff cubes + detect + extract

```bash
# group-differenced cubes (all detectors)
python scripts/02_build_cubes.py --config config/pipeline.yaml --target Terzan5 --segment Segment2
# detect variables + extract light curves (omit --max-sources for the full list)
python scripts/03_extract.py --config config/pipeline.yaml --target Terzan5 --segment Segment2
```
Produces the group-diff cubes, autocorrelation reference images, and the
per-detector extraction HDF5 files (positions, detection SNR, clipped light
curves, LS/BLS periods). Inspect any source with
`python scripts/plot_lightcurve.py ... --source <i>`.

## 4. Vet sources (or load shipped labels)

To reproduce the **published** catalog, skip manual vetting and use the shipped
labels: download `vetting_labels.csv` (and `dedup_groups.csv`) from the Zenodo
record referenced in the paper's Appendix and place them in the repository
root. To re-vet yourself, sort the diagnostic PNGs into `REAL/` and `FAKE/`
subfolders (this is the human-judgment step described in the paper's
classification section).

## 5. Build the catalog

```bash
# vetting labels + dedup -> per-target sources tables + populated light curves
python scripts/04_build_catalog.py --config config/pipeline.yaml --labels vetting_labels.csv
# saturation/slope corrections + best-stage selection
python scripts/05_corrections.py --config config/pipeline.yaml
```
`04_build_catalog.py` deduplicates the vetting labels into unique objects,
refines centroids, runs forced photometry, and writes the per-target `sources`
tables and light curves to `master_variable_catalog.h5`. `05_corrections.py`
then applies the saturation and slope corrections and records the best
correction stage per source/segment/channel (10% integration-scatter gate).

(Authors only: regenerate the shipped labels from the diagnostics tree with
`python scripts/export_vetting_labels.py --config config/pipeline.yaml --out vetting_labels.csv`.)

## Reproduce from products (recommended shortcut)

Download the intermediate products (group-diff cubes, extraction HDF5s) and the
vetting labels from the Zenodo record, place them at the paths configured in
`config/pipeline.yaml`, and run only stage 5 (the two catalog commands above).

## Verifying the result

The rebuilt `master_variable_catalog.h5` should contain 1,315 sources
(915 Liller 1, 400 Terzan 5) and match the published catalog within photometric
tolerance. The port was validated against the production pipeline: extracted
light curves reproduce the published ones with correlation 1.000, and the
best-correction-stage selection agrees for all validated sources.
