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
export CRDS_CONTEXT=jwst_XXXX.pmap      # the context recorded in the paper
cp config/pipeline.example.yaml config/pipeline.yaml   # edit paths + crds.context
```

## 1. Download raw data from MAST

```bash
python scripts/00_download_mast.py --program 5381 --product uncal --out ./data/jwst
```
Program GO-5381, NIRCam Module B. Four visits: Terzan 5 Seg 1 (dithered, second
epoch only) & Seg 2; Liller 1 Seg 3 & Seg 4.

## 2. Calibrate (calwebb_detector1 + Image2)

```bash
python scripts/01_calibrate.py --config config/pipeline.yaml
```
Runs `Detector1Pipeline` with `save_calibrated_ramp=True` (the ramp files are the
essential product — MAST does not archive them) then `Image2Pipeline` for WCS.
**Determinism depends entirely on `CRDS_CONTEXT` being pinned.**

## 3. Build group-diff cubes + detect + extract

```bash
python scripts/02_extract.py --config config/pipeline.yaml --channel sw   # nrcb1-4, F200W
python scripts/02_extract.py --config config/pipeline.yaml --channel lw   # nrcblong, F356W
```
Produces the group-diff cubes, autocorrelation reference images, the extraction
HDF5 files, and the diagnostic PNGs.

## 4. Vet sources (or load shipped labels)

To reproduce the **published** catalog, skip manual vetting and use the shipped
labels:
```bash
python scripts/fetch_zenodo.py --record <DOI> --what vetting_labels
```
To re-vet yourself, sort the diagnostic PNGs into `REAL/` and `FAKE/`
subfolders (this is the human-judgment step described in paper §3.8).

## 5. Build the catalog

```bash
python scripts/03_build_catalog.py --config config/pipeline.yaml
```
Mapping/dedup → centroid refinement → light-curve extraction → saturation &
slope corrections → best-stage selection → `master_variable_catalog.h5`.

## Reproduce from products (recommended shortcut)

Download the intermediate products + vetting labels from Zenodo and run only
stage 5:
```bash
python scripts/fetch_zenodo.py --record <DOI> --what cubes,extraction,vetting_labels
python scripts/03_build_catalog.py --config config/pipeline.yaml
```

## Verifying the result

The rebuilt `master_variable_catalog.h5` should contain 1,315 sources
(915 Liller 1, 400 Terzan 5) and match the published catalog within photometric
tolerance. A smoke comparison script lives in `tests/`.
