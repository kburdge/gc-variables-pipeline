# Full reproduction (TB-scale)

This is the complete path from raw MAST data to the published catalog. It
requires a large download (raw `uncal` for all visits) and multi-day compute
(group-diff cubes are 16 GB per detector). Most users want the [demo](../demo)
or the reproduce-from-products path instead (see below).

## 0. Environment

```bash
conda env create -f environment.yml
conda activate gc-variables    # sets CRDS_PATH/CRDS_SERVER_URL/CRDS_CONTEXT (see environment.yml)
cp config/pipeline.example.yaml config/pipeline.yaml   # edit paths; crds.context is pre-pinned
```

The environment pins `jwst==1.17.1` and `CRDS_CONTEXT=jwst_1322.pmap` — the
exact calibration used for the published reduction. The scripts refuse to run
with an unpinned context.

## 1. Download raw data from MAST

```bash
# preview what will be downloaded
python scripts/00_download_mast.py --config config/pipeline.yaml --list
# download everything (TB-scale; --all is required to confirm)
python scripts/00_download_mast.py --config config/pipeline.yaml --product uncal --all
```
Program GO-5381, NIRCam Module B. `--all` downloads the three undithered
science visits configured in the template (Terzan 5 Seg 2; Liller 1 Seg 3 &
Seg 4). The dithered Terzan 5 Seg 1 visit (used only for a second-epoch
extraction) is not in the template — add `Segment1` to the Terzan5 segments in
your config to fetch it. Files land in `<data_root>/<target>/<segment>/`.

## 2. Calibrate (calwebb_detector1 + Image2)

```bash
for tgt_seg in "Terzan5 Segment2" "Liller1 Segment3" "Liller1 Segment4"; do
    set -- $tgt_seg
    python scripts/01_calibrate.py --config config/pipeline.yaml --target $1 --segment $2
done
```
Runs `Detector1Pipeline` with `save_calibrated_ramp=True` (the ramp files are the
essential product — MAST does not archive them) then `Image2Pipeline` for WCS.
**Determinism depends on the pinned `jwst` version and `CRDS_CONTEXT`.**

## 3. Build cubes + detect + extract

```bash
# group-differenced + zeroframe cubes (all detectors)
python scripts/02_build_cubes.py --config config/pipeline.yaml --target Terzan5 --segment Segment2
# detect variables + extract light curves, both modes (omit --max-sources for the full list)
python scripts/03_extract.py --config config/pipeline.yaml --target Terzan5 --segment Segment2 --mode both
```
Produces the group-diff and zeroframe cubes, autocorrelation reference images,
and the per-detector extraction HDF5 files (positions, detection SNR, clipped
light curves, LS/BLS periods) for both the ramp (972-frame) and zeroframe
(96-frame) modes. Inspect any source with
`python scripts/plot_lightcurve.py ... --source <i>`.

## 4. Vet sources (or load shipped labels)

To reproduce the **published** catalog, skip manual vetting and use the shipped
labels: download `vetting_labels.csv` and `dedup_groups.csv` from the Zenodo
record referenced in the paper's Appendix and place them in the repository
root. To re-vet yourself, sort the diagnostic PNGs into `REAL/` and `FAKE/`
subfolders (this is the human-judgment step described in the paper's
classification section).

`dedup_groups.csv` encodes the author's manual deduplication decisions
(1-arcsec spatial grouping + visual inspection, including 83 groups split by
hand into multiple distinct sources). Shipping it is what makes the published
1,315-object source list — and its exact `master_id`s — reproducible.

## 5. Build the catalog

```bash
# manual dedup groupings -> per-target sources tables + populated light curves
python scripts/04_build_catalog.py --config config/pipeline.yaml \
    --dedup dedup_groups.csv --with-lightcurves
# saturation/slope corrections + best-stage selection (same --dedup!)
python scripts/05_corrections.py --config config/pipeline.yaml --dedup dedup_groups.csv
```
`04_build_catalog.py` reads the shipped dedup groupings (falling back to an
automatic 0.2" dedup of the vetting labels — with a warning — if none are
given; the fallback yields a few percent more objects and different IDs),
refines centroids, runs forced photometry, and writes the per-target `sources`
tables and light curves to `master_variable_catalog.h5`. `05_corrections.py`
then applies the saturation and slope corrections and records the best
correction stage per source/segment/channel (10% integration-scatter gate).
Both stages must be given the same `--dedup` (or none) so their `master_id`s
agree.

(Authors only: regenerate the shipped label products with
`python scripts/export_vetting_labels.py --config config/pipeline.yaml --out vetting_labels.csv`
and `python scripts/export_dedup_groups.py --mapping catalogs/master_source_mapping.json --out dedup_groups.csv`.)

## Reproduce from products (recommended shortcut)

Download the intermediate products (group-diff cubes, extraction HDF5s) and the
label products from the Zenodo record, place them at the paths configured in
`config/pipeline.yaml`, and run only stage 5 (the two catalog commands above).

## Verifying the result

The rebuilt `master_variable_catalog.h5` should contain exactly 1,315 sources
(915 Liller 1, 400 Terzan 5) with the published `master_id`s, and match the
published catalog within photometric tolerance. The port was validated against
the production pipeline: stage-3 extraction fluxes are bit-identical for all
validated sources (both ramp and zeroframe modes), extracted light curves
reproduce the published ones with correlation 1.000, and the
best-correction-stage selection agrees for all validated sources.
