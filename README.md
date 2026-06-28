# gc-variables-pipeline

JWST/NIRCam time-series photometry pipeline for detecting and characterizing
variable stars in crowded, heavily reddened stellar fields — built for the
bulge globular clusters **Liller 1** and **Terzan 5** (JWST GO-5381), and
applicable to any undithered NIRCam stare.

This is the reproduction package for Burdge et al. (2026), *"JWST Reveals a
Thousand Variable Stars in the Bulge Globular Clusters Terzan 5 and Liller 1."*
It turns raw `uncal` exposures from MAST into the published variable-star
catalog, by synthesizing ~21-second-cadence light curves from the individual
non-destructive detector readout groups within each integration ramp.

> **Status:** scaffolding in progress. See [`ROADMAP.md`](ROADMAP.md) for what
> is implemented vs. forthcoming. The reproducibility appendix of the paper
> walks through the same steps documented here.

## What it does

```
MAST uncal  ──▶  calwebb_detector1   ──▶  group-diff cubes  ──▶  autocorr
(raw)            (pinned CRDS context)     (21 s cadence)         detection
                                                                     │
   published catalog  ◀── catalog build ◀── human REAL/FAKE vetting ◀┘
   (Zenodo DOI)            (+ corrections)   (labels shipped on Zenodo)
```

A novel **lag-1 autocorrelation** detection image finds variables independent of
brightness; a multi-stage **saturation + slope correction** recovers photometry
for sources whose cores are saturated; and a deterministic catalog build
reproduces the published catalog from shipped human-vetting labels.

## Two ways to run

| Path | Data volume | Who it's for |
|------|-------------|--------------|
| **Demo** (`demo/`) | ~minutes from MAST, one detector × one segment | Anyone — runs on a workstation, verifies the toolchain end-to-end |
| **Full reproduction** ([`docs/full_reproduction.md`](docs/full_reproduction.md)) | TB-scale download, multi-day compute | Reproducing the complete published catalog |

To reproduce the **published catalog** without re-running the heavy calibration,
download the intermediate products + vetting labels from the Zenodo record (DOI
in [`CITATION.cff`](CITATION.cff)) and run the downstream catalog build.

## Quickstart (demo)

```bash
conda env create -f environment.yml
conda activate gc-variables
cp config/pipeline.example.yaml config/pipeline.yaml   # edit paths + CRDS context
./demo/run_demo.sh
```

## Reproducibility notes

- **CRDS context is pinned** in `environment.yml` / `config/pipeline.example.yaml`.
  JWST calibration is only deterministic if everyone uses the same reference
  files — set `CRDS_CONTEXT` to the value recorded in the paper.
- **No hardcoded paths.** All paths come from `config/pipeline.yaml`.
- **The catalog is reproducible** from the shipped vetting labels; the original
  REAL/FAKE classification was manual visual inspection (paper §3.8).

## Documentation

- [`docs/flowchart.md`](docs/flowchart.md) — full data-flow diagram
- [`docs/full_reproduction.md`](docs/full_reproduction.md) — the TB-scale path
- [`CLAUDE.md`](CLAUDE.md) — orientation for working in this repo with Claude Code

## Citation

If you use this pipeline, please cite the paper and the software DOI — see
[`CITATION.cff`](CITATION.cff).
