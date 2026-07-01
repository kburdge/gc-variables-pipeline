# unported/ — archival reference copies (NOT runnable here)

These are verbatim copies of production scripts from the author's private
pipeline, kept as the reference implementation for the parts of the workflow
that are not (yet) ported into `pipeline/` + `scripts/`:

- the astrometric calibration chain (`calibrate_lw_astrometry.py`,
  `calibrate_sw_astrometry_lw.py`, `run_astrometry_pipeline.sh`,
  `build_lw_match_table.py`) — documented in `docs/astrometry.md`
- the Terzan 5 Segment-1 dithered extraction/stitch (`add_segment1_terzan5.py`)
- the canonical manual-dedup mapping builder (`build_mapping_from_crossmatch.py`)
  — its *output* is shipped as `dedup_groups.csv` and ingested by
  `scripts/04_build_catalog.py --dedup`
- catalog rebuild / thumbnail / figure utilities

They contain hardcoded paths from the author's machine (`/data/...`) and are
NOT covered by the repo's "no hardcoded paths" guarantee, which applies to
`pipeline/` and `scripts/` only. Treat them as a method record: read them,
don't run them.
