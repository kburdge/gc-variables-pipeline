#!/usr/bin/env python3
"""
SourceTracker - Per-source debug logging for the JWST photometry pipeline.
===========================================================================

PURPOSE:
    When debugging why a specific source (by RA/Dec) was included, excluded,
    or mischaracterized by the pipeline, activate the SourceTracker to produce
    a detailed per-source log that traces every decision point.

    This replaces the need to manually replicate pipeline stages with
    standalone scripts. Instead, the pipeline itself tells you exactly what
    happened to your source.

ACTIVATION (CLI - ramp_pipeline.py):
    cd /data/Globulars_Pipeline
    python /data/Globulars_Pipeline/code/core/ramp_pipeline.py \\
        --target Liller1 --segment Segment3 --detectors nrcb4 --skip-cube \\
        --track-sources track.txt \\
        --track-output tracker_logs

ACTIVATION (CLI - variability_filter.py):
    cd /data/Globulars_Pipeline
    python /data/Globulars_Pipeline/code/analysis/variability_filter_v3.py \\
        --track-sources track.txt \\
        --track-output tracker_logs

INPUT FORMAT (track.txt):
    Lines of "RA Dec Label", whitespace-separated. Comments (#) and blank
    lines are ignored. Label is optional (auto-generated from coordinates).

        # RA           Dec          Label
        263.364556    -33.379315    src5837
        263.365040    -33.380042    src1524

    Or inline (no file needed):
        --track-sources "263.364556,-33.379315,src5837;263.365040,-33.380042,src1524"

ACTIVATION (Python API):
    from source_tracker import SourceTracker
    tracker = SourceTracker([
        (263.364556, -33.379315, 'src5837'),
        (263.365040, -33.380042, 'src1524'),
    ], match_radius_arcsec=0.5, output_dir='tracker_logs')

MULTIPROCESSING:
    The ramp pipeline uses fork-based multiprocessing with global variables.
    The tracker is stored as _global_tracker. Forked workers inherit a copy.
    Each worker logs to its local copy, then attaches entries to the result
    dict as result['_tracker_entries']. The parent process merges entries
    after each worker returns via tracker.merge_worker_entries().

OUTPUT:
    One log file per tracked source, e.g.:
        tracker_logs/src5837_263.3646_-33.3793.log

    Log entries are tagged by pipeline stage:

        [DETECT]      Source detection (DAOStarFinder morphology, significance)
        [DETECT_MISS] Source NOT detected (nearest detection distance shown)
        [PHOTOMETRY]  Aperture photometry (pixel count, NaN pixels, flux stats)
        [SATURATION]  Saturation test result
        [CLIP]        IQR outlier clipping (points before/after)
        [LS]          Lomb-Scargle periodogram (freq, period, power, significance)
        [BLS]         Box Least Squares transit search
        [SL_RATIO]    String-length ratio (noise metric)
        [FILTER]      Period/frequency filter pass/fail with exact thresholds
        [ROUTE]       Output routing (lightcurves_ramp/ vs lightcurves_ramp_bad/)
        [VARFILT]     Variability filter v6 (amplitude, drift, criteria)
        [DEDUP]       Deduplication (kept/removed, correlation, neighbor info)
        [XMATCH]      Cross-segment matching (matched/unmatched, separation)

TROUBLESHOOTING GUIDE (for LLMs and humans):

    Q: "Why was source X not detected?"
       Check [DETECT_MISS]. It shows the distance to the nearest DAOStarFinder
       detection. If no DETECT entry exists at all, increase --track-radius.
       Common causes: morphology cuts (sharpness/roundness), below 5-sigma.

    Q: "Source was detected but has no lightcurve file?"
       Check [FILTER]. The ramp pipeline requires best_freq <= max_freq_hr
       (default 3.0 cyc/hr = 20 min period) AND n_points >= 50. Also check
       [PHOTOMETRY] for flux issues and [CLIP] for excessive outlier removal.

    Q: "Source has a lightcurve but was filtered by variability_filter?"
       Check [VARFILT]. For P < 720 min, sources auto-pass. For P = 720 min
       (no detected period), need drift_ratio > 9.0 OR amplitude >= 5%.

    Q: "Source passed filter but was deduplicated away?"
       Check [DEDUP]. Shows if source was merged with a brighter neighbor
       within 0.5" and whether correlation exceeded the threshold.

    Q: "Source is in one segment but not the other?"
       Check [XMATCH]. Shows whether a counterpart was found within 0.3".
       Also run the tracker on BOTH segments to see the full picture.

    Q: "Source lightcurve was routed to bad_dir?"
       Check [ROUTE]. Either is_saturated=True or sl_ratio > threshold (100).

NULL TRACKER:
    When --track-sources is not provided, all code paths use NULL_TRACKER,
    a singleton with no-op methods. This ensures zero performance overhead
    in normal operation. The only cost is a single `if is_tracked:` boolean
    check per source per decision point in the worker.
"""

import os
from datetime import datetime


class NullTracker:
    """No-op tracker used when source tracking is disabled. Zero overhead."""

    def log(self, ra, dec, stage, message):
        pass

    def log_all(self, stage, message):
        pass

    def check_detection(self, positions, sig_vals, wcs, sources_table=None,
                        stage='DETECT'):
        pass

    def is_tracking(self, ra, dec):
        return False

    @property
    def active(self):
        return False

    def get_entries(self, label):
        return []

    def merge_worker_entries(self, worker_entries):
        pass

    def flush(self, section_header=None):
        pass


# Module-level singleton — use this everywhere when tracking is disabled
NULL_TRACKER = NullTracker()


class SourceTracker:
    """
    Tracks specific sources through the pipeline, producing per-source logs.

    Parameters
    ----------
    sources : list of (ra, dec, label) tuples
        Sources to track. RA/Dec in degrees, label is a string identifier.
    match_radius_arcsec : float
        Radius in arcseconds for matching tracked sources to detections.
    output_dir : str
        Directory for per-source log files.
    """

    def __init__(self, sources, match_radius_arcsec=0.5, output_dir='tracker_logs'):
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        self._sources = []
        self._match_radius = match_radius_arcsec
        self._output_dir = output_dir
        self._entries = {}  # label -> list of (timestamp, stage, message)

        for ra, dec, label in sources:
            sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            self._sources.append({
                'ra': ra, 'dec': dec, 'label': label, 'skycoord': sc
            })
            self._entries[label] = []

    @property
    def active(self):
        return len(self._sources) > 0

    def snapshot_entries(self):
        """Save current entries and clear them. Returns saved entries.

        Use before forking multiprocessing workers so they start with
        empty entries (avoids double-counting pre-fork log entries).
        Call restore_entries() after the pool completes.
        """
        saved = {label: list(entries) for label, entries in self._entries.items()}
        for label in self._entries:
            self._entries[label] = []
        return saved

    def restore_entries(self, saved):
        """Restore saved entries (prepend to current entries)."""
        for label, entries in saved.items():
            if label in self._entries:
                self._entries[label] = entries + self._entries[label]

    def _match(self, ra, dec):
        """Return label of the matching tracked source, or None."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        for src in self._sources:
            sep = coord.separation(src['skycoord']).arcsec
            if sep <= self._match_radius:
                return src['label']
        return None

    def is_tracking(self, ra, dec):
        """Check if a given RA/Dec matches any tracked source."""
        return self._match(ra, dec) is not None

    def log(self, ra, dec, stage, message):
        """Log a message for a source if it matches a tracked source."""
        label = self._match(ra, dec)
        if label is not None:
            ts = datetime.now().strftime('%H:%M:%S')
            self._entries[label].append((ts, stage, message))

    def log_all(self, stage, message):
        """Log a message to ALL tracked sources (for summary/context info)."""
        ts = datetime.now().strftime('%H:%M:%S')
        for label in self._entries:
            self._entries[label].append((ts, stage, message))

    def check_detection(self, positions, sig_vals, wcs, sources_table=None,
                        stage='DETECT'):
        """
        Check which tracked sources were/were not detected by DAOStarFinder.

        Parameters
        ----------
        positions : Nx2 array of pixel (x, y)
        sig_vals : array of significance values
        wcs : astropy WCS
        sources_table : astropy Table, optional
            Full DAOStarFinder output table (for sharpness/roundness logging).
        stage : str
            Log stage tag.
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        import numpy as np

        # Convert detected pixel positions to sky coordinates
        world = wcs.all_pix2world(positions, 0)
        det_coords = SkyCoord(ra=world[:, 0] * u.deg, dec=world[:, 1] * u.deg)

        for src in self._sources:
            seps = src['skycoord'].separation(det_coords).arcsec
            nearest_idx = int(np.argmin(seps))
            nearest_sep = float(seps[nearest_idx])

            if nearest_sep <= self._match_radius:
                px, py = positions[nearest_idx]
                sig = float(sig_vals[nearest_idx])
                msg = (f"MATCHED detection idx={nearest_idx}, "
                       f"px=({px:.1f}, {py:.1f}), sep={nearest_sep:.3f}\", "
                       f"sig={sig:.1f}")

                # Add morphology if DAOStarFinder table is available
                if sources_table is not None:
                    try:
                        row = sources_table[nearest_idx]
                        msg += (f", sharp={float(row['sharpness']):.4f}"
                                f", round1={float(row['roundness1']):.4f}"
                                f", round2={float(row['roundness2']):.4f}")
                    except (KeyError, IndexError):
                        pass

                self.log(src['ra'], src['dec'], stage, msg)
            else:
                # Report nearest detection even though it's outside radius
                px, py = positions[nearest_idx]
                sig = float(sig_vals[nearest_idx])
                self.log(src['ra'], src['dec'], f'{stage}_MISS',
                         f"NOT detected within {self._match_radius}\". "
                         f"Nearest at {nearest_sep:.2f}\" "
                         f"(idx={nearest_idx}, px=({px:.1f}, {py:.1f}), "
                         f"sig={sig:.1f})")

    def get_entries(self, label):
        """Return list of (timestamp, stage, message) for a label."""
        return self._entries.get(label, [])

    def merge_worker_entries(self, worker_entries):
        """
        Merge log entries returned from a multiprocessing worker result dict.

        Parameters
        ----------
        worker_entries : dict of {label: [(timestamp, stage, message), ...]}
        """
        if not worker_entries:
            return
        for label, entries in worker_entries.items():
            if label in self._entries:
                self._entries[label].extend(entries)

    def flush(self, section_header=None):
        """
        Write accumulated log entries to per-source log files.

        Uses append mode so entries from multiple pipeline stages
        (ramp_pipeline then variability_filter) accumulate in one file.

        Parameters
        ----------
        section_header : str, optional
            Header line to prepend (e.g., "=== Segment3 / nrcb4 ===").
        """
        os.makedirs(self._output_dir, exist_ok=True)

        for src in self._sources:
            label = src['label']
            entries = self._entries.get(label, [])

            ra_str = f"{src['ra']:.4f}"
            dec_str = f"{src['dec']:+.4f}"
            filename = f"{label}_{ra_str}_{dec_str}.log"
            filepath = os.path.join(self._output_dir, filename)

            # Check if file exists (for header vs append behavior)
            file_exists = os.path.exists(filepath)

            with open(filepath, 'a') as f:
                if not file_exists:
                    f.write(f"# SourceTracker log for {label}\n")
                    f.write(f"# RA={src['ra']:.6f}  Dec={src['dec']:.6f}\n")
                    f.write(f"# Match radius: {self._match_radius}\"\n\n")

                if section_header:
                    f.write(f"\n{section_header}\n")

                for ts, stage, msg in entries:
                    f.write(f"[{ts}] [{stage}] {msg}\n")

            if entries:
                print(f"  Tracker: wrote {len(entries)} entries for {label} "
                      f"-> {filepath}")

        # Clear entries after flushing (so next stage starts fresh)
        for label in self._entries:
            self._entries[label] = []


def load_track_file(path):
    """
    Load tracked sources from a file path or inline string.

    File format (whitespace-separated, # comments allowed):
        # RA           Dec          Label
        263.364556    -33.379315    src5837
        263.365040    -33.380042    src1524

    Inline format (semicolon-separated, comma-delimited fields):
        "263.364556,-33.379315,src5837;263.365040,-33.380042,src1524"

    Returns
    -------
    list of (ra, dec, label) tuples
    """
    sources = []

    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    sources.append((float(parts[0]), float(parts[1]), parts[2]))
                elif len(parts) == 2:
                    ra, dec = float(parts[0]), float(parts[1])
                    sources.append((ra, dec, f"src_{ra:.4f}_{dec:+.4f}"))
    else:
        # Try inline format: "ra,dec,label;ra,dec,label"
        for part in path.split(';'):
            fields = part.strip().split(',')
            if len(fields) >= 3:
                sources.append(
                    (float(fields[0]), float(fields[1]), fields[2].strip()))
            elif len(fields) == 2:
                ra, dec = float(fields[0]), float(fields[1])
                sources.append((ra, dec, f"src_{ra:.4f}_{dec:+.4f}"))

    return sources
