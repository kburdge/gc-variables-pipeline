#!/usr/bin/env python3
"""
Script to run the JWST Detector1Pipeline on every '*_uncal.fits' file
in the current directory in parallel, while limiting each Detector1Pipeline
call to a single core for the jump and ramp_fit steps.
"""
import os
import glob
from multiprocessing import Pool
from jwst.pipeline import Detector1Pipeline

# --- 1) Discover all uncal files in cwd
input_files = sorted(glob.glob('*_uncal.fits'))

# --- 2) Prepare output directory
output_dir = 'detector1_output'
os.makedirs(output_dir, exist_ok=True)

# --- 3) Worker function
def process_file(infile):
    print(f"[PID {os.getpid()}] Processing {infile} …")
    Detector1Pipeline.call(
        infile,
        save_results=True,
        save_calibrated_ramp=True,
        output_dir=output_dir,
        steps={
            'jump':     {'maximum_cores': '1'},
            'ramp_fit': {'maximum_cores': '1'}
        }
    )
    print(f"[PID {os.getpid()}] Finished {infile}")

if __name__ == '__main__':
    if not input_files:
        print("No '*_uncal.fits' files found in current directory.")
        exit(1)

    num_workers = min(4, len(input_files))  # adjust based on your machine or files
    print(f"Found {len(input_files)} uncal files. Running with {num_workers} workers…")

    with Pool(processes=num_workers) as pool:
        pool.map(process_file, input_files)

    print("All done! Outputs are in the 'detector1_output' directory.")

