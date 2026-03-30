#!/usr/bin/env python3
import os
import glob
from jwst.pipeline import Image2Pipeline
def main():
    # Directory containing the *_rateints.fits files.
    rateints_dir = '.'
    # Directory to store the resulting *_calints.fits files.
    output_dir = './calints'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all *_rateints.fits files in the directory.
    rateints_files = sorted(glob.glob(os.path.join(rateints_dir, '*_rateints.fits')))
    print(f"Found {len(rateints_files)} rateints files.")
    
    # Create an instance of the Image2Pipeline.
    image2_pipeline = Image2Pipeline()
    
    # Process each rateints file.
    for rateints_file in rateints_files:
        print(f"Processing {rateints_file}...")
        # Running the pipeline on a rateints file produces a _calints.fits product.
        result = image2_pipeline.call(rateints_file, output_dir=output_dir, save_results=True)
        print(f"Finished processing {rateints_file}.")
    
if __name__ == "__main__":
    main()

