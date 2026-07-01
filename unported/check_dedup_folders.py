#!/usr/bin/env python
"""
Scan cross_match folders for unsplit source groups.

For each obj folder (that has no subfolders), checks if the detections
cluster into 2+ spatially distinct groups. Reports folders that likely
need manual splitting.

Usage:
    python check_dedup_folders.py [--threshold 0.3]
"""
import os
import re
import argparse
import numpy as np
from collections import defaultdict

CROSS_MATCH = '/data/Globulars_Pipeline/diagnostics/cross_match'


def parse_coords_from_filename(fn):
    """Extract RA, Dec from diagnostic PNG filename."""
    # Pattern: ..._RA_DEC.png where RA and Dec are decimal degrees
    m = re.search(r'_([\d]+\.[\d]+)_([-]?[\d]+\.[\d]+)\.png$', fn)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def cluster_positions(positions, threshold_arcsec=0.3):
    """Simple friends-of-friends clustering on (ra, dec) positions."""
    if len(positions) <= 1:
        return [positions]
    
    threshold_deg = threshold_arcsec / 3600.0
    assigned = [-1] * len(positions)
    cluster_id = 0
    
    for i in range(len(positions)):
        if assigned[i] >= 0:
            continue
        # Start new cluster
        assigned[i] = cluster_id
        stack = [i]
        while stack:
            curr = stack.pop()
            ra1, dec1 = positions[curr]
            for j in range(len(positions)):
                if assigned[j] >= 0:
                    continue
                ra2, dec2 = positions[j]
                dist = np.sqrt((ra1 - ra2)**2 * np.cos(np.radians(dec1))**2 + (dec1 - dec2)**2)
                if dist < threshold_deg:
                    assigned[j] = cluster_id
                    stack.append(j)
        cluster_id += 1
    
    clusters = defaultdict(list)
    for i, c in enumerate(assigned):
        clusters[c].append(positions[i])
    return list(clusters.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Clustering threshold in arcsec (default 0.3)')
    parser.add_argument('--all', action='store_true',
                        help='Also check folders that already have subfolders')
    args = parser.parse_args()

    folders = sorted([d for d in os.listdir(CROSS_MATCH)
                      if d.startswith('obj') and os.path.isdir(os.path.join(CROSS_MATCH, d))])

    n_issues = 0
    
    for folder in folders:
        fpath = os.path.join(CROSS_MATCH, folder)
        
        # Check if already has subfolders
        subdirs = [d for d in os.listdir(fpath)
                   if os.path.isdir(os.path.join(fpath, d)) and d.isdigit()]
        
        if subdirs and not args.all:
            # Already split — check each subfolder for internal consistency
            for sd in sorted(subdirs):
                sdpath = os.path.join(fpath, sd)
                pngs = [f for f in os.listdir(sdpath) if f.endswith('.png')]
                positions = []
                filenames = []
                for png in pngs:
                    ra, dec = parse_coords_from_filename(png)
                    if ra is not None:
                        positions.append((ra, dec))
                        filenames.append(png)
                if len(positions) < 2:
                    continue
                clusters = cluster_positions(positions, args.threshold)
                if len(clusters) > 1:
                    n_issues += 1
                    print(f'\n*** {folder}/sub{sd}: {len(clusters)} clusters in subfolder ***')
                    for ci, cl in enumerate(clusters):
                        print(f'  Cluster {ci+1} ({len(cl)} detections):')
                        for ra, dec in cl:
                            # Find matching filename
                            for fn, (fra, fdec) in zip(filenames, positions):
                                if fra == ra and fdec == dec:
                                    print(f'    {fn}')
                                    break
            continue

        if subdirs:
            continue

        # No subfolders — check root PNGs
        pngs = [f for f in os.listdir(fpath) if f.endswith('.png') and f != 'summary.png']
        positions = []
        filenames = []
        for png in pngs:
            ra, dec = parse_coords_from_filename(png)
            if ra is not None:
                positions.append((ra, dec))
                filenames.append(png)
        
        if len(positions) < 2:
            continue
        
        clusters = cluster_positions(positions, args.threshold)
        if len(clusters) > 1:
            # Compute max separation between cluster centers
            centers = []
            for cl in clusters:
                ras = [p[0] for p in cl]
                decs = [p[1] for p in cl]
                centers.append((np.mean(ras), np.mean(decs)))
            
            max_sep = 0
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    ra1, dec1 = centers[i]
                    ra2, dec2 = centers[j]
                    sep = np.sqrt((ra1-ra2)**2 * np.cos(np.radians(dec1))**2 + (dec1-dec2)**2) * 3600
                    max_sep = max(max_sep, sep)
            
            n_issues += 1
            print(f'\n*** {folder}: {len(clusters)} clusters, max separation {max_sep:.2f}" ***')
            for ci, cl in enumerate(clusters):
                center_ra = np.mean([p[0] for p in cl])
                center_dec = np.mean([p[1] for p in cl])
                print(f'  Cluster {ci+1} at ({center_ra:.5f}, {center_dec:.5f}), {len(cl)} detections:')
                for ra, dec in cl:
                    for fn, (fra, fdec) in zip(filenames, positions):
                        if fra == ra and fdec == dec:
                            # Shorten filename for readability
                            short = re.sub(r'Terzan5_|Liller1_', '', fn)
                            short = re.sub(r'\.png$', '', short)
                            print(f'    {short}')
                            break
    
    print(f'\n{"="*60}')
    print(f'Total folders needing attention: {n_issues}')


if __name__ == '__main__':
    main()
