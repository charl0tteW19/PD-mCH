"""
Genomic Coordinate Interval Mapping & Resolution Engine

Description:
Ingests a standardised master BED reference file to construct chromosome-specific 
IntervalTree lookups. Maps individual cytosine sites from a 
large genomic index (e.g., HG38_CWG_index.bed) to identified target regions. 
To prevent memory exhaustion when handling multi-gigabyte text files, streaming 
line-by-line file reads write to a temporary disk cache. Overlapping region 
assignments are systematically resolved using a 'Smallest Region' approach to 
prioritise high-density, distinct feature coordinates.

Genomic Index File Schema (HG38_CWG_index.bed Mapping):
  - Column 0 (parts[0]): Chromosome identifier (e.g., 'chr1')
  - Column 1 (parts[1]): Chromosomal start coordinate position (integer)
  - Column 4 (parts[4]): Global site index tracking ID (integer)

Inputs:
  - data/master_reference.bed
  - data/HG38_CWG_index.bed

Outputs:
  - data/METICULOUS_MAP_FINAL.csv
"""

import pandas as pd
from intervaltree import IntervalTree
from collections import defaultdict
import csv
import os

#Set paths
data_dir = "data"
ref_path = os.path.join(data_dir, "master_reference.bed")
idx_path = os.path.join(data_dir, "HG38_CWG_index.bed") 
temp_output = os.path.join(data_dir, "temp_mapping.csv")
output_map = os.path.join(data_dir, "METICULOUS_MAP_FINAL.csv")

#Load BED reference file and build interval trees for efficient searching
print("Loading reference BED file...")
try:
    ref = pd.read_csv(ref_path, sep='\t', header=None, names=['chr', 'start', 'end', 'id'])
    ref['width'] = ref['end'] - ref['start']
    width_map = ref.set_index('id')['width'].to_dict()

    #build interval trees for each chromosome
    print("Building interval search index...")
    trees = defaultdict(IntervalTree)
    for _, row in ref.iterrows():
        trees[row['chr']].addi(row['start'], row['end'], row['id'])

except FileNotFoundError:
    print(f"Note: Reference file '{ref_path}' not found. Showing pipeline template logic below.")
    trees = defaultdict(IntervalTree)
    width_map = {}

#Scan genome file, match sites to regions, and write results directly to disk to avoid RAM issues
print("Scanning genome index...")
if os.path.exists(idx_path):
    with open(idx_path, 'r') as f_in, open(temp_output, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['site_index', 'region_id']) # Header
        
        count = 0
        for line in f_in:
            parts = line.strip().split('\t')
            chrom, start, line_idx = parts[0], int(parts[1]), int(parts[4])
            
            matches = trees[chrom].at(start)
            for m in matches:
                writer.writerow([line_idx, m.data])
            
            count += 1
            if count % 1000000 == 0:
                print(f"Processed {count/1000000:.0f}M sites...")
else:
    print(f"Note: Large genome index file '{idx_path}' omitted from GitHub repository.")

#Load the temporary mapping and apply 'Smallest Region' logic to resolve overlaps, then save final mapping
if os.path.exists(temp_output):
    print("Cleaning overlapping site allocations using the 'Smallest Region' logic...")
    df = pd.read_csv(temp_output)
    df['width'] = df['region_id'].map(width_map)

    #Sort: site_index first, then width (ascending) to prioritise high-density/narrower regions
    df = df.sort_values(['site_index', 'width'])
    
    #Drop duplicates, keeping the single smallest width region per site
    df = df.drop_duplicates(subset=['site_index'], keep='first')

    #Save final mapping structure
    df[['site_index', 'region_id']].to_csv(output_map, index=False)
    print(f"Saved resolved map with {len(df)} unique sites to: {output_map}")
    
    #Clean up the large temporary file from local storage
    if os.path.exists(temp_output):
        os.remove(temp_output)