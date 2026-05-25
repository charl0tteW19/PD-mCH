"""
High-Performance Feature Extraction Pipeline for Parkinson's Disease Datasets

Description:
This script is engineered for execution on remote high-performance computing (HPC) 
environments to extract features from large-scale patient cell-free DNA (cfDNA) 
methylation datasets (.txt files in rcsite format). 

Pre-processing Context (Pickle Map Generation):
-------------------------------------------------------------------------------
The prerequisite tracking asset ('marker_mapping_TOP400_charlotte.pkl') is a 
serialised Python dictionary generated prior to running this pipeline. It was 
constructed by filtering a master reference matrix ('METICULOUS_MAP_FINAL.csv') 
against the top 200 highly discriminatory neuronal regions and the top 200 
highly discriminatory oligodendrocyte regions (400 unique regions total) identified 
by the downstream lineage classifier. 

The resulting structure maps specific genomic site indices (individual line IDs) 
directly to their corresponding high-level Region IDs:
    { site_index (int): region_id (str) }
-------------------------------------------------------------------------------

Pipeline Methodology:
1. Loads the serialised marker region lookup map.
2. Streams raw patient sequencing matrices line-by-line to optimise RAM usage.
3. Maps individual genomic sites back to their parent target cell lineages.
4. Normalises raw counts directly into square-root CPM space.
5. Parallelises sample processing across CPU cores using process pooling.
6. Exports a compiled feature matrix labelled by clinical cohort group.
"""

import numpy as np
import os
import pandas as pd
import pickle
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_patient(file_path, mapping_dict, region_names):
    #Reads the 4-column rcsite format line-by-line to protect system memory.
    #Maps Site IDs to the Top 400 regions and calculates Sqrt-CPM normalisation.
    counts_per_region = {name: 0 for name in region_names}
    total_sample_reads = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                try:
                    count = int(parts[0])
                    site_id = int(parts[1])
                except ValueError:
                    continue 
                
                if count > 0:
                    total_sample_reads += count
                    if site_id in mapping_dict:
                        region_id = mapping_dict[site_id]
                        counts_per_region[region_id] += count
        
        if total_sample_reads == 0:
            return os.path.basename(file_path), None, "Zero reads in file"

        # Transform raw metrics into normalised Sqrt-CPM space
        row_data = []
        for r in region_names:
            cpm = (counts_per_region[r] / total_sample_reads) * 1_000_000
            row_data.append(np.sqrt(cpm))
            
        return os.path.basename(file_path), row_data, None

    except Exception as e:
        return os.path.basename(file_path), None, str(e)

def main():
    #Repository Configuration: Update these parameters before execution
    # Reference map asset generated from the top cell-type markers
    PICKLE_PATH  = 'marker_mapping_TOP400_charlotte.pkl'
    OUTPUT_NAME  = "PD_Control_Extracted_Features.csv"
    
    #System plug ins: Replace placeholders with target server directory path patterns
    CTRL_PATTERN = "PATH_TO_CLUSTER_STORAGE/parkinson-controls/*_rcSite.txt"
    PD_PATTERN   = "PATH_TO_CLUSTER_STORAGE/parkinson-cases/*_rcSite.txt"
    
    #Resource Allocation 
    MAX_WORKERS  = 16 

    #

    if not os.path.exists(PICKLE_PATH):
        print(f"Cluster Error: Reference dictionary '{PICKLE_PATH}' not found in runtime path.")
        print("Please ensure the .pkl file is placed in the active execution directory.")
        sys.exit(1)

    if "PATH_TO_CLUSTER_STORAGE" in CTRL_PATTERN or "PATH_TO_CLUSTER_STORAGE" in PD_PATTERN:
        print("Repository Note: Please update the placeholder environment paths (CTRL_PATTERN / PD_PATTERN)")
        print("at the top of the script with your cluster's absolute directory targets before executing.")
        sys.exit(1)

    print("Loading Unified Mapping Dictionary Meta-Asset")
    with open(PICKLE_PATH, 'rb') as f:
        mapping_dict = pickle.load(f)
    
    region_names = sorted(list(set(mapping_dict.values())))

    print("Querying Cluster Storage Nodes for Target Samples")
    ctrl_files = glob.glob(CTRL_PATTERN)
    pd_files = glob.glob(PD_PATTERN)
    all_files = ctrl_files + pd_files
    
    if not all_files:
        print("Cluster Error: Zero files detected. Verify your updated directory path mount points.")
        sys.exit(1)

    print(f"Total Cohort Size: {len(all_files)} files ({len(ctrl_files)} Controls, {len(pd_files)} PD cases)")

    results = []
    sample_names = []
    actual_labels = []

    print(f"\n Initialising Asynchronous Worker Pools ({MAX_WORKERS} Parallel Core Processes)")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        #Submit tasks asynchronously and map them to their specific tracking path
        future_to_file = {executor.submit(process_patient, f, mapping_dict, region_names): f for f in all_files}
        
        #Pull futures as they finish processing to prevent indexing drift
        for i, future in enumerate(as_completed(future_to_file)):
            full_path = future_to_file[future]
            filename, row, err = future.result()
            
            if err:
                print(f" [{i}/{len(all_files)}] Skipping sample {filename} due to error: {err}")
                continue
            
            results.append(row)
            sample_names.append(filename)
            
            #Label based on the structural path keyword markers
            actual_labels.append("Control" if "control" in full_path.lower() else "PD")

            if i % 50 == 0:
                print(f" Core Status: {i}/{len(all_files)} clinical samples processed safely...")

    print("\n Compiling Final Clinical Metric DataFrame")
    final_df = pd.DataFrame(results, index=sample_names, columns=region_names)
    final_df['Actual_Group'] = actual_labels
    
    final_df.to_csv(OUTPUT_NAME)
    print(f"Done")

if __name__ == "__main__":
    main()