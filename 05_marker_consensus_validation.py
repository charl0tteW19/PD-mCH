"""
Cross-Platform Directional Consensus Validation Engine

Description:
Validates region-specific methylation directionality by intersecting single-cell atlas 
lineage metadata with empirical MeD-seq intensity distributions. Extracts genomic metadata 
(Hyper/Hypo-methylation state and Neuronal/Oligo cell-type lineages) from regional tags. 
Establishes a cohort-wide baseline intensity threshold at a designated percentile (50th median) 
to systematically verify that hyper-methylated markers exhibit elevated read counts while 
hypo-methylated markers track below the threshold, yielding a high-confidence consensus backbone.

Inputs:
  - data/CLEANED_REGRESSION_MATRIX.csv

Outputs:
  - data/VALIDATED_MARKER_LIST_CPM.csv
"""

import pandas as pd
import numpy as np
import os

#set paths
data_dir = "data"
input_path = os.path.join(data_dir, "CLEANED_REGRESSION_MATRIX.csv")
output_path = os.path.join(data_dir, "VALIDATED_MARKER_LIST_CPM.csv")

#Set the 50th percentile (median) acts as the high/low signal threshold delimiter
INTENSITY_PERCENTILE_CUTOFF = 0.50 

print("Loading quality-filtered Sqrt-CPM matrix...")

if os.path.exists(input_path):
    df = pd.read_csv(input_path).fillna(0)

    #Parse metadata from region_id to extract directionality (Hyper/Hypo) and cell type classifications
    def parse_metadata(region_id):
        rid = str(region_id).lower()
        direction = 'Hyper' if 'hyper' in rid else ('Hypo' if 'hypo' in rid else 'Unknown')
        
        #Mapping single-cell atlas lineage identifiers to consolidated cohorts
        if 'oligo' in rid:
            cell_type = 'Oligo'
        elif 'exc' in rid or 'inh' in rid:
            cell_type = 'Neuronal'
        else:
            cell_type = 'Other'
        return pd.Series([direction, cell_type])

    print("Extracting lineage directionality and cell type classifications...")
    metadata = df['region_id'].apply(parse_metadata)
    df['Direction'] = metadata[0]
    df['Cell_Type'] = metadata[1]

    #Calculate average intensity across all samples for each region to establish a unified signal metric
    sample_cols = [c for c in df.columns if c not in ['region_id', 'Direction', 'Cell_Type']]
    df['Avg_Intensity'] = df[sample_cols].mean(axis=1)

    #Establish the high signal threshold based on the specified percentile cutoff of the average intensity distribution
    high_signal_threshold = df['Avg_Intensity'].quantile(INTENSITY_PERCENTILE_CUTOFF)

    # Define validation logic to confirm that hyper-methylated regions exhibit intensities above the threshold while hypo-methylated regions fall below it, ensuring cross-platform consistency
    def validate_consistency(row):
        #Hyper-methylated markers must display signal intensity above the threshold baseline
        if row['Direction'] == 'Hyper':
            return row['Avg_Intensity'] >= high_signal_threshold
        #Hypo-methylated markers are characterized by zero/minimal read counts (below threshold)
        elif row['Direction'] == 'Hypo':
            return row['Avg_Intensity'] < high_signal_threshold
        return False

    print("Evaluating cross-platform signal consistency...")
    df['Is_Validated'] = df.apply(validate_consistency, axis=1)

    #Filter the dataframe to retain only those regions that meet the validation criteria, creating a consensus marker list that aligns with biological expectations of methylation patterns
    validated_markers = df[df['Is_Validated'] == True].copy()

    print(f"\n--- CONSENSUS FILTER RESULTS (CPM SCALE) ---")
    print(f"Total Regions Checked:  {len(df)}")
    print(f"Validated Regions:      {len(validated_markers)}")
    print(f"High Signal Threshold:  {high_signal_threshold:.4f}")
    print(f"\nBreakdown of Validated Markers:")
    print(validated_markers.groupby(['Cell_Type', 'Direction']).size())

    #Save the cross-validated marker backbone
    validated_markers.to_csv(output_path, index=False)
    print(f"\nValidated Marker List saved to: {output_path}")

else:
    print("Note: Input matrix not found. Skipping consensus validation step for public repository.")