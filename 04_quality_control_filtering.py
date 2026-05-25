"""
Data Cleaning & Low-Sparsity Feature Filtration Engine

Description:
Performs downstream quality control filtering on the consolidated intensity matrix. 
Implements a biological baseline fill where missing data points (NaNs) are explicitly 
coerced to zero, reflecting the zero-methylated fragment architecture of enrichment-based 
MeD-seq assays. Applies a 25% minimum representation threshold across the sample cohort 
to strip out highly sparse, uninformative genomic regions before downstream processing.

Inputs:
  - data/FINAL_CPM_MATRIX.csv

Outputs:
  - data/CLEANED_REGRESSION_MATRIX.csv
"""

import pandas as pd
import numpy as np
import os

#set paths
data_dir = "data"
input_matrix = os.path.join(data_dir, "FINAL_CPM_MATRIX.csv")
output_matrix = os.path.join(data_dir, "CLEANED_REGRESSION_MATRIX.csv")

print("Loading consolidated intensity matrix...")

if os.path.exists(input_matrix):
    df = pd.read_csv(input_matrix)

    #NaNs to 0s (Enrichment-based MeD-seq: Missing = 0 methylated fragments)
    print("Addressing gaps: Filling NaNs with 0...")
    df = df.fillna(0)

    #Filter sparse regions: Retain features with data in at least 25% of samples
    print("Executing sparsity filter (retaining features with >= 25% sample presence)...")
    sample_cols = [c for c in df.columns if c != 'region_id']
    
    #Count how many samples have a signal greater than 0 for each region
    non_zero_counts = (df[sample_cols] > 0).sum(axis=1)
    
    #Filter the dataframe
    df_filtered = df[non_zero_counts >= (len(sample_cols) * 0.25)]
    print(f"Features filtered down from {len(df)} to {len(df_filtered)} active regions.")

    #export the cleaned, filtered matrix for regression tasks
    df_filtered.to_csv(output_matrix, index=False)
    print(f"SUCCESS: Cleaned matrix saved to {output_matrix}")

    # Note: For regression tasks, the matrix is now ready with rows as regions and columns as samples.
    # For machine learning downstream tasks, transpose the matrix so:
    # Rows = Samples, Columns = Genomic Regions (Features)
    # X = df_filtered.set_index('region_id').T
    
else:
    print("Note: Intensity matrix not found. Skipping quality control filter step for public repository.")