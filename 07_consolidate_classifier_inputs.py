"""
Consolidated Master Feature Matrix Generation Engine

Description:
Integrates independent high-throughput intensity datasets by executing an inner join 
on verified genomic region identifiers. Merges the cross-platform validated brain matrix 
(containing N, O, and DN profiles) with the peripheral immune reference matrix (IM). 
This alignment isolates a convergent feature backbone present across all cohorts, 
eliminating non-intersecting batch signatures to establish a standardised input matrix 
for downstream multiclass classification tasks.

Inputs:
  - data/BRAIN_MATRIX_VALIDATED_ONLY.csv
  - data/FINAL_IMMUNE_MATRIX_SQRT_CPM.csv

Outputs:
  - data/FINAL_CLASSIFIER_INPUT_MASTER.csv
"""

import pandas as pd
import os

#path set up
data_dir = "data"
brain_path = os.path.join(data_dir, "BRAIN_MATRIX_VALIDATED_ONLY.csv")
immune_path = os.path.join(data_dir, "FINAL_IMMUNE_MATRIX_SQRT_CPM.csv")
output_path = os.path.join(data_dir, "FINAL_CLASSIFIER_INPUT_MASTER.csv")


if os.path.exists(brain_path) and os.path.exists(immune_path):
    #Load independent matrices
    print("Loading validated brain and processed peripheral immune matrices...")
    df_brain = pd.read_csv(brain_path)
    df_immune = pd.read_csv(immune_path)

    #Inner join matching on 'region_id' ensures we only retain features present in both datasets, maximising classifier input consistency.
    print(f"Aligning {len(df_brain):,} brain-derived genomic regions with {len(df_immune):,} immune regions...")
    master_df = pd.merge(df_brain, df_immune, on='region_id', how='inner')

    #Sort by region_id
    master_df = master_df.sort_values('region_id').reset_index(drop=True)

    #Export
    final_count = len(master_df)
    print(f"\n--- Master Input Feature Matrix Audit ---")
    print(f"Total Convergent Regions (Features): {final_count:,}")
    print(f"Total Consolidated Samples (Columns): {len(master_df.columns) - 1}")

    #Export the reference input matrix
    master_df.to_csv(output_path, index=False)
    print(f"\nMatrix saved to: {output_path}")

    #Verify tracking matrix boundaries
    print(f"First 3 Matrix Columns: {list(master_df.columns[:3])} ... Final Matrix Column: {master_df.columns[-1]}")

else:
    print("Note: Primary input cohort arrays omitted to comply with data safety frameworks.")
    print("Pipeline template structured to process local multi-batch sequencing matrix merges.")