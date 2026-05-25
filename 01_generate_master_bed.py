"""
Genomic Coordinate Standardisation & Master Reference BED Generator

Description:
Ingests raw differential methylation region (DMR) Excel spreadsheets containing
hypermethylated and hypomethylated genomic coordinate attributes. This script
harmonises contrasting inputs, constructs highly specific string-based region identifiers,
omits overlapping duplicate features, and exports a sorted, tab-delimited BED file
optimised for fast interval lookup trees downstream.

Inputs:
  - data/methylated_regions_hyper.xlsx
  - data/methylated_regions_hypo.xlsx

Outputs:
  - data/master_reference.bed
"""


import pandas as pd
import os

#Set up paths
data_dir = "data"
hyper_path = os.path.join(data_dir, "methylated_regions_hyper.xlsx")
hypo_path = os.path.join(data_dir, "methylated_regions_hypo.xlsx")
output_bed = os.path.join(data_dir, "master_reference.bed")

#Load the Excel files and create the master BED file

try:
    df_hyper = pd.read_excel(hyper_path)
    df_hypo = pd.read_excel(hypo_path)
except FileNotFoundError:
    print(f"Note: Original Excel files not found in '{data_dir}/'. Tracking logic shown below.")
    #creating empty dataframes to show the expected structure
    df_hyper = pd.DataFrame(columns=['chr', 'dmr_start', 'dmr_end', 'direction', 'type'])
    df_hypo = pd.DataFrame(columns=['chr', 'dmr_start', 'dmr_end', 'direction', 'type'])

#Combine them into one master list
df_combined = pd.concat([df_hyper, df_hypo], ignore_index=True)

#Format as BED
print("Formatting regions...")

#Create a unique ID: chr_start_end_direction_type
#Example: chr1_124000_129000_hypo_Oligo
df_combined['region_id'] = (
    df_combined['chr'].astype(str) + "_" + 
    df_combined['dmr_start'].astype(str) + "_" + 
    df_combined['dmr_end'].astype(str) + "_" + 
    df_combined['direction'].astype(str) + "_" + 
    df_combined['type'].astype(str)
)

#Select only the BED columns
master_bed = df_combined[['chr', 'dmr_start', 'dmr_end', 'region_id']]

#clean up duplicates just in case the paper's lists had any overlaps
master_bed = master_bed.drop_duplicates()

#sort for better searching performance with the IntervalTree later
master_bed = master_bed.sort_values(['chr', 'dmr_start'])

#Export the master BED file
if not master_bed.empty:
    master_bed.to_csv(output_bed, sep='\t', header=False, index=False)
    print(f"SUCCESS: Master reference bed created with {len(master_bed)} regions.")
    print(f"Saved to: {output_bed}")