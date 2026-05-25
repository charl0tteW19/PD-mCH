"""
High-Throughput Regional Aggregation & Sqrt-CPM Normalisation Engine

Description:
Ingests raw, high-density single-nucleotide sample files to aggregate methylated read 
counts across mapped target regions. Utilises a Polars out-of-core streaming 
architecture to maintain low RAM utilisation while performing lazy joins against a master 
coordinate lookup file. Computes exact sample library sizes, scales aggregated intensities 
to Counts Per Million (CPM), and applies a variance-stabilising square-root transformation.

Inputs:
  - data/METICULOUS_MAP_FINAL.csv
  - data/raw_samples/*.txt

Outputs:
  - data/FINAL_CPM_MATRIX.csv
"""

import polars as pl
import os
import glob
import gc

#set paths
data_dir = "data"
map_path = os.path.join(data_dir, "METICULOUS_MAP_FINAL.csv")
sample_folder = os.path.join(data_dir, "raw_samples")
output_matrix = os.path.join(data_dir, "FINAL_CPM_MATRIX.csv")

temp_dir = os.path.join(data_dir, "individual_samples_tmp")

#individual processing with streaming to handle large files without RAM issues
if os.path.exists(map_path) and os.path.exists(sample_folder):
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Loading Meticulous Map into memory...")
    site_map = pl.read_csv(map_path).select(["site_index", "region_id"])

    sample_files = glob.glob(os.path.join(sample_folder, "*.txt"))
    total_files = len(sample_files)
    processed_temp_files = []

    for i, fpath in enumerate(sample_files):
        s_name = os.path.basename(fpath).replace(".txt", "")
        print(f"[{i+1}/{total_files}] High-throughput streaming for: {s_name}...")

        try:
            #Compute exact Library Size via lazy scanning to save RAM
            total_filtered_reads = (
                pl.scan_csv(fpath, separator='\t', has_header=False, with_column_names=lambda x: ["total", "idx", "p", "m"])
                .select(pl.col("total").sum())
                .collect()
                .item()
            )

            #Lazy join, regional aggregation, and Sqrt-CPM normalisation
            t_path = os.path.join(temp_dir, f"{s_name}.parquet")
            
            (
                pl.scan_csv(fpath, separator='\t', has_header=False, with_column_names=lambda x: ["total", "site_index", "plus", "minus"])
                .filter(pl.col("total") > 0)
                .join(site_map.lazy(), on="site_index")
                .group_by("region_id")
                .agg([
                    ((pl.col("plus") + pl.col("minus")).sum()).alias(s_name)
                ])
                .with_columns(
                    (pl.col(s_name) / total_filtered_reads * 1_000_000).sqrt()
                )
                .collect(streaming=True) #forces disk buffer scaling for massive sets
                .write_parquet(t_path)
            )
            
            processed_temp_files.append(t_path)
            gc.collect()

        except Exception as e:
            print(f"Error processing sample {s_name}: {e}")

    #final parallel merge of all individual sample profiles into one master matrix
    print("\nMerging individual sample Parquet profiles...")
    if processed_temp_files:
        lf_master = pl.scan_parquet(processed_temp_files[0])
        for f in processed_temp_files[1:]:
            lf_master = lf_master.join(pl.scan_parquet(f), on="region_id", how="outer", coalesce=True)

        print("Writing finalized expression/intensity matrix...")
        lf_master.collect(streaming=True).write_csv(output_matrix)

    #clean up file architecture traces
    for f in processed_temp_files: 
        try: os.remove(f)
        except: pass
    try: os.rmdir(temp_dir)
    except: pass

    print(f"\n Done. Matrix saved to: {output_matrix}")

else:
    print("Note: Public pipeline running in demonstration mode.")
    print("Raw sample sequences and primary tracking datasets are omitted to preserve patient privacy boundaries.")