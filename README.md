# PD-mCH
#Lineage-specific non-CpG methylation classification pipeline for PD cfDNA

This repository contains the data processing, quality control, and machine learning pipelines used to isolate cell-type specific non-CpG (mCH) methylation signatures from cell-free DNA (cfDNA).

---

#This repository contains code only. None of the actual sequencing matrices, patient cohort records, or raw clinical tracking arrays are uploaded here. 

The scripts are written to assume a local directory structure (detailed below).

---

#Local Directory Architecture
To run these scripts locally, your workspace should be set up as follows:

PD-mCH/
├── data/
│   ├── METICULOUS_MAP_FINAL.csv             # Coordinate lookup map
│   ├── raw_samples/                         # Folder for raw single-sample .txt traces
│   ├── BRAIN_MATRIX_VALIDATED_ONLY.csv      # Processed brain reference matrix
│   └── FINAL_IMMUNE_MATRIX_SQRT_CPM.csv     # Processed peripheral immune reference matrix
├── results/                                 # Automatically generated plots and log summaries
└── .gitignore
