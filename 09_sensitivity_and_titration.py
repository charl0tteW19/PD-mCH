"""
In Silico Sensitivity Sweep & Linear Titration Mixture Modeling

Description:
Executes high-sensitivity validation experiments to determine the quantitative 
limits of cell-free DNA (cfDNA) marker detection. This script tests the transferability 
of brain-derived cell signatures spiked into a dominant peripheral immune background.

Core Transformation Logic:
  - Inverse Transformation: Ingests variance-stabilised square-root Counts Per Million 
    (sqrt-CPM) features and squares them (matrix ** 2) to restore true linear 
    proportions before executing mixture maths.
  - Regularisation Layer: Adds a controlled Gaussian noise vector to simulated linear 
    profiles and applies a floor bounding mask (np.clip) to prevent mathematical 
    anomalies below zero reads.
  - Evaluation Metrics: Runs an incremental sweep across feature cluster densities (K) 
    and a linear dilution titration cascade (0% to 5%) to map assay signal lift.

Inputs:
  - data/FINAL_CLASSIFIER_INPUT_MASTER.csv
  - results/Top_Distinguishing_Regions_K500.csv

Outputs:
  - results/Figure_Sensitivity_Region_Sweep.png
  - results/Figure_Linear_Titration_Sweep.png
  - results/sensitivity_sweep_density_N.csv
  - results/sensitivity_sweep_density_O.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

#Set paths and hyperparameters for sensitivity analysis and titration experiments
data_dir = "data"
output_dir = "results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

INPUT_MASTER = os.path.join(data_dir, "FINAL_CLASSIFIER_INPUT_MASTER.csv")
K500_PATH    = os.path.join(output_dir, "Top_Distinguishing_Regions_K500.csv")

#Plot Asset Names
REGION_SWEEP_OUT = os.path.join(output_dir, "Figure_Sensitivity_Region_Sweep.png")
LINEAR_TITR_OUT  = os.path.join(output_dir, "Figure_Linear_Titration_Sweep.png")

#Hyperparameters
SPIKE_LEVEL   = 0.001   #0.1% brain signal spike fraction
NOISE_LEVEL   = 0.0001 
SWEEP_STEPS   = list(range(0, 510, 10)) 
FIXED_K       = 200     #Evaluation analysis point
MIXTURE_STEPS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05] # 0% to 5%

custom_palette = {'N': '#1f77b4', 'O': '#2ca02c'}

#Core functions for label parsing, sensitivity simulation, and mixture titration modeling

def parse_label(sample_name):
    #Map sample name patterns to class labels
    s = str(sample_name).upper()
    if "-DN_CWG" in s: return "DN"
    if "-N_CWG" in s:   return "N"
    if "-O_CWG" in s:   return "O"
    return "IM"

def simulate_sensitivity(X_df, y_labels, target_class, target_regions, spike_level):
    #In silico simulation of sensitivity by spiking target signal into immune background and measuring percent lift
    X_linear = X_df**2
    np.random.seed(42) 
    noise = np.random.normal(0, NOISE_LEVEL, X_linear.shape)
    X_linear = np.clip(X_linear + noise, 0, None)
    
    target_profile = X_linear.loc[y_labels == target_class].mean()
    immune_samples = X_linear.loc[y_labels == "IM"]
    
    c_scores = [samp[target_regions].sum() for _, samp in immune_samples.iterrows()]
    p_scores = [((samp * (1 - spike_level)) + (target_profile * spike_level))[target_regions].sum() 
                for _, samp in immune_samples.iterrows()]
    
    return np.array(c_scores), np.array(p_scores)

def simulate_mixture_at_k(X_df, y_labels, target_class, target_regions, spike_level):
    #Calculate percent lift for a given mixture fraction by spiking target signal into immune samples and measuring individual sample lifts
    X_linear = X_df**2
    np.random.seed(42)
    
    target_profile = X_linear.loc[y_labels == target_class].mean()
    immune_samples = X_linear.loc[y_labels == "IM"]
    
    c_scores = np.array([samp[target_regions].sum() for _, samp in immune_samples.iterrows()])
    p_scores = np.array([((samp * (1 - spike_level)) + (target_profile * spike_level))[target_regions].sum() 
                         for _, samp in immune_samples.iterrows()])
    
    return ((p_scores - c_scores) / (c_scores + 1e-10)) * 100

#Execution block to run sensitivity analysis and linear titration experiments, generate plots, and save results

if __name__ == "__main__":
    if os.path.exists(INPUT_MASTER) and os.path.exists(K500_PATH):
        print("Loading data for sensitivity analysis and titration experiments...")
        df = pd.read_csv(INPUT_MASTER)
        X_master = df.set_index('region_id').T
        y_all = np.array([parse_label(s) for s in X_master.index])
        
        k500_df = pd.read_csv(K500_PATH)

        #Experiment A: Region Density Sweep
        print("\n Running Experiment A: Sensitivity vs. Region Density Sweep...")
        plot_data = {"N": [], "O": []}
        dots_data = {"N": None, "O": None} 

        for target in ["N", "O"]:
            col_name = f"Top_{target}_Regions"
            target_regions_all = k500_df[col_name].dropna().tolist()
            
            sweep_results = []
            for k in SWEEP_STEPS:
                current_regions = target_regions_all[:min(k, len(target_regions_all))]
                
                if k == 0:
                    lift = 0.0
                else:
                    c, p = simulate_sensitivity(X_master, y_all, target, current_regions, SPIKE_LEVEL)
                    lift = ((p.mean() - c.mean()) / (c.mean() + 1e-10)) * 100
                    
                    #Capture raw spread at the fixed benchmark point
                    if k == FIXED_K:
                        dots_data[target] = ((p - c) / (c + 1e-10)) * 100
                
                sweep_results.append({"K": k, "Percent_Lift": lift})
                plot_data[target].append(lift)

            #Save metrics to backing sheet
            out_file = os.path.join(output_dir, f"sensitivity_sweep_density_{target}.csv")
            pd.DataFrame(sweep_results).to_csv(out_file, index=False)

        #Visualise Experiment A results
        print("Producing figure")
        plt.figure(figsize=(10, 6))
        for target in ["N", "O"]:
            color = custom_palette[target]
            label_str = 'Neurons' if target=='N' else 'Oligodendrocytes'
            
            #Plot continuous line trends
            plt.plot(SWEEP_STEPS, plot_data[target], 
                     color=color, marker='o', markersize=4, lw=2, alpha=0.8,
                     label=f"{label_str} Trend")

            #Plot individual sample points at K=200 benchmark
            if dots_data[target] is not None:
                x_vals = [FIXED_K] * len(dots_data[target])
                plt.scatter(x_vals, dots_data[target], color=color, s=15, alpha=0.3, zorder=3)
                
                #Plot central distribution mean marker
                mean_val_fixed = plot_data[target][SWEEP_STEPS.index(FIXED_K)]
                plt.scatter(FIXED_K, mean_val_fixed, color=color, s=80, edgecolors='black', linewidths=1.5, zorder=5)

        plt.title("Sensitivity Analysis: Signal Lift vs. Number of Regions (K)", fontsize=14)
        plt.xlabel("Number of Regions (K)", fontsize=12)
        plt.ylabel("Percent Lift over Immune Background (%)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.axvline(x=FIXED_K, color='black', linestyle='-', alpha=0.4, lw=1.5, label=f'K={FIXED_K} Analysis Point')
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()
        plt.savefig(REGION_SWEEP_OUT, dpi=600)
        plt.close()

        #Experiment B: Linear Titration Sweep
        print(f"\n Running Linear Mixture Fraction Sweep at K={FIXED_K}")
        plt.figure(figsize=(9, 6))

        for target in ["N", "O"]:
            col_name = f"Top_{target}_Regions"
            target_regions = k500_df[col_name].dropna().tolist()[:FIXED_K]
            
            means, stds = [], []
            for f in MIXTURE_STEPS:
                individual_lifts = simulate_mixture_at_k(X_master, y_all, target, target_regions, f)
                means.append(individual_lifts.mean())
                stds.append(individual_lifts.std())

            color = custom_palette[target]
            label_text = 'Neurons' if target=='N' else 'Oligodendrocytes'
            x_pct = [f * 100 for f in MIXTURE_STEPS]

            #Plot Assay Trend Line with Standard Deviation bounds
            plt.errorbar(x_pct, means, yerr=stds, fmt='-o', color=color, 
                         ecolor=color, elinewidth=1.5, capsize=4, lw=2.5, markersize=7,
                         label=f"{label_text} (Mean ± SD)")

        #Visualise experiment B
        print("Producing figure")
        plt.title(f"Classifier Sensitivity: Linearity of Signal Detection (K={FIXED_K})", fontsize=14)
        plt.xlabel("Target Cell Mixture Fraction (%)", fontsize=12)
        plt.ylabel("Percent Lift over Immune Background (%)", fontsize=12)
        
        plt.xscale('linear')
        plt.xlim(-0.2, 5.2)
        plt.xticks([0, 1, 2, 3, 4, 5], ['0%', '1%', '2%', '3%', '4%', '5%'])
        plt.ylim(-1, None)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.4, lw=1)
        plt.legend(loc='upper left', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(LINEAR_TITR_OUT, dpi=600)
        plt.close()
        
        print(f"Figures exported safely to '{output_dir}'.")
        
    else:
        print("Note: Required file infrastructure (Master Matrix or Top Regions file) is missing.")