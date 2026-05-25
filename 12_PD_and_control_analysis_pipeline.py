"""
Unified Clinical Evaluation & Machine Learning Analytics Pipeline

Description:
A reproducible execution framework for evaluating cell-free DNA (cfDNA) 
methylation alterations in Parkinson's Disease. This script executes end-to-end 
downstream statistics including:
  - Feature selection using targeted lineage marker regions.
  - Cross-validated Random Forest classification modelling (GroupKFold).
  - Row-wise epigenetic noise quantification.
  - Cell-type ratio tracking (Neuronal-to-Oligodendrocyte fraction via raw sums).
  - Cross-sectional baseline diagnostic profiling.
  - Normality-conditional longitudinal trajectory analysis over a 2-year window.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import ttest_ind, ttest_rel, levene, shapiro, mannwhitneyu, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

#Path configurations 
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH      = os.path.join(BASE_DIR, "..", "data", "MASTER_MATRIX_FILTERED_FINAL.csv")
MARKER_LIST_PATH = os.path.join(BASE_DIR, "..", "results", "Top_Distinguishing_Regions_K500.csv")
OUTPUT_DIR      = os.path.join(BASE_DIR, "..", "results", "pipeline_outputs")

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

#Colour palette
COLOUR_CTRL = "teal"
COLOUR_PD   = "plum"
CUSTOM_PALETTE = {'Control': COLOUR_CTRL, 'PD': COLOUR_PD}

def run_comprehensive_analysis_suite():
    print("=" * 70)
    print("Initiating analysis")
    print("=" * 70)
    
    # Static Validation Guardrail
    if not os.path.exists(INPUT_PATH) or not os.path.exists(MARKER_LIST_PATH):
        print("Execution Error: Prerequisite data assets are missing from relative directories.")
        print("Please verify presence of 'data/MASTER_MATRIX_FILTERED_FINAL.csv' and 'results/Top_Distinguishing_Regions_K500.csv'.")
        sys.exit(1)
        
    df = pd.read_csv(INPUT_PATH, index_col=0)
    markers_df = pd.read_csv(MARKER_LIST_PATH)
    
    #Feature isolation based on top distinguishing regions for Neuronal and Oligodendrocyte lineages
    pot_n = markers_df['Top_N_Regions'].dropna().head(200).tolist()
    pot_o = markers_df['Top_O_Regions'].dropna().head(200).tolist()
    X_raw = df.drop(columns=['Actual_Group', 'Patient_ID', 'Timepoint'])
    
    def apply_sparsity_filter(marker_list, data_matrix):
        #only retain features with less than 90% zero values across all samples 
        valid_subset = [m for m in marker_list if m in data_matrix.columns]
        zero_proportions = (data_matrix[valid_subset] == 0).mean()
        return zero_proportions[zero_proportions < 0.90].index.tolist()

    clean_n = apply_sparsity_filter(pot_n, X_raw)
    clean_o = apply_sparsity_filter(pot_o, X_raw)
    all_markers = clean_n + clean_o
    
    print(f"STATUS: Passing Features -> Neuronal: {len(clean_n)} | Oligodendrocyte: {len(clean_o)}")

    
    analysis_df = df[['Actual_Group', 'Patient_ID', 'Timepoint']].copy()
    X_ctrl_baseline = X_raw.loc[df['Actual_Group'] == 'Control']
    
    #Generative Component Z-Scores
    for label, marker_set in [("Neuronal", clean_n), ("Oligo", clean_o)]:
        mean_ref = X_ctrl_baseline[marker_set].mean()
        std_ref  = X_ctrl_baseline[marker_set].std().replace(0, 1)
        analysis_df[f"{label}_Z"] = ((X_raw[marker_set] - mean_ref) / std_ref).mean(axis=1)

    #Establish Raw Aggregated Profile Matrix Columns
    analysis_df['Neuronal_Raw_Sum'] = X_raw[clean_n].mean(axis=1)
    analysis_df['Oligo_Raw_Sum']    = X_raw[clean_o].mean(axis=1)

    #Compute Structural Properties (Noise and Cellular Balances)
    analysis_df['Epigenetic_Noise'] = X_raw[all_markers].std(axis=1)
    analysis_df['N_O_Ratio']        = analysis_df['Neuronal_Raw_Sum'] / analysis_df['Oligo_Raw_Sum'].replace(0, 1)
    
    #Filter for Cross-Sectional Baselines
    baseline_mask = analysis_df['Timepoint'] != '2-Year'
    baseline_data = analysis_df[baseline_mask].copy()

    #Machine Learning Classifier Construction & Validation (Random Forest with GroupKFold)
    binary_labels = np.where(df['Actual_Group'] == "PD", 1, 0)
    classifier = RandomForestClassifier(n_estimators=1000, max_depth=8, random_state=42, class_weight='balanced_subsample')
    
    #Cross-validation using strict patient groupings to prevent identity leakage
    predicted_probabilities = cross_val_predict(
        classifier, X_raw[all_markers], binary_labels, 
        cv=GroupKFold(n_splits=5), groups=df['Patient_ID'], method="predict_proba"
    )[:, 1]
    
    model_auc = roc_auc_score(binary_labels, predicted_probabilities)

    #Cross-Sectional Baseline Diagnostic Profiling (Neuronal Z-Score Focus)
    ctrl_neuronal_z = analysis_df.loc[analysis_df['Actual_Group'] == 'Control', 'Neuronal_Z']
    pd_base_neuronal_z = analysis_df.loc[(analysis_df['Actual_Group'] == 'PD') & (analysis_df['Timepoint'] == 'Baseline'), 'Neuronal_Z']
    
    #Diagnostic Normality and Shift Tests
    _, p_norm_ctrl = shapiro(ctrl_neuronal_z)
    _, p_norm_base = shapiro(pd_base_neuronal_z)
    t_diag, p_diag = ttest_ind(pd_base_neuronal_z, ctrl_neuronal_z)
    _, p_mw = mannwhitneyu(pd_base_neuronal_z, ctrl_neuronal_z)
    cohens_d = (pd_base_neuronal_z.mean() - ctrl_neuronal_z.mean()) / np.sqrt((pd_base_neuronal_z.std()**2 + ctrl_neuronal_z.std()**2) / 2)
    
    #Longitudinal Changes (Neuronal Z Tracking)
    paired_neuronal = analysis_df[analysis_df['Actual_Group'] == 'PD'].pivot(index='Patient_ID', columns='Timepoint', values='Neuronal_Z').dropna()
    t_long_neuronal, p_long_neuronal = ttest_rel(paired_neuronal['2-Year'], paired_neuronal['Baseline'])

    #Baseline Epigenetic Noise Comparisons
    noise_ctrl = baseline_data.loc[baseline_data['Actual_Group'] == 'Control', 'Epigenetic_Noise']
    noise_pd   = baseline_data.loc[baseline_data['Actual_Group'] == 'PD', 'Epigenetic_Noise']
    t_noise, p_noise = ttest_ind(noise_pd, noise_ctrl)
    levene_noise_stat, levene_noise_p = levene(noise_ctrl, noise_pd)

    #Baseline N/O Ratio Spatial Metrics
    ratio_ctrl = baseline_data.loc[baseline_data['Actual_Group'] == 'Control', 'N_O_Ratio']
    ratio_pd   = baseline_data.loc[baseline_data['Actual_Group'] == 'PD', 'N_O_Ratio']
    t_ratio, p_ratio = ttest_ind(ratio_pd, ratio_ctrl)
    levene_ratio_stat, levene_ratio_p = levene(ratio_ctrl, ratio_pd)

    #Isolated Component Evaluation (Numerator/Denominator Raw Tracks)
    ctrl_neuronal_raw_mean = baseline_data.loc[baseline_data['Actual_Group'] == 'Control', 'Neuronal_Raw_Sum']
    pd_neuronal_raw_mean   = baseline_data.loc[baseline_data['Actual_Group'] == 'PD', 'Neuronal_Raw_Sum']
    ctrl_oligo_raw_mean    = baseline_data.loc[baseline_data['Actual_Group'] == 'Control', 'Oligo_Raw_Sum']
    pd_oligo_raw_mean      = baseline_data.loc[baseline_data['Actual_Group'] == 'PD', 'Oligo_Raw_Sum']
    
    _, p_comp_num = ttest_ind(ctrl_neuronal_raw_mean, pd_neuronal_raw_mean)
    _, p_comp_den = ttest_ind(ctrl_oligo_raw_mean, pd_oligo_raw_mean)

    #Longitudinal Shifts for Cellular Balance Ratio (N/O Tracking Loops)
    pd_longitudinal = analysis_df[analysis_df['Actual_Group'] == 'PD'].copy().reset_index()
    possible_patient_keys = ['Patient_ID', 'patient_id', 'PATIENT_ID']
    patient_key = next((k for k in possible_patient_keys if k in pd_longitudinal.columns), pd_longitudinal.columns[0])
    
    paired_ratio = pd_longitudinal.pivot(index=patient_key, columns='Timepoint', values='N_O_Ratio').dropna()
    paired_ratio['Delta'] = paired_ratio['2-Year'] - paired_ratio['Baseline']
    
    #Assess rate-of-change normality to dictate statistical test model
    _, p_norm_delta_ratio = shapiro(paired_ratio['Delta'])
    
    if p_norm_delta_ratio > 0.05:
        t_long_ratio, p_long_ratio = ttest_rel(paired_ratio['Baseline'], paired_ratio['2-Year'])
        ratio_test_engine = f"Parametric Paired T-Test (t = {t_long_ratio:.4f}, p = {p_long_ratio:.6f})"
    else:
        w_long_ratio, p_long_ratio = wilcoxon(paired_ratio['Baseline'], paired_ratio['2-Year'])
        ratio_test_engine = f"Non-Parametric Wilcoxon Signed-Rank (W = {w_long_ratio:.4f}, p = {p_long_ratio:.6f})"

    #Stats report 
    print("\n" + "="*65)
    print("COMPREHENSIVE ANALYSIS SUMMARY REPORT")
    print("="*65)
    print(f"Classification Performance\n  Cross-Validated Model Area Under ROC Curve (AUC): {model_auc:.4f}")
    print("-" * 65)
    print(f"Cross-Sectional Neuronal Burden (Control vs Baseline):\n  T-Test Assessment p-value: {p_diag:.6f} (M-W U p-value: {p_mw:.6f})\n  Calculated Cohen's d Effect Size: {cohens_d:.4f}")
    print(f"  Normality Tracking Profile -> Controls: p={p_norm_ctrl:.4f} | Baseline Cases: p={p_norm_base:.4f}")
    print("-" * 65)
    print(f"Cellular Component Noise Variations:\n  Noise Shift T-Test p-value: {p_noise:.6f}\n  Levene Variance Homogeneity Check: W = {levene_noise_stat:.4f} (p = {levene_noise_p:.6f})")
    print("-" * 65)
    print(f"Cellular fraction assessments (N/O Metrics):\n  N/O Ratio Difference T-Test p-value: {p_ratio:.6f}")
    print(f"  Levene Ratio Variance Check: W = {levene_ratio_stat:.4f} (p = {levene_ratio_p:.6f})")
    print(f"  Component Contribution p-values -> Neuronal Numerator (Raw Sum): {p_comp_num:.4f} | Oligo Denominator (Raw Sum): {p_comp_den:.4f}")
    print("-" * 65)
    print(f"Longitudinal Outcomes (Baseline vs 2-Year Progressions):\n  Neuronal Coordinate Trajectories Paired T-Test p-value: {p_long_neuronal:.4f}")
    print(f"  N/O Cell-Type Balance Path Evaluation: {ratio_test_engine}")
    print(f"  Delta Profile Distribution Normality (Shapiro-Wilk): p = {p_norm_delta_ratio:.6f}")
    print("="*65 + "\n")

    #Plotting Configurations
    sns.set_style("white")

    #Population Signal Densities (Neuronal Z Distributions)
    plt.figure(figsize=(7, 5))
    sns.kdeplot(ctrl_neuronal_z, label='Healthy Controls', fill=True, colour=COLOUR_CTRL, alpha=0.35)
    sns.kdeplot(pd_base_neuronal_z, label='PD Baseline', fill=True, colour=COLOUR_PD, alpha=0.35)
    plt.title("Neuronal Z-Score Density Overlaps", fontweight='bold', pad=12)
    plt.xlabel("Neuronal Z-Score")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "signal_density_distribution.png"), dpi=300)
    plt.close()

    #Receiver Operating Characteristic (ROC Curve)
    fpr, tpr, _ = roc_curve(binary_labels, predicted_probabilities)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, colour='#8E4585', lw=2.5, label=f'Classifier Performance (AUC = {model_auc:.2f})')
    plt.plot([0, 1], [0, 1], colour='darkgray', linestyle='--', lw=1.2)
    plt.title('Diagnostic Model ROC Curve', fontweight='bold', pad=12)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_roc_curve.png"), dpi=300)
    plt.close()

    #Individual Clinical Progressions (Neuronal Spaghetti)
    plt.figure(figsize=(7, 6))
    pd_only = analysis_df[analysis_df['Actual_Group'] == 'PD'].copy()
    sns.lineplot(data=pd_only, x='Timepoint', y='Neuronal_Z', hue='Patient_ID', palette='binary', alpha=0.15, legend=False, linewidth=1)
    sns.pointplot(data=pd_only, x='Timepoint', y='Neuronal_Z', colour='black', scale=1.1, errwidth=2, capsize=.05)
    plt.title("Longitudinal Neuronal Burden Trajectories", fontweight='bold', pad=12)
    plt.xlabel("Evaluation Window")
    plt.ylabel("Neuronal Z-Score")
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "longitudinal_trajectories_neuronal.png"), dpi=300)
    plt.close()

    #Epigenetic Noise & N/O Ratio Side-by-Side Spatial Comparisons
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    #Sub-plot 1: Row Noise Properties
    sns.boxplot(data=baseline_data, x='Actual_Group', y='Epigenetic_Noise', palette=CUSTOM_PALETTE, ax=axes[0], width=0.45, showfliers=False)
    sns.stripplot(data=baseline_data, x='Actual_Group', y='Epigenetic_Noise', colour='black', alpha=0.25, size=4, ax=axes[0], jitter=0.15)
    axes[0].set_title("Row-Wise Epigenetic Noise Metrics", fontweight='bold', pad=10)
    axes[0].set_xlabel("Cohort")
    axes[0].set_ylabel("Standard Deviation Value")
    #Sub-plot 2: Cell Ratios
    sns.boxplot(data=baseline_data, x='Actual_Group', y='N_O_Ratio', palette=CUSTOM_PALETTE, ax=axes[1], width=0.45, showfliers=False)
    sns.stripplot(data=baseline_data, x='Actual_Group', y='N_O_Ratio', colour='black', alpha=0.25, size=4, ax=axes[1], jitter=0.15)
    axes[1].set_title("Neuronal-To-Oligodendrocyte Balance Ratios", fontweight='bold', pad=10)
    axes[1].set_xlabel("Cohort")
    axes[1].set_ylabel("N/O Ratio Value")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "noise_and_fraction_contrasts.png"), dpi=300)
    plt.close()

    #Cross-Sectional Baseline Oligodendrocyte Signal Variances (Using Raw Aggregated Sum)
    plt.figure(figsize=(6, 5.5))
    sns.boxplot(data=baseline_data, x='Actual_Group', y='Oligo_Raw_Sum', palette=CUSTOM_PALETTE, width=0.4, showfliers=False)
    sns.stripplot(data=baseline_data, x='Actual_Group', y='Oligo_Raw_Sum', colour='black', alpha=0.25, size=4, jitter=0.15)
    plt.title(f"Baseline Oligodendrocyte mCH Intensity Shifts\n(p = {p_comp_den:.4f})", fontweight='bold', pad=12)
    plt.xlabel("Cohort")
    plt.ylabel("Aggregated Oligodendrocyte Methylation Intensity")
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_oligodendrocyte_burden.png"), dpi=300)
    plt.close()

    #Cell-Type Balance Path Monitoring (N/O Trajectory Spaghetti + Change Distribution)
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [2, 1]})
    time_series_points = ['Baseline', '2-Year']
    #Spaghetti Paths
    for _, path_row in paired_ratio.iterrows():
        axes3[0].plot(time_series_points, [path_row['Baseline'], path_row['2-Year']], colour='plum', alpha=0.20, linewidth=1.2)
        axes3[0].scatter(time_series_points, [path_row['Baseline'], path_row['2-Year']], colour='purple', alpha=0.25, s=15)
    #Master Group Mean Overlay Vector
    axes3[0].plot(time_series_points, [paired_ratio['Baseline'].mean(), paired_ratio['2-Year'].mean()], colour='black', marker='o', linewidth=3, markersize=8, label='Cohort Mean Track')
    axes3[0].set_title("Longitudinal Balance Paths (N/O Ratio Over Time)", fontweight='bold', pad=10)
    axes3[0].set_ylabel("Neuronal-to-Oligodendrocyte Ratio")
    axes3[0].grid(axis='y', linestyle='--', alpha=0.4)
    #Delta Distribution Vector Boxplot
    sns.boxplot(data=paired_ratio, y='Delta', colour='plum', width=0.35, ax=axes3[1], showfliers=False)
    sns.stripplot(data=paired_ratio, y='Delta', colour='black', alpha=0.25, size=4, jitter=0.12, ax=axes3[1])
    axes3[1].axhline(0, colour='red', linestyle='--', alpha=0.6, linewidth=1.2)
    axes3[1].set_title("Distribution of Delta Changes", fontweight='bold', pad=10)
    axes3[1].set_ylabel("Rate of Change ($\Delta$ N/O Ratio)")
    axes3[1].set_xticklabels([])
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "longitudinal_cell_type_balance.png"), dpi=300)
    plt.close()

    #export
    analysis_df.to_csv(os.path.join(OUTPUT_DIR, "calculated_study_metrics.csv"))
    print(f"Pipeline executions successfully completed.")
    print(f"Metrics exports saved to: \n -> {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    run_comprehensive_analysis_suite()