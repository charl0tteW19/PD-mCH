"""
Exploratory Data Analysis & Statistical Validation Pipeline

Description:
Performs comprehensive exploratory data analysis and statistical validation on the 
cross-platform validated feature matrix. Generates the foundational manuscript figures 
(Figures 1A-1D) including PCA dimensional reduction with KDE contour boundaries, 
Kolmogorov-Smirnov (K-S) distribution tests, sample-to-sample correlation matrices, 
and highly discriminatory regional expression heatmaps. Output metrics are logged 
automatically to support formal thesis verification.

Inputs:
  - data/VALIDATED_MARKER_LIST_CPM.csv

Outputs:
  - results/thesis_statistical_summary.txt
  - results/Figure_1A_PCA_Clustering.png
  - results/Figure_1B_KDE_Distributions.png
  - results/Figure_1C_Sample_Correlation.png
  - results/Figure_1D_Directional_Heatmaps.png
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp
from skbio.stats.distance import permanova
from skbio import DistanceMatrix
from scipy.spatial.distance import pdist, squareform

#Colour palette for PCA and KDE plots (user-specified)
custom_palette = {'N': '#1f77b4', 'O': '#2ca02c', 'DN': '#e377c2'}
sns.set_theme(style="white")

data_dir = "data"
output_dir = "results"

input_path = os.path.join(data_dir, "VALIDATED_MARKER_LIST_CPM.csv")
stats_log_path = os.path.join(output_dir, "thesis_statistical_summary.txt")

if os.path.exists(input_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading validated cross-platform feature matrix...")
    df = pd.read_csv(input_path)

    metadata_cols = ['region_id', 'Direction', 'Cell_Type', 'Avg_Intensity', 'Is_Validated', 'Variance']
    sample_cols = [c for c in df.columns if c not in metadata_cols]

    #Transpose matrix for sample-level analysis (Rows = Samples)
    X = df[sample_cols].T
    y_cell = [s.split('-')[1].split('_')[0] if '-' in s else s for s in X.index]

    #Initialise log
    stats_file = open(stats_log_path, "w")
    stats_file.write("--- THESIS PROOF OF CONCEPT: STATISTICAL VALDIATION ---\n\n")

    #PCA and PERMANOVA for sample clustering validation
    print("Executing PCA and Permutational MANOVA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame(coords, columns=['PC1', 'PC2'])
    pca_df['Cell_Type'] = y_cell

    #Calculate distance metrics and run PERMANOVA
    dist_matrix = DistanceMatrix(squareform(pdist(X, metric='euclidean')), ids=X.index)
    p_results = permanova(dist_matrix, grouping=y_cell, permutations=999)
    stats_file.write(f"1. PCA CLUSTERING SEPARATION (PERMANOVA)\n   Pseudo-F: {p_results['test statistic']:.4f}\n   p-value: {p_results['p-value']:.4f}\n\n")

    #Plot PCA with user-specified palette and circular markers, adding KDE contours to highlight clustering logic
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cell_Type', s=150, palette=custom_palette, marker='o', edgecolor='white', linewidth=0.4, alpha=0.95)
    
    for cell_type in pca_df['Cell_Type'].unique():
        subset = pca_df[pca_df['Cell_Type'] == cell_type]
        if len(subset) > 1:
            sns.kdeplot(x=subset['PC1'], y=subset['PC2'], levels=[0.5, 1.0], alpha=0.25, fill=True, color=custom_palette.get(cell_type))
            
    plt.title(f"PCA Separation of Sorted Brain Cells (PERMANOVA p={p_results['p-value']:.4f})")
    plt.savefig(os.path.join(output_dir, 'Figure_PCA_Clustering.png'), dpi=600)
    plt.close()

    # --- 2. DISCRIMINATORY DISTRIBUTION SHIFT (K-S Test) ---
    print("Generating KDE plots")
    df_melted = df.melt(id_vars=['Direction'], value_vars=sample_cols, var_name='Sample', value_name='Intensity')
    df_melted['Cell_Type'] = df_melted['Sample'].apply(lambda s: s.split('-')[1].split('_')[0] if '-' in s else s)

    hyper_data = df_melted[df_melted['Direction'] == 'Hyper']['Intensity']
    hypo_data = df_melted[df_melted['Direction'] == 'Hypo']['Intensity']
    ks_stat, ks_p = ks_2samp(hyper_data, hypo_data)
    stats_file.write(f"2. BIOLOGICAL DIRECTIONALITY SHIFT (K-S TEST)\n   D-statistic: {ks_stat:.4f}\n   p-value: {ks_p:.4e}\n\n")

    #KDE plots to visualise the distributional shift between Hyper and Hypo groups across all samples, faceted by Direction and colored by Cell Type
    g = sns.FacetGrid(df_melted, col="Direction", hue="Cell_Type", height=5, aspect=1.2, palette=custom_palette)
    g.map(sns.kdeplot, "Intensity", fill=True, alpha=0.4)
    g.add_legend(title="Cell Type")
    g.set_axis_labels("Signal Intensity (Normalised Sqrt-CPM)", "Density")
    ks_title = f"MeD-seq Signal Consensus (K-S p < 0.0001)" if ks_p < 0.0001 else f"MeD-seq Signal Consensus (K-S p = {ks_p:.4f})"
    g.fig.suptitle(ks_title, y=1.05)
    plt.savefig(os.path.join(output_dir, 'Figure_KDE_Distributions.png'), dpi=600, bbox_inches='tight')
    plt.close()

    #Intra-group correlation analysis to establish a technical reproducibility baseline, replicating the sample correlation matrix from Figure 1C and calculating mean intra-group Pearson r values
    print("Assessing sample-to-sample correlation vectors...")
    corr_matrix = X.T.corr()
    intra_corr = []
    for cell_type in set(y_cell):
        indices = [i for i, val in enumerate(y_cell) if val == cell_type]
        if len(indices) > 1:
            sub_corr = corr_matrix.iloc[indices, indices]
            mask = np.triu(np.ones(sub_corr.shape), k=1).astype(bool)
            intra_corr.append(sub_corr.where(mask).stack().mean())

    mean_r = np.mean(intra_corr) if intra_corr else 0
    stats_file.write(f"3. TECHNICAL REPRODUCIBILITY BASELINE\n   Mean Intra-group Pearson r: {mean_r:.4f}\n\n")

    #Visualise the sample correlation matrix with a viridis color scheme, setting a fixed range to highlight strong correlations and replicating the aesthetics of Figure 1C
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap='viridis_r', vmin=0.5, vmax=1, square=True)
    plt.title(f"Intra-group Reproducibility Matrix (Mean r={mean_r:.3f})")
    plt.savefig(os.path.join(output_dir, 'Figure_Sample_Correlation.png'), dpi=600, bbox_inches='tight')
    plt.close()

    #Top discriinaory regions heatmaps 
    print("Generating biological directionality marker panels...")
    df['Variance'] = df[sample_cols].var(axis=1)
    top_hyper = df[df['Direction'] == 'Hyper'].nlargest(50, 'Variance')
    top_hypo = df[df['Direction'] == 'Hypo'].nlargest(50, 'Variance')

    #Plot heatmaps of the top 50 validated Hyper and Hypo markers, sorting columns by cell type to replicate the aesthetics of Figure 1D and using distinct color palettes for each directionality group
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    sns.heatmap(top_hyper[sample_cols], ax=ax1, cmap='magma', robust=True, yticklabels=False)
    ax1.set_title('Top 50 Validated HYPER-Methylated Regions')
    sns.heatmap(top_hypo[sample_cols], ax=ax2, cmap='viridis', robust=True, yticklabels=False)
    ax2.set_title('Top 50 Validated HYPO-Methylated Regions')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Figure_Directional_Heatmaps.png'), dpi=600)
    plt.close()

    stats_file.close()
    print(f"Metrics saved to: {stats_log_path}")

else:
    print("Note: Validated tracking index missing. Figure generation script running in headless evaluation template layout.")