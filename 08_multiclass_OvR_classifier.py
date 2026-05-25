"""
Multiclass One-vs-Rest (OvR) mCH Cell-Type Classifier Pipeline

Description:
Implements a multiclass classification framework using an OvR logistic regression 
model to differentiate between brain-derived cohorts (DN, N, O) and peripheral immune 
references (IM). Employs Leave-One-Donor-Out (LODO) cross-validation and rigorous 
group-aware label permutation testing to validate predictive performance.

Inputs:
  - data/FINAL_CLASSIFIER_INPUT_MASTER.csv

Outputs:
  - results/Classifier_Bundle_K500.joblib
  - results/Top_Distinguishing_Regions_K500.csv
  - results/Multiclass_ROC_Curve.png
  - results/Multiclass_PR_Curve.png
  - results/Confusion_Matrix_Normalised.png
  - results/Permutation_Histogram.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import re

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, 
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)
from sklearn.multiclass import OneVsRestClassifier

#Configuration & Paths
data_dir = "data"
output_dir = "results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

INPUT_MASTER = os.path.join(data_dir, "FINAL_CLASSIFIER_INPUT_MASTER.csv")

#Experiment Parameters
ALL_CLASSES     = np.array(["DN", "IM", "N", "O"])
K_PER_CLASS     = 500  
N_PERMUTATIONS  = 300 

#Signal-to-Maximum-Noise Regularisation Constant (Section 4 Methods)
EPSILON         = 1e-6

#Palette Definition
# IM: Purple, N: Blue, O: Green, DN: Pink
custom_palette  = {'IM': '#9467bd', 'N': '#1f77b4', 'O': '#2ca02c', 'DN': '#e377c2'}

#Output Asset Mapping
MODEL_OUT   = os.path.join(output_dir, f"Classifier_Bundle_K{K_PER_CLASS}.joblib")
REGIONS_OUT = os.path.join(output_dir, f"Top_Distinguishing_Regions_K{K_PER_CLASS}.csv")
FIG_OUT1  = os.path.join(output_dir, "Multiclass_ROC_Curve.png")
FIG_OUT2  = os.path.join(output_dir, "Multiclass_PR_Curve.png")
FIG_OUT3  = os.path.join(output_dir, "Confusion_Matrix_Normalised.png")
FIG_OUT4  = os.path.join(output_dir, "Permutation_Histogram.png")

#Labelling and Grouping Functions

def parse_label(sample_name):
    #Maps sample names to class labels based on specific substrings, defaulting to 'IM' for unmatched patterns.
    s = str(sample_name).upper()
    if "-DN_CWG" in s: return "DN"
    if "-N_CWG" in s:   return "N"
    if "-O_CWG" in s:   return "O"
    return "IM"

def get_groups(sample_names):
    #Derives group identifiers from sample names, treating immune samples as independent groups.
    groups = []
    for name in sample_names:
        s = str(name).upper()
        match = re.match(r'^(\d+)-', s)
        if match:
            groups.append(match.group(1))
        else:
            groups.append(s) # Immune cells act as independent donors
    return np.array(groups)

def select_top_k_by_ovr_ratio(X_train_np, y_train, k_per_class=500):
    #Isolates top features for each class based on the ratio of class-specific means to the maximum mean of other classes, ensuring a balanced feature selection across all classes.
    unique_classes = np.unique(y_train)
    all_local_indices = []
    class_means = {cls: X_train_np[y_train == cls].mean(axis=0) for cls in unique_classes}

    for target_cls in unique_classes:
        target_mean = class_means[target_cls]
        others = [class_means[c] for c in class_means if c != target_cls]
        if not others: continue 
            
        max_other_mean = np.maximum.reduce(others)
        ratio = np.nan_to_num(target_mean / (max_other_mean + EPSILON))
        
        top_local = np.argsort(-ratio)[:min(k_per_class, len(ratio))].astype(int)
        all_local_indices.extend(top_local)

    return np.unique(all_local_indices).astype(int)

#Engine executing Leave-One-Group-Out cross-validation with integrated feature selection and performance scoring.

def run_lodo_cv(X_np, y, groups, pipeline, k_pc):
    #Executes Leave-One-Group-Out cross-validation, applying feature selection within each training fold and capturing both predicted classes and probability scores for comprehensive performance evaluation.
    logo = LeaveOneGroupOut()
    y_pred = np.empty_like(y)
    y_scores = np.zeros((len(y), len(ALL_CLASSES)))

    for tr_idx, te_idx in logo.split(X_np, y, groups):
        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr = y[tr_idx]
        
        idx = select_top_k_by_ovr_ratio(X_tr, y_tr, k_per_class=k_pc)
        pipeline.fit(X_tr[:, idx], y_tr)
        
        y_pred[te_idx] = pipeline.predict(X_te[:, idx])
        probs = pipeline.predict_proba(X_te[:, idx])
        
        for j, cls in enumerate(pipeline.named_steps['clf'].classes_):
            col = np.where(ALL_CLASSES == cls)[0][0]
            y_scores[te_idx, col] = probs[:, j]
            
    return y_pred, y_scores

#Pipeline and plotting execution, including permutation testing for empirical significance estimation of observed classifier performance.

if __name__ == "__main__":
    if os.path.exists(INPUT_MASTER):
        print("Step 1: Loading Master Matrix and Preparing Data Structures...")
        df = pd.read_csv(INPUT_MASTER)
        X_raw = df.set_index('region_id').T  
        
        sample_names = X_raw.index.tolist()
        y_all = np.array([parse_label(s) for s in sample_names])
        groups_all = get_groups(sample_names)
        X_np = X_raw.values.astype(np.float32)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=5000, C=1.0)))
        ])

        print("\n Step 2: Running Leave-One-Group-Out Cross-Validation to Obtain Predictions and Scores...")
        y_pred, y_scores = run_lodo_cv(X_np, y_all, groups_all, pipeline, K_PER_CLASS)
        Y_bin = label_binarize(y_all, classes=ALL_CLASSES)
        observed_macro_auc = roc_auc_score(Y_bin, y_scores, average='macro')

        #Figure: Receiver Operating Characteristic (ROC) Curves
        print("Plotting ROC Curves")
        plt.figure(figsize=(8, 8))
        for i, cls in enumerate(ALL_CLASSES):
            fpr, tpr, _ = roc_curve(Y_bin[:, i], y_scores[:, i])
            auc_val = roc_auc_score(Y_bin[:, i], y_scores[:, i])
            plt.plot(fpr, tpr, color=custom_palette[cls], lw=2, label=f"{cls} (AUC = {auc_val:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Classifier (AUC = 0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Multiclass ROC Curves (LODO CV)\nObserved Macro-AUC: {observed_macro_auc:.3f}")
        plt.legend(loc="lower right")
        plt.savefig(FIG_OUT1, dpi=600)
        plt.close()

        #Precision-Recall Curves
        print("Plotting Precision-Recall Curves")
        plt.figure(figsize=(8, 8))
        for i, cls in enumerate(ALL_CLASSES):
            prec, rec, _ = precision_recall_curve(Y_bin[:, i], y_scores[:, i])
            ap_val = average_precision_score(Y_bin[:, i], y_scores[:, i])
            plt.plot(rec, prec, color=custom_palette[cls], lw=2, label=f"{cls} (AP = {ap_val:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Multiclass Precision-Recall Curves")
        plt.legend(loc="lower left")
        plt.savefig(FIG_OUT2, dpi=600)
        plt.close()

        #Normalised Confusion Matrix
        print("Plotting Normalised Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 7))
        cm = confusion_matrix(y_all, y_pred, labels=ALL_CLASSES, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ALL_CLASSES)
        disp.plot(cmap='Blues', ax=ax, values_format='.2f')
        plt.title("Normalised Confusion Matrix (Recall Proportions)")
        plt.savefig(FIG_OUT3, dpi=600)
        plt.close()

        #300 Group-Aware Label Permutation
        print(f"\nExecuting {N_PERMUTATIONS} Group-Aware Permutations")
        perm_macro_aucs = []
        
        #Capture unique donor to class mapping for group-aware permuting
        unique_groups = np.unique(groups_all)
        group_to_class = {grp: y_all[groups_all == grp][0] for grp in unique_groups}
        
        np.random.seed(42)  #Secure reproducibility seed
        
        for p in range(1, N_PERMUTATIONS + 1):
            if p % 50 == 0:
                print(f"Permutation Flight {p}/{N_PERMUTATIONS}...")
                
            #Permute labels at the DONOR/GROUP level to preserve sample structures
            shuffled_classes = list(group_to_class.values())
            np.random.shuffle(shuffled_classes)
            shuffled_group_map = dict(zip(unique_groups, shuffled_classes))
            
            y_permuted = np.array([shuffled_group_map[grp] for grp in groups_all])
            
            #Evaluate permuted layout performance
            _, y_scores_perm = run_lodo_cv(X_np, y_permuted, groups_all, pipeline, K_PER_CLASS)
            Y_bin_perm = label_binarize(y_permuted, classes=ALL_CLASSES)
            perm_macro_aucs.append(roc_auc_score(Y_bin_perm, y_scores_perm, average='macro'))

        #Calculate empirical p-value based on upper tail distribution
        perm_macro_aucs = np.array(perm_macro_aucs)
        empirical_p = np.sum(perm_macro_aucs >= observed_macro_auc) / N_PERMUTATIONS

        print(f"Permutation Analysis Complete: Observed={observed_macro_auc:.3f}, Empirical p={empirical_p:.4f}")

        #Plotting the Null Distribution Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(perm_macro_aucs, bins=25, color='gainsboro', edgecolor='silver', alpha=0.8, label='Permuted Null Distribution')
        plt.axvline(x=observed_macro_auc, color='red', linestyle='--', lw=2, label=f'Observed Macro-AUC ({observed_macro_auc:.3f})')
        plt.title(f"Permutation Framework Validation\n(Macro-AUC Distribution vs Chance, p = {empirical_p:.4f})")
        plt.xlabel("Macro-AUC Value")
        plt.ylabel("Frequency Count")
        plt.legend(loc="upper left")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(FIG_OUT4, dpi=600)
        plt.close()

        #Save final model and top features for interpretability and future validation
        print("\n Exporting Final Model and Top Distinguishing Regions...")
        final_idx = select_top_k_by_ovr_ratio(X_np, y_all, k_per_class=K_PER_CLASS)
        pipeline.fit(X_np[:, final_idx], y_all)
        
        joblib.dump({"selected_regions": X_raw.columns[final_idx].tolist(), "pipeline": pipeline}, MODEL_OUT)

        marker_dict = {}
        for target_cls in np.unique(y_all):
            t_mean = X_np[y_all == target_cls].mean(axis=0)
            o_means = [X_np[y_all == c].mean(axis=0) for c in np.unique(y_all) if c != target_cls]
            if o_means:
                max_o = np.maximum.reduce(o_means)
                ratio = np.nan_to_num(t_mean / (max_o + EPSILON))
                top_indices = np.argsort(-ratio)[:K_PER_CLASS].astype(int)
                marker_dict[f"Top_{target_cls}_Regions"] = X_raw.columns[top_indices].tolist()
        
        pd.DataFrame(marker_dict).to_csv(REGIONS_OUT, index=False)
        print(f"Figures saved in '{output_dir}'.")
        
    else:
        print("Note: Master reference file missing. Script execution suspended.")