# survival_analysis.py

import os

from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
def filter_genes_in_var_names(adata, gene_list):
    """
    保留 gene_list 中在 adata.var_names 中存在的基因。

    参数：
    adata : AnnData
        包含基因信息的 AnnData 对象。
    gene_list : list
        要筛选的基因列表。

    返回：
    list
        仅包含在 adata.var_names 中存在的基因的列表。
    """
    filtered_gene_list = [gene for gene in gene_list if gene in adata.var_names]
    print(f"Filtered {len(filtered_gene_list)} genes from {len(gene_list)} genes")
    return filtered_gene_list

def save_filtered_genes(filtered_gene_list_dict, save_folder):
    # 找到最长的基因列表长度
    max_len = max(len(gene_list) for gene_list in filtered_gene_list_dict.values())
    
    # 对齐每个基因列表的长度
    aligned_dict = {state: gene_list + [None] * (max_len - len(gene_list)) for state, gene_list in filtered_gene_list_dict.items()}
    
    df = pd.DataFrame(aligned_dict)
    df.to_csv(os.path.join(save_folder, "filtered_genes_in_gene_scores.csv"), index=False)

def get_gene_scores_to_adata(bulk_survival_adata, gene_list_dict):
    print(f"Start getting gene scores to adata")
    filtered_gene_list_dict = {}
    for state, gene_list in gene_list_dict.items():
        filtered_gene_list = filter_genes_in_var_names(bulk_survival_adata, gene_list)
        filtered_gene_list_dict[state] = filtered_gene_list
        sc.tl.score_genes(bulk_survival_adata, filtered_gene_list, score_name=f'gene_score_{state}_raw')
        bulk_survival_adata.obs[f'gene_score_{state}'] = StandardScaler().fit_transform(bulk_survival_adata.obs[[f'gene_score_{state}_raw']])
    print(f"Done getting gene scores to adata")
    return bulk_survival_adata, filtered_gene_list_dict

def comparitive_standard_check(bulk_survival_adata):
    print(f"Start doing comparitive standard check for age")
    # Ensure 'age' is numeric
    bulk_survival_adata.obs['age'] = pd.to_numeric(bulk_survival_adata.obs['age'], errors='coerce')
    
    # Handle missing age values
    if bulk_survival_adata.obs['age'].isna().sum() > 0:
        # Option 1: Impute missing values
        median_age = bulk_survival_adata.obs['age'].median()
        bulk_survival_adata.obs['age'].fillna(median_age, inplace=True)
        # Option 2: Drop rows with missing 'age'
        # bulk_survival_adata = bulk_survival_adata[bulk_survival_adata.obs['age'].notna()]
    print(f"Done doing comparitive standard check for age")
    return bulk_survival_adata


def cox_hazard_analysis(bulk_survival_adata, gene_list_dict, save_folder,prefix=''):
    #bulk_survival_adata = get_gene_scores_to_adata(bulk_survival_adata, gene_list_dict)
    bulk_survival_adata = comparitive_standard_check(bulk_survival_adata)
    formula = ' + '.join([f'gene_score_{state}' for state in gene_list_dict.keys()])
    formula = formula + ' + age'
    print(f"formula: {formula}")
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(bulk_survival_adata.obs, duration_col='relapse_time', event_col='relapse_status', formula=formula)
    cph.plot()
    plt.title(f'Hazard Ratio for {prefix}')
    plt.savefig(os.path.join(save_folder, f"hazard_ratio_{prefix}_gene_score.png"))
    plt.close()

def kaplan_meier_analysis(bulk_survival_adata, score_col, cutoff, save_folder, prefix='Gene Score'):
    bulk_survival_adata.obs['risk_group'] = np.where(bulk_survival_adata.obs[score_col] > cutoff, 'High risk', 'Low risk')
    kmf = KaplanMeierFitter()
    
    plt.figure(figsize=(8, 6))
    for group in ['High risk', 'Low risk']:
        mask = bulk_survival_adata.obs['risk_group'] == group
        kmf.fit(bulk_survival_adata.obs.loc[mask, 'relapse_time'], event_observed=bulk_survival_adata.obs.loc[mask, 'relapse_status'], label=group)
        kmf.plot_survival_function()
    plt.title(f'Kaplan Meier Survival Curve Based on {prefix}')
    plt.xlabel('Time (Days)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_folder, f"survival_curve_{prefix}.png"))
    plt.close()

