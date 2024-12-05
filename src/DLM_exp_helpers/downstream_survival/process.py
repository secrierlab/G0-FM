# Get the directory of the current file
import os
import sys
src_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(src_loc)
print("⭐️ src_loc:", src_loc)


import argparse
import pandas as pd
import anndata as ad
import pickle

from src.DLM_exp_helpers.downstream_survival.tcga import preprocess_bulk_survival_data, load_bulk_survival_adata
from src.DLM_exp_helpers.downstream_survival.cox_and_km import cox_hazard_analysis, kaplan_meier_analysis, get_gene_scores_to_adata, save_filtered_genes

def main():
    parser = argparse.ArgumentParser(description="Survival Analysis on Bulk RNA Data with gene list")
    parser.add_argument('--preprocess_flag', type=bool, nargs='?', const=True, default=False, help="Enable or disable data preprocessing")
    parser.add_argument('--topk_genes_file', type=str, help="Path to the top-K genes file")
    parser.add_argument('--K', type=int, help="Number of top genes to consider")
    # load from adata
    parser.add_argument('--bulk_survival_adata', type=str,required=True, help="Path to save AnnData object")
    # load from csv and preprocess
    parser.add_argument('--data_file', type=str, help="Path to the data CSV file")
    parser.add_argument('--label_file', type=str, help="Path to the label CSV file")
    parser.add_argument('--write_adata', action='store_true', help="Flag to write AnnData object")
    
    # save plots and results
    parser.add_argument('--output_folder', type=str, help="Folder to save plots and results")


    
    args = parser.parse_args()
    
    if args.preprocess_flag:
        bulk_survival_adata = preprocess_bulk_survival_data(args.data_file, args.label_file, 
                                                            {'relapse_time':'Progress.Free.Survival..Months.',
                                                             'relapse_status':'Progression.Free.Status',
                                                             'age':'age_at_index'},
                                                            args.bulk_survival_adata)
    else:
        bulk_survival_adata = load_bulk_survival_adata(args.bulk_survival_adata)

    print(f"{bulk_survival_adata.obs.head(5)}")
    print(f"{bulk_survival_adata.var.head(5)}")
    print(f"there are {len(bulk_survival_adata.var_names)} genes in adata")

    # 从csv中读取基因列表，只要前K个
    top_genes_df = pd.read_csv(args.topk_genes_file)
    topK_genes_df = top_genes_df.head(args.K)
    state_list = topK_genes_df.columns.tolist()
    gene_list_dict = {}
    for state in state_list[1:]:
        print(f"Processing state: {state}")
        gene_list = topK_genes_df[state].tolist()
        gene_list_dict[state] = gene_list
    bulk_survival_adata, filtered_gene_list_dict = get_gene_scores_to_adata(bulk_survival_adata, gene_list_dict)
    cox_hazard_analysis(bulk_survival_adata, filtered_gene_list_dict, args.output_folder,prefix="Gene Score")

    for state in state_list[1:]:
        # Assuming threshold is calculated elsewhere
        best_cutoff = bulk_survival_adata.obs[f'gene_score_{state}'].median()
        kaplan_meier_analysis(bulk_survival_adata, f'gene_score_{state}', best_cutoff, args.output_folder, prefix=state)

    # save filtered genes as csv
    save_filtered_genes(filtered_gene_list_dict, args.output_folder)

    


if __name__ == "__main__":
    main()
