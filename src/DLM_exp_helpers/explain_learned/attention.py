import argparse
import pickle
import json
import numpy as np
import pandas as pd


# 加载Attention矩阵和标签
# 因为step2目标的输出可能是Attention矩阵和label的形式，所以在这里确认
def load_Attention_matrix_with_label(feature_path):
    """Load feature and label data from a pickle file."""
    with open(feature_path, 'rb') as f:
        feature, label = pickle.load(f)
    print("Loaded feature and label data with shapes:", feature.shape, label.shape)
    return feature, label

def process_binary_features(feature, label):
    """Process features for binary classification by calculating means for each label state."""
    state_0_indices = np.where(label == 0)[0]
    state_1_indices = np.where(label == 1)[0]
    
    feature_state_0 = feature[state_0_indices]
    feature_state_1 = feature[state_1_indices]

    mean_state_0 = np.mean(feature_state_0, axis=0).reshape(1, -1)[:,:-1].flatten()
    mean_state_1 = np.mean(feature_state_1, axis=0).reshape(1, -1)[:,:-1].flatten()
    mean_state_all = np.mean(feature, axis=0).reshape(1, -1)[:,:-1].flatten()

    print("Processed means for state 0, state 1, and all states.")
    return mean_state_0, mean_state_1, mean_state_all



def save_to_csv(mean_states_list, gene_name_list, csv_folder,
                is_binary:bool=True, k=None):
    """Save mean states to CSV and top-K genes to a csv file."""
    if is_binary:
        mean_state_0, mean_state_1, mean_state_all = mean_states_list
        # Create DataFrame and save to CSV
        df = pd.DataFrame({
        'mean_state_0': mean_state_0,
        'mean_state_1': mean_state_1,
            'mean_state_all': mean_state_all
        }, index=gene_name_list)
        
        if k is None: k = len(gene_name_list)
        # Get top K genes
        topK_genes_state_0 = df['mean_state_0'].nlargest(k).index.tolist()
        topK_genes_state_1 = df['mean_state_1'].nlargest(k).index.tolist()
        topK_genes_state_all = df['mean_state_all'].nlargest(k).index.tolist()

        # Save top-K gene names to a pickle file
        topK_genes_dict = {
            'topK_genes_state_0': topK_genes_state_0,
            'topK_genes_state_1': topK_genes_state_1,
            'topK_genes_state_all': topK_genes_state_all
        }

    else:
        raise NotImplementedError("All features processing is not implemented.")
    
    # save _score to csv
    csv_path = f"{csv_folder}/mean_states_score.csv"
    df.to_csv(csv_path)
    print(f"CSV file saved to {csv_path}")

    # to csv
    csv_path = f"{csv_folder}/top{k}_genes.csv"
    pd.DataFrame(topK_genes_dict).to_csv(csv_path)
    print(f"Top K genes saved to {csv_path}")

    return topK_genes_dict

