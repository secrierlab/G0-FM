import argparse
import os
# Get the directory of the current file
src_loc = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys
sys.path.append(src_loc)
print("⭐️ src_loc:", src_loc)



from src.DLM_exp_helpers.explain_learned.attention import load_Attention_matrix_with_label, process_binary_features, save_to_csv
from src.DLM_exp_helpers.utils import load_gene_list

def main(args):
    # Load gene name list
    gene_name_list = load_gene_list(args.gene_name)

    # Load data
    feature, label = load_Attention_matrix_with_label(args.feature_with_label)
    
    if args.binary:
        # Process binary classifier features
        mean_state_0, mean_state_1, mean_state_all = process_binary_features(feature, label)
    else:
        #mean_states_list = process_all_features(feature)
        raise NotImplementedError("All features processing is not implemented yet.")
    

    
    # Save results
    save_to_csv([mean_state_0, mean_state_1, mean_state_all], gene_name_list, 
                args.csv_folder, 
                args.binary, 
                args.k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gene feature data.")
    parser.add_argument('--feature_with_label', type=str, required=True, help="Path to the feature_with_label pickle file")
    parser.add_argument('--csv_folder', type=str, required=True, help="Folder to save CSV output")
    parser.add_argument('--gene_name', type=str, required=True, help="Path to the gene_name JSON file")
    parser.add_argument("--binary", action="store_true", help="Whether to process binary classifier features")
    parser.add_argument('--k', type=int, default=None, help="Top K genes to extract")
    args = parser.parse_args()
    main(args)