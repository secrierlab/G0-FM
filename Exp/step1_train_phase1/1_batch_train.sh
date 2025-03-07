#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the repository root (two levels up from script directory)
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "REPO_ROOT: $REPO_ROOT"

TASK_NAME="Qcat_scMultiNet_"
CODE_LOC="${REPO_ROOT}/code/scLLM/"
RAW_DATA_LOC="${REPO_ROOT}/Data/step0_preprocess/scBERT_preprocess/Dataset_Cenk_Q/TrVal_dataset_GT_QuiescenceStatus_with_id.pkl"

VOCAB_LOC="${REPO_ROOT}/code/scLLM_support_data/support_data/vocab_16k.json"
VOCAB_PARAMS="${REPO_ROOT}/code/scLLM_support_data/support_data/gene2vec_16906_200.npy"

MODEL_CKPT="${REPO_ROOT}/pretrained/panglao_pretrain.pth"

OUT_LOC="${REPO_ROOT}/Data/step1_train_phase1/"

for BINARIZE_TARGET in {0..2}; do
    echo "BINARIZE_TARGET: $BINARIZE_TARGET"
    NEW_TASK_NAME="${TASK_NAME}${BINARIZE_TARGET}"
    #python scBERT_Train.py --task_name $TASK_NAME --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc $OUT_LOC --binarize $BINARIZE_TARGET
    python scMulti_binary_Train.py --task_name $NEW_TASK_NAME --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc $OUT_LOC --binarize $BINARIZE_TARGET
done

#BINARIZE_TARGET=1
#python scBERT_Train.py --task_name $TASK_NAME --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc $OUT_LOC --binarize $BINARIZE_TARGET
#python scBERT_mut_Train.py --task_name $TASK_NAME --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc $OUT_LOC --binarize $BINARIZE_TARGET
