#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the repository root (two levels up from script directory)
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "REPO_ROOT: $REPO_ROOT"

# Define paths and parameters
CODE_LOC="${REPO_ROOT}/code"
DATA_LOC="${REPO_ROOT}/Data"
RAW_DATA_LOC1="${DATA_LOC}/step0_preprocess/scBERT_preprocess/Dataset_Cenk_Q/TrVal_dataset_GT_QuiescenceStatus_with_id.pkl"

echo "RAW_DATA_LOC1: $RAW_DATA_LOC1"

VOCAB_LOC="${CODE_LOC}/scLLM_support_data/support_data/vocab_16k.json"

echo "VOCAB_LOC: $VOCAB_LOC"

# Model checkpoints after 1_batch_train.sh
MODEL_CKPT_Multi=(
"${DATA_LOC}/step1_train_phase1/cls_ckpt/scBERTQcat_scMultiNet_00/_epoch=03-auroc_val=0.92.ckpt"
"${DATA_LOC}/step1_train_phase1/cls_ckpt/scBERTQcat_scMultiNet_11/_epoch=01-auroc_val=0.98.ckpt"
"${DATA_LOC}/step1_train_phase1/cls_ckpt/scBERTQcat_scMultiNet_22/_epoch=03-auroc_val=0.95.ckpt"
)

VOCAB_PARAMS="${CODE_LOC}/scLLM_support_data/support_data/gene2vec_16906_200.npy"

OUT_FOLDER="${DATA_LOC}/step1_train_phase1/features"

echo "OUT_FOLDER: $OUT_FOLDER"

# Create output directories if they don't exist
mkdir -p "${OUT_FOLDER}/train"
mkdir -p "${OUT_FOLDER}/val"

OUT_LOC_Multi_train=(
"${OUT_FOLDER}/train/cls0.pkl"
"${OUT_FOLDER}/train/cls1.pkl"
"${OUT_FOLDER}/train/cls2.pkl"
"${OUT_FOLDER}/train/cls3.pkl"
"${OUT_FOLDER}/train/cls4.pkl"
)

OUT_LOC_Multi_val=(
"${OUT_FOLDER}/val/cls0.pkl"
"${OUT_FOLDER}/val/cls1.pkl"
"${OUT_FOLDER}/val/cls2.pkl"
"${OUT_FOLDER}/val/cls3.pkl"
"${OUT_FOLDER}/val/cls4.pkl"
)

# Run inference tasks
for i in {0..2}
do
    python scMulti_binary_infer.py \
        --code_loc "$CODE_LOC" \
        --raw_data_loc "$RAW_DATA_LOC1" \
        --vocab_loc "$VOCAB_LOC" \
        --model_ckpt "${MODEL_CKPT_Multi[i]}" \
        --vocab_params "$VOCAB_PARAMS" \
        --out_loc "${OUT_LOC_Multi_train[i]}" \
        --index_label "target_label" \
        --train_phase "train"

    python scMulti_binary_infer.py \
        --code_loc "$CODE_LOC" \
        --raw_data_loc "$RAW_DATA_LOC1" \
        --vocab_loc "$VOCAB_LOC" \
        --model_ckpt "${MODEL_CKPT_Multi[i]}" \
        --vocab_params "$VOCAB_PARAMS" \
        --out_loc "${OUT_LOC_Multi_val[i]}" \
        --index_label "target_label" \
        --train_phase "val"
done