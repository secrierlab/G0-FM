#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Get the repository root (two levels up from script directory)
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "REPO_ROOT: $REPO_ROOT"
# Configuration
TASK_NAME="G0_LM"
DATASET_NAME="G0_arrest"
TARGET_LABEL="QuiescenceStatus"
RAW_DATA_LOC="${REPO_ROOT}/Data/processed/${DATASET_NAME}/TrVal_dataset_${TARGET_LABEL}.pkl"  # Update this path
OUT_LOC="${REPO_ROOT}/Outputs/${TASK_NAME}_x_${DATASET_NAME}/"
ENV_FILE="${SCRIPT_DIR}/.env"
DATASET_PARA_FILE="${REPO_ROOT}/Data/processed/${DATASET_NAME}/scBERT_dataset_setting.yml"  # Update this path

# Create output directory if it doesn't exist
mkdir -p "$OUT_LOC"


for i in {0..2}
do
    TASK_NAME="${TASK_NAME}_${i}"
    # Run training script
    python train_phase1.py \
        --repo_loc "$REPO_ROOT" \
        --env_file "$ENV_FILE" \
        --dataset_para_file "$DATASET_PARA_FILE" \
        --exp_name "$TASK_NAME" \
        --raw_data_loc "$RAW_DATA_LOC" \
        --out_loc "$OUT_LOC"
        --binarize $i
done