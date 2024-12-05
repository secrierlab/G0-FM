#!/bin/bash

# Define paths and parameters
CODE_LOC="/home/shi/WorkSpace/projects/DormancyLM_workspace/scLM/code/" # path to repo end/with/scMultiNet_project/
DATA_LOC="/home/shi/WorkSpace/projects/DormancyLM_workspace/Data/" # path to data folder
RAW_DATA_LOC1=${DATA_LOC}"/step0_preprocess/scBERT_preprocess/Dataset_Cenk_Q/TrVal_dataset_GT_QuiescenceStatus_with_id.pkl" #/home/shi/WorkSpace/projects/scMultiNet_Data/Step_1_data/Dataset_all_counts_inhibitor/TrVal_dataset_Ground_Truth.pkl

echo "RAW_DATA_LOC1: " $RAW_DATA_LOC1

VOCAB_LOC=${CODE_LOC}+"/scLLM_support_data/support_data/vocab_16k.json"

echo "VOCAB_LOC: " $VOCAB_LOC

# model_check_points after 1_batch_train.sh 定义模型检查点和输出位置
MODEL_CKPT_Multi=(
${DATA_LOC}"/step1_train_phase1/cls_ckpt/scBERTQcat_scMultiNet_00/_epoch=03-auroc_val=0.92.ckpt"
${DATA_LOC}"/step1_train_phase1/cls_ckpt/scBERTQcat_scMultiNet_11/_epoch=01-auroc_val=0.98.ckpt"
${DATA_LOC}"/step1_train_phase1/cls_ckpt/scBERTQcat_scMultiNet_22/_epoch=03-auroc_val=0.95.ckpt"
)


VOCAB_PARAMS=${CODE_LOC}"/scLLM_support_data/support_data/gene2vec_16906_200.npy"

OUT_FOLDER=${DATA_LOC}"/step4_importnace_score/"

echo "OUT_FOLDER: " $OUT_FOLDER

OUT_LOC_Multi_train=(
${OUT_FOLDER}"/train/cls0.pkl"
${OUT_FOLDER}"/train/cls1.pkl"
${OUT_FOLDER}"/train/cls2.pkl"
)

OUT_LOC_Multi_val=(
${OUT_FOLDER}"/val/cls0.pkl"
${OUT_FOLDER}"/val/cls1.pkl"
${OUT_FOLDER}"/val/cls2.pkl"
)

# 循环执行推理任务
for i in {0..2}
do
    python scMulti_binary_infer_importance.py --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC1 --vocab_loc $VOCAB_LOC --model_ckpt ${MODEL_CKPT_Multi[i]} --vocab_params $VOCAB_PARAMS --out_loc ${OUT_LOC_Multi_train[i]} --index_label "target_label" --train_phase "train"
    python scMulti_binary_infer_importance.py --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC1 --vocab_loc $VOCAB_LOC --model_ckpt ${MODEL_CKPT_Multi[i]} --vocab_params $VOCAB_PARAMS --out_loc ${OUT_LOC_Multi_val[i]} --index_label "target_label" --train_phase "val"
done