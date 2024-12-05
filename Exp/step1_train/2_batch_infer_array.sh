#!/bin/bash -l

# Batch script to run a serial array job under SGE.

# Request 48 hours of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:00

# Request 16 gigabytes of RAM for each core/thread.
#$ -l mem=16G

# Request GPU.
#$ -l gpu=1

# Request shared memory parallel.
#$ -pe smp 8

# Request temp space.
#$ -l tmpfs=50G 

# Set up the job array. In this instance we have requested 8 tasks.
#$ -t 2-9

# Set the name of the job.
#$ -N GSE_infer

# Set the working directory to somewhere in your scratch space.
#$ -wd /home/ucbtsp5/Scratch/24Exp02_scLLM/EXPOUT/Logs/

# Run the application.

echo "$JOB_NAME $SGE_TASK_ID"

# 8. Run the application.
module load python3/3.9
module load openjpeg/2.4.0/gnu-4.9.2
module load openslide/3.4.1/gnu-4.9.2

# Initialize Conda
__conda_setup="$('$HOME/Scratch/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$HOME/Scratch/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/Scratch/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="$HOME/Scratch/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

source ~/.bashrc
which conda
conda activate scLLM

# 定义CSV文件路径
CSV_FILE="/home/ucbtsp5/Scratch/24Exp02_scLLM/Data/GSE234181_data.csv"
# Step 1: 用GSE234181_data.csv的filename列$SGE_TASK_ID行内容赋值给$NEW_TASK_NAME
NEW_TASK_NAME=$(sed -n "${SGE_TASK_ID}p" "$CSV_FILE" | cut -d',' -f1)

# Step 2: 用GSE234181_data.csv的full_path列$SGE_TASK_ID行内容赋值给$RAW_DATA_LOC1
RAW_DATA_LOC1=$(sed -n "${SGE_TASK_ID}p" "$CSV_FILE" | cut -d',' -f2)

# 输出变量以供检查
echo "Task Name: $NEW_TASK_NAME"
echo "Raw Data Location: $RAW_DATA_LOC1"

# (2) 根据NEW_TASK_NAME生成checkpoint文件夹路径
MODEL_CKPT_FOLDER="/home/ucbtsp5/Scratch/24Exp02_scLLM/PreTrained/${NEW_TASK_NAME}"

# (3) 获取checkpoint文件（假设每个文件夹里有一个ckpt文件）
MODEL_CKPT=$(find "$MODEL_CKPT_FOLDER" -type f -name "*.ckpt" | head -n 1)

# 输出模型路径以供检查
echo "Model Checkpoint: $MODEL_CKPT"

# (4) 创建推理结果的输出文件夹
OUT_LOC="/home/ucbtsp5/Scratch/24Exp02_scLLM/EXPOUT/attention_out/PreTrained${NEW_TASK_NAME}"

# 如果推理结果文件夹不存在则创建
mkdir -p "$OUT_LOC"
mkdir -p "$OUT_LOC/train"
mkdir -p "$OUT_LOC/val"

# (5) 调用推理脚本，分别保存训练集和验证集的结果
CODE_LOC="/home/ucbtsp5/Scratch/24Exp02_scLLM/G0-LM/src/"
VOCAB_LOC="/home/ucbtsp5/Scratch/24Exp02_scLLM/G0-LM/src/scLLM_support_data/support_data/vocab_16k.json"
VOCAB_PARAMS="/home/ucbtsp5/Scratch/24Exp02_scLLM/G0-LM/src/scLLM_support_data/support_data/gene2vec_16906_200.npy"


# 训练集推理
python /home/ucbtsp5/Scratch/24Exp02_scLLM/G0-LM/Exp/step1_train/scMulti_simple_Infer.py --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC1 --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc "$OUT_LOC/train" --index_label "target_label" --train_phase "train"

# 验证集推理
python /home/ucbtsp5/Scratch/24Exp02_scLLM/G0-LM/Exp/step1_train/scMulti_simple_Infer.py --code_loc $CODE_LOC --raw_data_loc $RAW_DATA_LOC1 --vocab_loc $VOCAB_LOC --model_ckpt $MODEL_CKPT --vocab_params $VOCAB_PARAMS --out_loc "$OUT_LOC/val" --index_label "target_label" --train_phase "val"
