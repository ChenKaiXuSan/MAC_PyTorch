#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_body_infer
#PBS -t 0-2
#PBS -o logs/pegasus/sam3d_body/train_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/sam3d_body/train_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
PROJECT_ROOT="/work/SKIING/chenkaixu/MAC_ACM_MM/MAC_PyTorch"
cd "${PROJECT_ROOT}" || exit 1

mkdir -p logs/pegasus/sam3d_body

source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body

conda env list

# === 2. 推理参数（按需修改） ===
DATA_ROOT="/work/SKIING/chenkaixu/MAC_ACM_MM/data/video"
OUT_ROOT="/work/SKIING/chenkaixu/MAC_ACM_MM/data/sam3d_body"

# 模型
MODEL_ROOT_PATH="/work/SKIING/chenkaixu/MAC_ACM_MM/MAC_PyTorch/ckpt/sam-3d-body-dinov3"

# 声明普通数组
my_array=("train" "val" "test")

# 直接通过索引访问
process_flag="${my_array[$PBS_SUBREQNO]}"

echo "🏁 Infer job started at: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Root: ${DATA_ROOT}"
echo "Output Root: ${OUT_ROOT}"
echo "Model Root Path: ${MODEL_ROOT_PATH}"
echo "Process Flag: ${process_flag}"

# === 3. 执行推理（每个作业只跑一个 fold） ===
python -m SAM3Dbody.main \
    model.root_path=${MODEL_ROOT_PATH} \
    paths.video_path=${DATA_ROOT} \
    paths.output_path=${OUT_ROOT} \
    infer.data_types=[${process_flag}] \

echo "🏁 Infer job finished at: $(date)"