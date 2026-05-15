#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_body_infer_train
#PBS -t 0-29
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
PBS_SUBREQNO_PAD=$(printf "%02d" $PBS_SUBREQNO)
DATA_ROOT="/work/SKIING/chenkaixu/MAC_ACM_MM/data/video"
OUT_ROOT="/work/SKIING/chenkaixu/MAC_ACM_MM/data/sam3d_body"
video_split_path=${PROJECT_ROOT}/pegasus/train_split_map/part_${PBS_SUBREQNO_PAD}.txt

# 模型
MODEL_ROOT_PATH="/work/SKIING/chenkaixu/MAC_ACM_MM/MAC_PyTorch/ckpt/sam-3d-body-dinov3"

# 声明普通数组

# 直接通过索引访问
process_flag="train"

workers_per_gpu=6

echo "🏁 Infer job started at: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Root: ${DATA_ROOT}"
echo "Output Root: ${OUT_ROOT}"
echo "Model Root Path: ${MODEL_ROOT_PATH}"
echo "Process Flag: ${process_flag}"

# === 3. 执行推理（每个作业只跑一个 fold） ===
python -m SAM3Dbody.main_split \
    model.root_path=${MODEL_ROOT_PATH} \
    paths.video_path=${DATA_ROOT} \
    paths.output_path=${OUT_ROOT} \
    infer.data_types=[${process_flag}] \
    paths.video_split_path=${video_split_path} \
    infer.workers_per_gpu=${workers_per_gpu} \

echo "🏁 Infer job finished at: $(date)"