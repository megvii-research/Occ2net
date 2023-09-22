#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/tartanair_test_1600.py"
main_cfg_path="configs/loftr/tartanair/loftr_ds_quadtree_eval.py"
ckpt_path="logs/tb_logs/indoor-bs=16/version_0/checkpoints/epoch=29-step=337599.ckpt"
# ckpt_path="./logs/tb_logs/tartanair-quadtree-ds-bs=8/version_0/checkpoints/last.ckpt"
dump_dir="dump/loftr_ds_indoor"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=-1
torch_num_workers=4
batch_size=1  # per gpu

python3 -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark 
    