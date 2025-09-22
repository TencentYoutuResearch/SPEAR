#!/bin/bash
set -x

nnodes=1
nproc_per_node=2
master_addr=127.0.0.1
master_port=12345
node_rank=0

project_name="retool_sft"
experiment_name=qwen-3-32b-instruct
ROOT_PATH=${1:-$PWD}

TRAIN_DATA=$ROOT_PATH/datasets/ReTool-SFT/data/train-00000-of-00001.parquet
EVAL_DATA=$ROOT_PATH/datasets/ReTool-SFT/data/train-00000-of-00001.parquet
MODEL_PATH=$ROOT_PATH/model/Qwen3-32B-Instruct
SAVE_PATH=$ROOT_PATH/checkpoints/$project_name/$experiment_name
echo $SAVE_PATH


torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     --master-addr=$master_addr \
     --master-port=$master_port \
     --node-rank=$node_rank \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=1 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true