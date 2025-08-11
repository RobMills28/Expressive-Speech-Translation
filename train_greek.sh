#!/bin/bash
# This script launches the fine-tuning job for Greek on CosyVoice-2.

# Configure GPU usage. Set to the number of GPUs you want to use inside the container.
export CUDA_VISIBLE_DEVICES="0"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# --- Training Parameters ---
PRETRAINED_MODEL_DIR="/app/CosyVoice/pretrained_models/CosyVoice2-0.5B"
OUTPUT_MODEL_DIR="/app/exp/cosyvoice2/llm/greek_sft"

# --- TorchRun Command ---
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS \
  /app/CosyVoice/cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config /app/CosyVoice/examples/libritts/cosyvoice2/conf/greek_sft.yaml \
    --train_data /app/data/greek_sft/parquet/data.list \
    --cv_data /app/data/greek_sft/parquet/data.list \
    --qwen_pretrain_path $PRETRAINED_MODEL_DIR/CosyVoice-BlankEN \
    --model llm \
    --checkpoint $PRETRAINED_MODEL_DIR/llm.pt \
    --model_dir $OUTPUT_MODEL_DIR \
    --tensorboard_dir /app/tensorboard/cosyvoice2/llm/greek_sft \
    --ddp.dist_backend "gloo" \
    --num_workers 4 \
    --prefetch 100 \
    --pin_memory \
    --use_amp