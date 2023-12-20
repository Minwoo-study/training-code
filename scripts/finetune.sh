#!/usr/bin/env bash

export OMP_NUM_THREADS=4
export WANDB_PROJECT="arspraxia_uft"
export CUDA_VISIBLE_DEVICES=0,1,2,3

RUN_NAME="llama-7b-chat-uft"
OUTPUT_DIR="output/$WANDB_PROJECT/$RUN_NAME"

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
TRAIN_DATASET="../../dataset/unsupervised_txt/train.llama.arrow"
EVAL_DATASET="../../dataset/unsupervised_txt/eval.llama.arrow"

BSZ=2

accelerate launch \
    './training/hf_trainer.py' \
    --model_name_or_path "$MODEL_NAME" \
    --train_file "$TRAIN_DATASET" \
    --eval_file "$EVAL_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --report_to "wandb" \
    --do_train --do_eval \
    --ddp_find_unused_parameters false \
    --optim 'adamw_torch_fused' \
    --seed 0 --data_seed 0 \
    --logging_first_step true --logging_steps 1 \
    --dataloader_num_workers 30 \
    --per_device_train_batch_size "$BSZ" --per_device_eval_batch_size "$BSZ" \
    --low_cpu_mem_usage false \
    --evaluation_strategy "steps" --eval_steps 5000 \
    --save_strategy "steps" --save_steps 5000 \
    --save_total_limit 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 28 \
    --num_train_epochs 2 \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules 'q_proj,v_proj' \
    --run_name "$RUN_NAME" \
    --uft \
    $@
