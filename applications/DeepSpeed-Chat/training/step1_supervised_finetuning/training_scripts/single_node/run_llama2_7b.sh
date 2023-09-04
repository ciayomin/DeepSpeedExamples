#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MODEL_PATH="/cloud-model/huggingFace/Models/llama-2/7B/"
LOG_DIR="./training_log_output/"
OUTPUT="/disk1/work/xiaym/models/sft/llama2/7B/"
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT}

# Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP
# pvduy/sharegpt_alpaca_oa_vicuna_format   anon8231489123/ShareGPT_Vicuna_unfiltered
deepspeed main.py \
   --data_path Chinese_Llama_Alpaca \
   --data_split 98,1,1 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 4096 \
   --learning_rate 1e-4 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --gradient_checkpoint \
   --seed 1234 \
   --lora_dim 8 \
   --lora_module_name layers. \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $OUTPUT \
   &>> $LOG_DIR/llama2_training.log
