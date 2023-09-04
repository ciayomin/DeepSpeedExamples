#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MODEL_PATH="/disk1/models/llama/13B/"
LOG_DIR="./training_log_output/"
OUTPUT="/disk1/work/xiaym/models/dsc/llama/actor"
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT}

# Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP
deepspeed main.py \
   --data_path pvduy/sharegpt_alpaca_oa_vicuna_format \
   --data_split 1,1,2 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 2048 \
   --learning_rate 3e-4 \
   --num_train_epochs 4 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --lora_dim 8 \
   --lora_module_name layers. \
   --gradient_checkpoint \
   --seed 1234 \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $OUTPUT \
   &>> $LOG_DIR/llama_training.log