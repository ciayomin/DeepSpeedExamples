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
# pvduy/sharegpt_alpaca_oa_vicuna_format
deepspeed main.py \
   --data_path anon8231489123/ShareGPT_Vicuna_unfiltered \
   --data_split 98,1,1 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 2048 \
   --learning_rate 3e-4 \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --gradient_checkpoint \
   --seed 1234 \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $OUTPUT \
   &>> $LOG_DIR/llama_training.log