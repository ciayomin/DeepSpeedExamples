#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MODEL_PATH="/disk1/models/llama/7B/"
LOG_DIR="./training_log_output/"
OUTPUT="/disk1/work/xiaym/models/dsc/llama/actor"
mkdir -p ${OUTPUT}

deepspeed main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --model_name_or_path /disk1/models/llama/7B/ \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --gradient_checkpoint \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $OUTPUT \
   &>> $LOG_DIR/llama_training.log