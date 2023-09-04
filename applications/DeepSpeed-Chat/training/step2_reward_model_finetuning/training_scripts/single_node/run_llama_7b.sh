#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MODEL_PATH="/disk1/models/llama/7B/" # Provide the ckpt path of the actor model
LOG_DIR="./training_log_output/"
OUTPUT="/disk1/work/xiaym/models/dsc/llama/critic"
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT}

# Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP
deepspeed  main.py \
   --data_path Dahoas/rm-static stanfordnlp/SHP \
   --data_split 2,4,4 \
   --num_padding_at_beginning 0 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 2048 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.01 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage 2 \
   --gradient_checkpoint \
   --deepspeed \
   --output_dir $OUTPUT \
   &>> $LOG_DIR/llama_training.log