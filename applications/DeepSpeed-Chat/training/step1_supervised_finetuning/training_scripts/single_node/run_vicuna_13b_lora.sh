#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MODEL_PATH="/disk1/models/vicuna/vicuna-13b/"
LOG_DIR="./training_log_output/"
OUTPUT="/disk1/work/xiaym/models/dsc/vicuna/actor"
mkdir -p ${OUTPUT}

Actor_Lr=5e-4
Critic_Lr=5e-6

#Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP
deepspeed main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 2048 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --gradient_checkpointing \
   --lora_dim 128 \
   --lora_module_name layers. \
   --deepspeed \
   --output_dir $OUTPUT \
   &>> $LOG_DIR/vicuna_training.log