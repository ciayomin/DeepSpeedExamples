#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT="../../output/llama/actor"

MODEL_NAME_OR_PATH="/disk1/models/llama/7B/"

mkdir -p $OUTPUT

#Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets

deepspeed --hostfile=hostfile.txt main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --model_name_or_path ${MODEL_NAME_OR_PATH} \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 512 \
   --learning_rate 1e-4 \
   --weight_decay 0.1 \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --lora_dim 128 \
   --lora_module_name layers. \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
