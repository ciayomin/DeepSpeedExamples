#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


ACTOR_ZERO_STAGE="--actor_zero_stage 2"
CRITIC_ZERO_STAGE="--critic_zero_stage 2"
ACTOR_MODEL_PATH="../../output/llama/actor" # Provide the ckpt path of the actor model
CRITIC_MODEL_PATH="../../output/llama/critic" # Provide the ckpt path of the critic model

LOG_DIR="./training_log_output/"

Actor_Lr=5e-4
Critic_Lr=5e-6

OUTPUT="/disk1/work/xiaym/models/dsc/llama/final"
mkdir -p ${OUTPUT}

deepspeed --master_port 12346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   ${ACTOR_ZERO_STAGE} \
   ${CRITIC_ZERO_STAGE} \
   --actor_lora_dim 128 \
   --actor_lora_module_name layers. \
   --critic_lora_dim 128 \
   --critic_lora_module_name layers. \
   --only_optimize_lora \
   --output_dir ${OUTPUT} \
    &> ${OUTPUT}/llama_training.log
