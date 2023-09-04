#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#ACTOR_MODEL_PATH="/disk1/work/xiaym/models/dsc/opt/actor"
ACTOR_MODEL_PATH="/disk1/models/opt/opt-66b/"
CRITIC_MODEL_PATH="/disk1/work/xiaym/models/dsc/opt/critic/"
#CRITIC_MODEL_PATH="/disk1/models/opt/opt-350m/"

OUTPUT="/disk1/work/xiaym/models/dsc/opt/final"
LOG_DIR="./training_log_output/"
mkdir -p $OUTPUT

Actor_Lr=5e-4
Critic_Lr=5e-6

deepspeed --master_port 13346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 1 \
   --per_device_train_batch_size 1 \
   --per_device_mini_train_batch_size 1 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --actor_lora_dim 128 \
   --actor_lora_module_name decoder.layers. \
   --offload_reference_model \
   --output_dir $OUTPUT \
    &>> $LOG_DIR/opt_training.log
