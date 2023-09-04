#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#ACTOR_MODEL_PATH="/disk1/work/xiaym/models/dsc/llama/actor/"
ACTOR_MODEL_PATH="/disk1/models/llama/65B/"
CRITIC_MODEL_PATH="/disk1/work/xiaym/models/dsc/llama/critic/"

OUTPUT="/disk1/work/xiaym/models/dsc/llama/final"
LOG_DIR="./training_log_output/"
mkdir -p $OUTPUT

Actor_Lr=5e-4
Critic_Lr=5e-6

deepspeed --master_port 13346 main.py \
   --data_path Dahoas/rm-static \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 1 \
   --per_device_mini_train_batch_size 1 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 1024 \
   --max_prompt_seq_len 1024 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 10 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --offload_reference_model \
   --offload \
   --disable_actor_dropout \
   --actor_lora_dim 4 \
   --actor_lora_module_name layers. \
   --critic_lora_dim 4 \
   --critic_lora_module_name layers. \
   --enable_hybrid_engine \
   --inference_tp_size 2 \
   --tp_gather_partition_size 2 \
   --release_inference_cache \
   --output_dir $OUTPUT \
    &>> $LOG_DIR/llama_training.log