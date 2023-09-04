#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
ACTOR_MODEL_PATH="/disk1/models/llama/7B/" # Provide the ckpt path of the actor model
CRITIC_MODEL_PATH="/disk1/work/xiaym/models/dsc/vicuna/critic" # Provide the ckpt path of the critic model

LOG_DIR="./training_log_output"
OUTPUT="/disk1/work/xiaym/models/dsc/llama/final"
mkdir -p ${OUTPUT}

Actor_Lr=5e-6
Critic_Lr=9.65e-6

#Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets openai/webgpt_comparisons stanfordnlp/SHP
deepspeed --master_port 12346 main.py \
   --data_path openai/webgpt_comparisons \
   --data_split 8,1,1 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 4 \
   --per_device_mini_train_batch_size 4 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 1024 \
   --max_prompt_seq_len 1024 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 2 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 8 \
   --num_warmup_steps 10 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --disable_actor_dropout \
   --offload_reference_model \
   --enable_hybrid_engine \
   --inference_tp_size 2 \
   --tp_gather_partition_size 8 \
   --output_dir $OUTPUT \
    &>> $LOG_DIR/llama_training.log
