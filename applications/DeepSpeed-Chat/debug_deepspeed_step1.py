"""
Step1:
single_gpu: --master_port=29501 --num_gpus 1 ./training/step1_supervised_finetuning/main.py --gradient_accumulation_steps 2 --lora_dim 128 --zero_stage 0 --deepspeed --output_dir ./output/llama/actor --model_name_or_path
/disk1/work/huangj/models/llama/7B
Step2:
single_gpu: --master_port=29501 --num_gpus 1 ./training/step2_reward_model_finetuning/main.py --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --zero_stage 0 --deepspeed --output_dir ./output/critic --model_name_or_path
facebook/opt-350m
Step3: --master_port=29501 ./training/step3_rlhf_finetuning/main.py --actor_model_name_or_path ./output/actor --critic_model_name_or_path ./output/critic  --actor_zero_stage 0 --critic_zero_stage 0 --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing --output_dir ./output/final
"""
import os

from deepspeed.launcher.runner import main

# os.environ["PATH"] = os.environ["PATH"] + ":/root/miniconda3/bin/"


if __name__ == '__main__':
    main()