# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer, pipeline,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    args = parser.parse_args()

    return args



def print_utils(gen_output):
    output = str(gen_output[0]["generated_text"])
    # output = output.replace("<|endoftext|></s>", "")
    print(output)


def prompt_eval(args, generator_baseline, generator_fintuned, prompts):
    for prompt in prompts:
        print("==========Baseline: Greedy=========")
        r_base = generator_baseline(prompt,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_base)
        print("==========finetune: Greedy=========")
        r_finetune_g = generator_fintuned(prompt,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_finetune_g)
        print("====================prompt end=============================")
        print()
        print()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path_baseline,
                                              fast_tokenizer=False)

    model_config = AutoConfig.from_pretrained(args.model_name_or_path_baseline)

    model_baseline = AutoModelForCausalLM.from_pretrained(args.model_name_or_path_baseline, device_map="auto",
                                                 trust_remote_code=True,
                                                 from_tf=bool(".ckpt" in args.model_name_or_path_baseline),
                                                 config=model_config).half()

    model_fintuned = AutoModelForCausalLM.from_pretrained(args.model_name_or_path_finetune, device_map="auto",
                                                          trust_remote_code=True,
                                                          from_tf=bool(".ckpt" in args.model_name_or_path_finetune),
                                                          config=model_config).half()

    model_baseline.config.end_token_id = tokenizer.eos_token_id
    model_baseline.config.pad_token_id = model_baseline.config.eos_token_id
    model_baseline.resize_token_embeddings(len(tokenizer))
    generator_baseline = pipeline("text-generation",
                         model=model_baseline,
                         tokenizer=tokenizer,
                         device_map="auto")

    model_fintuned.config.end_token_id = tokenizer.eos_token_id
    model_fintuned.config.pad_token_id = model_fintuned.config.eos_token_id
    model_fintuned.resize_token_embeddings(len(tokenizer))
    generator_fintuned = pipeline("text-generation",
                         model=model_fintuned,
                         tokenizer=tokenizer,
                         device_map="auto")

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    if args.language == "English":
        prompts = [
            "Human: Please tell me about Microsoft in a few sentence? Assistant:",
            "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant:",
            "Human: Write a short poem about a wise frog. Assistant:",
            "Human: Who was president of the United States in 1955? Assistant:",
            "Human: How does a telescope work? Assistant:",
            "Human: Why do birds migrate south for the winter? Assistant:"
        ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:",
            "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]

    prompt_eval(args, generator_baseline, generator_fintuned, prompts)


if __name__ == "__main__":
    main()
