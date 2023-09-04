# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import re
import logging

# import ray
import torch
import transformers  # noqa: F401
import os
import json
from transformers import pipeline, set_seed, AutoModelForCausalLM
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()
    return args


def get_generator(path):
    # if os.path.exists(path):
    #     # Locally tokenizer loading has some issue, so we need to force download
    #     model_json = os.path.join(path, "config.json")
    #     if os.path.exists(model_json):
    #         model_json_file = json.load(open(model_json))
    #         model_name = model_json_file["_name_or_path"]
    #         tokenizer = AutoTokenizer.from_pretrained(model_name,
    #                                                   fast_tokenizer=True)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    # tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(path,
                                                 low_cpu_mem_usage=True, torch_dtype=torch.float16,
                                                 device_map="auto")

    # model.config.end_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(len(tokenizer))
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         device_map="auto")
    return generator


def get_user_input(user_input):
    tmp = input("Enter input (type 'quit' to exit, 'clear' to clean memory): ")
    new_inputs = f"Human: {tmp}\n Assistant:"
    user_input += f" {new_inputs}"
    return user_input, tmp == "quit", tmp == "clear"


def get_model_response(generator, user_input, max_new_tokens):
    response = generator(user_input, max_new_tokens=max_new_tokens,
                         do_sample=True,
                         temperature=0.7)
    return response


def process_response(response, num_rounds):
    output = str(response[0]["generated_text"])
    output = output.replace("<|endoftext|>", "")
    all_positions = [m.start() for m in re.finditer("Human: ", output)]
    place_of_second_q = -1
    if len(all_positions) > num_rounds:
        place_of_second_q = all_positions[num_rounds]
    if place_of_second_q != -1:
        output = output[0:place_of_second_q]
    return output


def get_model_response_2(model, tokenizer, prompt):
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
    )
    output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return outputs

def main(args):
    # generator = get_generator(args.path)
    # set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.path, low_cpu_mem_usage=True, torch_dtype=torch.float16,
          device_map = "auto",
    )

    user_input = ""
    num_rounds = 0
    while True:
        num_rounds += 1
        user_input, quit, clear = get_user_input(user_input)

        if quit:
            break
        if clear:
            user_input, num_rounds = "", 0
            continue

        # response = get_model_response(generator, user_input,
        #                               args.max_new_tokens)
        # output = process_response(response, num_rounds)
        output = get_model_response_2(
            model, tokenizer, user_input)

        print("-" * 30 + f" Round {num_rounds} " + "-" * 30)
        print(f"{output}")
        user_input = f"{output}\n\n"


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    main(args)

# Example:
"""
 Human: what is internet explorer?
 Assistant:
Internet Explorer is an internet browser developed by Microsoft. It is primarily used for browsing the web, but can also be used to run some applications. Internet Explorer is often considered the best and most popular internet browser currently available, though there are many other options available.

 Human: what is edge?
 Assistant:
 Edge is a newer version of the Microsoft internet browser, developed by Microsoft. It is focused on improving performance and security, and offers a more modern user interface. Edge is currently the most popular internet browser on the market, and is also used heavily by Microsoft employees.
"""
