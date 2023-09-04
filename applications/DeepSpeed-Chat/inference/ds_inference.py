import datetime
import math

import deepspeed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.deepspeed import HfDeepSpeedConfig


def create_ds_model(model_path, tokenizer):
    ds_config = {
        "train_batch_size": 8,
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {
            "stage": 3,
            "offload_param": {
                "device": "none"
            },
            "offload_optimizer": {
                "device": "none"
            },
            "memory_efficient_linear": False,
        },
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "gradient_accumulation_steps": 1,
    }

    model_config = AutoConfig.from_pretrained(model_path)

    HfDeepSpeedConfig(ds_config)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        config=model_config)
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))

    model, *_ = deepspeed.initialize(
        model=model,
        config=ds_config,
        dist_init_required=True)
    return model


def inference(model, tokenizer, prompt):
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids).cuda(),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return outputs


def main():
    model_path = "/disk1/work/xiaym/models/dsc/llama/actor_004"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    model = create_ds_model(model_path, tokenizer)

    print(datetime.datetime.now())
    prompt = "Human: Please tell me about Microsoft in a few sentence? Assistant:"
    output = inference(model, tokenizer, prompt)
    print(output)

    print(datetime.datetime.now())
    prompt = "Human: Please tell me about Microsoft in a few sentence? Assistant:"
    output = inference(model, tokenizer, prompt)
    print(output)
    print(datetime.datetime.now())


if __name__ == "__main__":
    main()