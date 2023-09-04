import argparse
import gc
import os
import struct
import time
from io import BytesIO
from typing import Optional, List

import ctranslate2
import numpy as np
import torch

from ctranslate2.converters.transformers import register_loader, ModelLoader
from ctranslate2.specs import ModelSpec
from ctranslate2.specs.model_spec import CURRENT_BINARY_VERSION, _dtype_to_type_id
from transformers.models.bert_japanese.tokenization_bert_japanese import spm

from utils.utils import print_rank_0


class InferenceModel():

    def __init__(self, model_name_or_path, actor_model, tokenizer, rank):
        self.model_name_or_path = model_name_or_path
        self.model = actor_model
        self.tokenizer = tokenizer
        self.rank = rank

    def get_generator(self):
        start = time.time()
        converter = CustomTransformersConverter(
            model=self.model,
            tokenizer=self.tokenizer,
            model_name_or_path=self.model_name_or_path,
            copy_files=["tokenizer.model"],
            load_as_float16=True,
            low_cpu_mem_usage=True,
        )

        files = converter.convert(
            "./temp",
            quantization="int8",
            force=True,
        )

        generator = ctranslate2.Generator(model_path="vicuna", device="cuda", files=files, device_index=self.rank)
        end = time.time()
        print_rank_0(f"convert actor model complete cost {(end - start):.2f}s", rank=self.rank)

        return generator

    def generate(self,  input_ids, mask, max_length):
        start = time.time()
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        sp = spm.SentencePieceProcessor(os.path.join(self.model_name_or_path, "tokenizer.model"))
        prompt_tokens = [sp.encode(prompt, out_type=str) for prompt in prompts]

        generator = self.get_generator()
        results = generator.generate_batch(prompt_tokens, max_length=max_length, min_length=max_length)
        seq = [result.sequences_ids[0] for result in results]
        end = time.time()
        print_rank_0(f"generate complete cost {(end - start):.2f}s", rank=self.rank)
        # print_rank_0(self.tokenizer.batch_decode(seq, skip_special_tokens=True), rank=self.rank)
        return seq


class CustomTransformersConverter(ctranslate2.converters.TransformersConverter):

    def __init__(
            self,
            model,
            tokenizer,
            model_name_or_path: str,
            activation_scales: Optional[str] = None,
            copy_files: Optional[List[str]] = None,
            load_as_float16: bool = False,
            revision: Optional[str] = None,
            low_cpu_mem_usage: bool = False,
    ):
        self._model = model
        self._tokenizer = tokenizer
        super().__init__(model_name_or_path,
                         activation_scales,
                         copy_files,
                         load_as_float16,
                         revision,
                         low_cpu_mem_usage)

    def load_model(self, model_class, model_name_or_path, **kwargs):
        return self._model

    def load_tokenizer(self, tokenizer_class, model_name_or_path, **kwargs):
        return self._tokenizer


    def convert(
        self,
        output_dir: str,
        vmap: Optional[str] = None,
        quantization: Optional[str] = None,
        force: bool = False,
    ):
        """Converts the model to the CTranslate2 format.

        Arguments:
          output_dir: Output directory where the CTranslate2 model is saved.
          vmap: Optional path to a vocabulary mapping file that will be included
            in the converted model directory.
          quantization: Weight quantization scheme
            (possible values are: int8, int8_float16, int16, float16).
          force: Override the output directory if it already exists.

        Returns:
          Path to the output directory.

        Raises:
          RuntimeError: If the output directory already exists and :obj:`force`
            is not set.
          NotImplementedError: If the converter cannot convert this model to the
            CTranslate2 format.
        """

        model_spec = self._load()
        if model_spec is None:
            raise NotImplementedError(
                "This model is not supported by CTranslate2 or this converter"
            )
        if vmap is not None:
            model_spec.register_vocabulary_mapping(vmap)

        model_spec.validate()
        model_spec.optimize(quantization=quantization)

        return self.save(model_spec)

    def save(self, model_spec: ModelSpec):
        """Saves this model on disk.

        Arguments:
          output_dir: Output directory where the model is saved.
        """
        files = {}
        model = self._serialize(model_spec)
        files['model.bin'] = model

        files['config.json'] = """{
                                  "bos_token": "<s>",
                                  "eos_token": "</s>",
                                  "unk_token": "<unk>"
                                }""".encode("utf-8")

        with open('/disk1/work/xiaym/dev/llama_convert/vicuna_ct2/vocabulary.txt', "rb") as vocab:
            files['vocabulary.txt'] = vocab.read()

        for filename, path in model_spec._files.items():
            in_file = open(path, "rb")  # opening for [r]eading as [b]inary
            data = in_file.read()
            in_file.close()
            files[filename] = data

        return files

    def _serialize(self, model_spec: ModelSpec):
        """Serializes the model variables."""
        variables = []
        aliases = []
        for variable in model_spec.variables(ordered=True):
            if isinstance(variable[1], str):
                aliases.append(variable)
            else:
                variables.append(variable)

        model = BytesIO()

        def _write_string(string):
            model.write(struct.pack("H", len(string) + 1))
            model.write(string.encode("utf-8"))
            model.write(struct.pack("B", 0))

        model.write(struct.pack("I", CURRENT_BINARY_VERSION))
        _write_string(model_spec.name)
        model.write(struct.pack("I", model_spec.revision))
        model.write(struct.pack("I", len(variables)))
        for name, value in variables:
            _write_string(name)
            model.write(struct.pack("B", len(value.shape)))
            for dim in value.shape:
                model.write(struct.pack("I", dim))
            model.write(struct.pack("B", _dtype_to_type_id(value.dtype)))
            model.write(struct.pack("I", value.nbytes))
            model.write(value.tobytes())
        model.write(struct.pack("I", len(aliases)))
        for alias, variable_name in aliases:
            _write_string(alias)
            _write_string(variable_name)

        content = model.getvalue()
        model.close()
        return content

@register_loader("LlamaConfig")
class LlamaLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "LlamaForCausalLM"

    def get_model_spec(self, model):
        spec = ctranslate2.specs.TransformerDecoderModelSpec.from_config(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            activation=ctranslate2.specs.Activation.SWISH,
            pre_norm=True,
            ffn_glu=True,
            rms_norm=True,
            rotary_dim=0,
            rotary_interleave=False,
        )

        self.set_decoder(spec.decoder, model.model)
        self.set_linear(spec.decoder.projection, model.lm_head)
        return spec

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token

    def set_layer_norm(self, spec, layer_norm):
        spec.gamma = layer_norm.weight.numpy()

    def set_decoder(self, spec, module):
        spec.scale_embeddings = False
        self.set_embeddings(spec.embeddings, module.embed_tokens)
        self.set_layer_norm(spec.layer_norm, module.norm)

        for layer_spec, layer in zip(spec.layer, module.layers):
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm, layer.input_layernorm
            )
            self.set_layer_norm(
                layer_spec.ffn.layer_norm, layer.post_attention_layernorm
            )

            wq = layer.self_attn.q_proj.weight.numpy()
            wk = layer.self_attn.k_proj.weight.numpy()
            wv = layer.self_attn.v_proj.weight.numpy()
            wo = layer.self_attn.o_proj.weight.numpy()

            layer_spec.self_attention.linear[0].weight = np.concatenate([wq, wk, wv])
            layer_spec.self_attention.linear[1].weight = wo

            self.set_linear(layer_spec.ffn.linear_0, layer.mlp.gate_proj)
            self.set_linear(layer_spec.ffn.linear_0_noact, layer.mlp.up_proj)
            self.set_linear(layer_spec.ffn.linear_1, layer.mlp.down_proj)

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()
