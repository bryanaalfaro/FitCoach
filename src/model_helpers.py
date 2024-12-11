# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Model Loading Helper Functions."""

import gc
import os
from typing import Callable, Optional, Union

import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.customllama.modeling_llama import LlamaForCausalLM
from src.model_wrappers import BaseVLModelWrapper, StreamVLModelWrapper
from src.utils import get_vlm_lang_handler
from src.vision_modules.adapter import XAttnAdapter
from src.vision_modules.vision_model import HypermodelFeatureProcessor

transformers.models.llama.modeling_llama.LlamaForCausalLM = LlamaForCausalLM


def model_builder_fn(
    llama_2_dir: str,
    checkpoint_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = torch.float,
    strict_checkpoint: bool = True,
    max_memory: Optional[dict] = None,
    peft_config: Optional[dict] = None,
    xattn_config: Optional[dict] = None,
    **kwargs,
) -> dict[str, Union[object, Callable]]:
    """
    :param llama_2_dir:
        Path to location of LLaMA2-7B weights.
    :param checkpoint_path:
        Path to Stream-VLM checkpoint.
    :param torch_dtype:
        Cast model to a specified dtype.
    :param strict_checkpoint:
        Whether the checkpoint state dict keys should strictly match the model.
    :param max_memory:
        When using device map, it tries to load each gpu to the brim before moving to the next.
        A dict in the following format can be provided to set a memory limit for each gpu: {0:
        '2GB'}
    :param peft_config:
        Configuration spec for parameter efficient fine-tuning:
        https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig
    :param xattn_config:
        Configuration spec for the cross attention layers added to the base LLM.

    :return:
        Dict containing model modules and checkpoint.
    """
    # Init vision model
    model_vision = HypermodelFeatureProcessor()
    vision_feats_encoded_dim = model_vision.get_embed_dim()

    # Load base LLM model
    kwargs.update({"trust_remote_code": True})
    model_lang = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=llama_2_dir,
        device_map="cpu",
        torch_dtype=torch_dtype,
        max_memory=max_memory,
        **kwargs,
    )

    # Init cross-attention layers in the LLM decoder
    model_lang_handler = get_vlm_lang_handler(model_lang)
    xattn_config.update({"vision_feat_dim": vision_feats_encoded_dim})
    if hasattr(model_lang_handler, "init_xattn"):
        model_lang_handler.init_xattn(**xattn_config)
    else:
        raise Exception("Custom LLM decoder was not correctly imported.")

    # Init adapter layer to transform features from the vision model to the cross-attention layers
    model_lang_embed_tokens_layer = model_lang_handler.embed_tokens
    model_adapter = XAttnAdapter(
        model_lang_embed_tokens_layer,
        model_lang_handler.adapter_insert_layers,
    )

    # Combine model components
    model_module = torch.nn.Module()
    model_module.add_module("vision", model_vision)
    model_module.add_module("adapter", model_adapter)
    model_module.add_module("lang", model_lang)

    # Init parameter efficient LLM
    if peft_config is not None:
        peft_config = LoraConfig(**peft_config)
        model_module.lang = get_peft_model(model_module.lang, peft_config)

    # Load remaining components from the checkpoint file
    if checkpoint_path is not None:
        if kwargs.get("pretrained_model_name_or_path", None) is None:
            kwargs.update({"pretrained_model_name_or_path": llama_2_dir})
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.split(checkpoint_path)[0]
        )

        # Resize embeddings to ensure that the number of tokens matches and update layer on adapter
        model_module.lang.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        model_module.adapter.embed_tokens = get_vlm_lang_handler(model_module.lang).embed_tokens

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in checkpoint.keys():
            model_module.load_state_dict(checkpoint["state_dict"], strict=strict_checkpoint)
        else:
            model_module.load_state_dict(checkpoint, strict=strict_checkpoint)

    # Init model wrapper
    model_wrapper = StreamVLModelWrapper

    return {"wrapper": model_wrapper, "model_module": model_module}


def make_model(
    llama_2_dir: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    fp16: bool = False,
    bf16: bool = False,
    trainable: Union[bool, dict] = True,
    **model_kwargs,
) -> BaseVLModelWrapper:
    """
    :param llama_2_dir:
        Path to location of LLaMA2-7B weights.
    :param checkpoint_path:
        Path to Stream-VLM checkpoint
    :param device:
        Device to load the model on.
    :param fp16:
        Boolean for whether model weights should be loaded with fp16.
    :param bf16:
        Boolean for whether model weights should be loaded with bf16.
    :param trainable:
        Whether the model is trainable (Keep False).

    :return:
        Wrapped model.
    """
    assert not (bf16 and fp16), "ERROR: Set only one of either bf16 or fp16 to be True."
    if bf16:
        model_dtype = torch.bfloat16
    elif fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float

    device = device or "cuda" if torch.cuda.is_available() else "cpu"
    max_memory = model_kwargs.get("max_memory")

    # Get tokenizer
    checkpoint_dir = None
    if checkpoint_path:
        checkpoint_dir = (
            os.path.split(checkpoint_path)[0]
            if os.path.isfile(checkpoint_path)
            else checkpoint_path
        )
    tokenizer_kwargs = {
        "pretrained_model_name_or_path": checkpoint_dir or llama_2_dir,
        "trust_remote_code": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)

    # Get model
    model_kwargs.update(
        {
            "llama_2_dir": llama_2_dir,
            "torch_dtype": model_dtype,
            "checkpoint_path": checkpoint_path,
            "max_memory": max_memory,
            "trust_remote_code": True,
        }
    )
    model = model_builder_fn(**model_kwargs)

    # Set model device
    for sub_model in model.values():
        try:
            sub_model.to(device, dtype=model_dtype)
        except (AttributeError, TypeError, RuntimeError) as e:
            print(e, type(make_model), " cannot be sent to ", device)

    # Wrap model
    wrapper_kwargs = {
        "device": device,
        "train_llm": trainable.get("llm", False),
        "train_xattn": trainable.get("xattn", False),
        "train_vision": trainable.get("vision", False),
    }
    wrapper_kwargs.update(tokenizer_kwargs)
    model_wrapper = model.pop("wrapper")
    model = model_wrapper(model["model_module"], tokenizer, **wrapper_kwargs)

    # Free up some memory
    gc.collect()
    torch.cuda.empty_cache()

    return model
