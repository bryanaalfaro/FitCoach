# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""VLM Model Wrappers."""

import math
from typing import Union

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from src.constants import (
    FEEDBACK_BEGIN_TOKEN,
    FEEDBACK_END_TOKEN,
    INFERENCE_SPEED,
    VISION_TOKEN,
)
from src.utils import get_vlm_lang_handler


class BaseVLModelWrapper:
    """Base wrapper class for Vision Language models

    :param model:
        The model to be wrapped.
    :param tokenizer:
        The tokenizer used by the model.
    :param device:
        The device the model is loaded.
    """

    def __init__(
        self, model: nn.Module, tokenizer: PreTrainedTokenizerBase, device: str = "cuda", **kwargs
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = self.model.lang.config

        # Update tokenizer attributes
        self._initialize_tokenizer_attrs()

        self.train_llm = kwargs.get("train_llm", False)
        self.train_vision = kwargs.get("train_vision", False)
        self.train_xattn = kwargs.get("train_xattn", False)
        self.trainable = self.train_llm or self.train_xattn or self.train_vision

    def __call__(self, *args, **kwargs):
        """The forward pass can be invoked with a direct call on the model"""
        return self.forward(*args, **kwargs)

    def __getattr__(self, *args, **kwargs):
        """Attributes on the wrapper class are still checked first"""
        return getattr(self.model, *args, **kwargs)

    def __hasattr__(self, *args, **kwargs):
        """Attribute checks are done on the core model itself"""
        return hasattr(self.model, *args, **kwargs)

    def _initialize_tokenizer_attrs(self):
        """Set tokenizer attributes"""
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"

    def forward(
        self,
        video: torch.tensor,
        input_ids: torch.tensor,
        vision_xattn_mask: torch.tensor,
        attention_mask: torch.tensor,
        **kwargs
    ) -> dict:
        """
        :param video:
            Tensor containing the video frames features (or raw video frames)
        :param input_ids:
            Tensor containing interleaved text tokens.
        :param vision_xattn_mask:
            Boolean tensor indicating where cross attention is applied. The number of
            locations set to True should be of the same length as video.
        :param attention_mask:
            Boolean attention mask for the base language model.

        :return:
            Language model output
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, *args, **kwargs):  # noqa D102
        raise NotImplementedError

    def parameters(self, recurse: bool = True) -> list[nn.Parameter]:
        """Get a list of combined parameters from all modules.

        :param recurse:
            If `True`, recursively iterates over all submodules.

        :return:
            The list of trainable parameters in the model.
        """
        _attrs = list(self.model.adapter.parameters(recurse=recurse))
        if self.train_vision:
            _attrs += list(self.model.vision.parameters(recurse=recurse))
        if self.train_llm:
            _attrs += list(self.model.lang.parameters(recurse=recurse))
        elif self.train_xattn:
            for layer in get_vlm_lang_handler(self.model.lang).layers:
                if layer.xattn_layer is not None:
                    _attrs += list(layer.xattn_layer.parameters(recurse=recurse))
        return _attrs

    def to(self, *args, **kwargs):
        """Ensure the returned model is still wrapped"""
        model = self.model.to(*args, **kwargs)
        return self.__class__(model, self.tokenizer, self.deepspeed_enabled)

    def train(self, *args, **kwargs):
        """Set model to train mode."""
        if isinstance(self.model, torch.nn.Module):
            for module in self.model.children():
                module.train(*args, **kwargs)


class StreamVLModelWrapper(BaseVLModelWrapper):
    """Model wrapper for streaming VLMs.

    This wrapper implements generation methods for interactive video-language tasks.
    """

    def __init__(
        self, model: nn.Module, tokenizer: PreTrainedTokenizerBase, device: str = "cuda", **kwargs
    ) -> None:
        super().__init__(model, tokenizer, device, **kwargs)

        self.special_tokens_dict = {
            VISION_TOKEN: self.tokenizer.encode(VISION_TOKEN)[-1],
            FEEDBACK_BEGIN_TOKEN: self.tokenizer.encode(FEEDBACK_BEGIN_TOKEN)[-1],
            FEEDBACK_END_TOKEN: self.tokenizer.encode(FEEDBACK_END_TOKEN)[-1],
        }

    def forward(
        self,
        video: torch.tensor,
        input_ids: torch.tensor,
        vision_xattn_mask: torch.tensor,
        attention_mask: torch.tensor,
        **kwargs
    ) -> dict:
        """
        :param video:
            Tensor containing the video frames features (or raw video frames)
        :param input_ids:
            Tensor containing interleaved text tokens.
        :param vision_xattn_mask:
            Boolean tensor indicating where cross attention is applied. The number of
            locations set to True should be of the same length as video.
        :param attention_mask:
            Boolean attention mask for the base language model.

        :return:
            Language model output
        """
        # Encode video
        if not self.train_vision:
            with torch.no_grad():
                encoded_video = self.model.vision(video)
        else:
            encoded_video = self.model.vision(video)

        # Adapt video features
        multi_model_embedding = self.model.adapter(encoded_video, input_ids, vision_xattn_mask)

        # Shift video features and input text tokens by 1 for training
        if attention_mask is None:
            model_attention_mask = torch.zeros_like(input_ids[:, :-1]).int()
            model_attention_mask[input_ids[:, :-1] != self.tokenizer.pad_token_id] = 1
        else:
            model_attention_mask = attention_mask[:, :-1]

        for key, embedding in multi_model_embedding.items():
            if (key in ["vision_xattn_mask", "language_timestamps"]) and embedding is not None:
                multi_model_embedding[key] = multi_model_embedding[key][:, :-1]
            elif isinstance(embedding, dict):
                multi_model_embedding[key]["vision"] = embedding["vision"][:, 1:]

        # Forward pass through the model
        lang_out = self.model.lang(
            inputs_embeds=multi_model_embedding, attention_mask=model_attention_mask
        )

        return lang_out

    @torch.no_grad()
    def _generate_interactive(
        self,
        encoded_video: torch.tensor,
        input_ids: torch.tensor,
        vision_xattn_mask: torch.tensor,
        feats_frequency: int,
        max_feedback_length: int,
        do_sample: bool,
        temperature: float,
        **kwargs
    ) -> torch.tensor:
        """Interactively generate feedback until the end of the input video.

        :param encoded_video:
            Encoded input video (by self.model.vision).
        :param input_ids:
            Input text tokens (usually the input prompt).
        :param vision_xattn_mask:
            Boolean tensor indicating where cross attention should be applied. The number of
            locations set to True should be of the same length as video.
        :param feats_frequency:
            Number of video features per second (usually the input video fps)
        :param max_feedback_length:
            Maximum possible length of a feedback.
        :param do_sample:
            Set to True to use temperature based sampling else greedy decoding is used by default.
        :param temperature:
            Temperature for sampling. Ignored if do_sample is False

        :return:
            List of output tokens.
        """
        assert vision_xattn_mask is not None
        output_ids = input_ids.clone()

        input_vision_idx = kwargs.get("input_vision_idx", 2)  # number of initial input frames
        skip_blind_frames = [False] * (
            input_vision_idx - 1
        )  # mask to skip (blind) frames that are received during feedback prediction

        past_key_values = None  # kv cache

        curr_response_len = 0
        feedback_mode = False
        # Continue generating until end of video
        while input_vision_idx < encoded_video["feats"].shape[1]:
            # Prepare video input
            encoded_video_feats = (
                encoded_video["feats"] if isinstance(encoded_video, dict) else encoded_video
            )
            encoded_video_spatial_res = (
                encoded_video.get("spatial_res", None) if isinstance(encoded_video, dict) else None
            )
            encoded_video_in_range = {
                "feats": encoded_video_feats[:, 1:input_vision_idx][
                    :, np.logical_not(skip_blind_frames)
                ],
                "spatial_res": encoded_video_spatial_res,
            }

            # Adapt video features
            multi_model_embedding = self.model.adapter(
                encoded_video_in_range, output_ids, vision_xattn_mask
            )

            # Generate next token logits
            lang_out = self.model.lang(
                inputs_embeds=multi_model_embedding,
                attention_mask=torch.ones_like(output_ids).to(self.device),
                use_cache=True,
                past_key_values=past_key_values,
            )

            # Update kv cache
            past_key_values = lang_out["past_key_values"]

            # Sample next token
            if not do_sample:  # Greedy decoding
                lang_out_logits = torch.argmax(lang_out["logits"], dim=-1)
                output_ids = torch.cat([output_ids, lang_out_logits[:, -1][:, None]], dim=1)
            else:  # Temperature based samping
                lang_out_logits = lang_out["logits"][:, -1]
                scaled_logits = lang_out_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                sampled_token = torch.multinomial(probs, num_samples=1)
                output_ids = torch.cat([output_ids, sampled_token], dim=1)

            # Sanity checks to deal with invalid states
            if feedback_mode:
                # if non-text tokens are produced in feedback mode then end the current feedback
                if (
                    output_ids[0, -1] == self.special_tokens_dict[VISION_TOKEN]
                    or output_ids[0, -1] == self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]
                    or curr_response_len > max_feedback_length
                ):
                    output_ids[0, -1] = self.special_tokens_dict[FEEDBACK_END_TOKEN]
            else:
                # if not in feedback mode only possible tokens are vision or feedback start tokens
                if (
                    output_ids[0, -1] != self.special_tokens_dict[VISION_TOKEN]
                    and output_ids[0, -1] != self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]
                ):
                    output_ids[0, -1] = self.special_tokens_dict[VISION_TOKEN]

            # State changes based on output
            if (
                output_ids[0, -1] == self.special_tokens_dict[VISION_TOKEN]
            ):  # continue to consume vision tokens
                input_vision_idx += 1
                skip_blind_frames.append(False)
            elif (
                output_ids[0, -1] == self.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]
            ):  # change to feedback mode
                curr_response_len = 0
                feedback_mode = True
            elif (
                output_ids[0, -1] == self.special_tokens_dict[FEEDBACK_END_TOKEN]
            ):  # finish producing feedbacks
                feedback_mode = False
                skip_forward = math.floor((curr_response_len / INFERENCE_SPEED) * feats_frequency)
                input_vision_idx += skip_forward
                skip_blind_frames += [True] * skip_forward
                curr_response_len = 0

            # Update masks depending upon outputs
            if output_ids[0, -1] == self.special_tokens_dict[VISION_TOKEN]:
                vision_xattn_mask_pad = torch.ones(input_ids.shape[0], 1) * 2
                vision_xattn_mask_pad = vision_xattn_mask_pad.to(vision_xattn_mask)
                vision_xattn_mask = torch.cat([vision_xattn_mask, vision_xattn_mask_pad], dim=1)
            else:
                vision_xattn_mask_pad = torch.zeros(input_ids.shape[0], 1)
                vision_xattn_mask_pad = vision_xattn_mask_pad.to(vision_xattn_mask)
                vision_xattn_mask = torch.cat([vision_xattn_mask, vision_xattn_mask_pad], dim=1)

        output_ids = output_ids.cpu().numpy()
        return output_ids

    def to_torch_tensor_for_generation(self, t: Union[np.ndarray, list]) -> torch.tensor:
        """
        Converts a numpy array or list to a torch tensor.
        """
        if (t is not None) and (not isinstance(t, torch.Tensor)):
            t = torch.as_tensor(t)[None].to(self.device)
        return t

    @torch.no_grad()
    def generate(
        self,
        input_prompt: list[int],
        video: Union[str, torch.tensor],
        vision_xattn_mask: torch.tensor,
        **kwargs
    ) -> torch.tensor:
        """Generate feedback interactively from the model

        :param video:
            Input video (features) of shape [num of frames, height, width, channels].
        :param input_prompt:
            Input text prompt tokens.
        :param vision_xattn_mask:
            Boolean tensor indicating where cross attention should be applied. The number of
            locations set to True should be of the same length as video.
        """
        assert len(video.shape) == 4

        video, vision_xattn_mask, input_prompt = map(
            lambda t: self.to_torch_tensor_for_generation(t),
            [video, vision_xattn_mask, input_prompt],
        )

        encoded_video = self.model.vision(video)
        return self._generate_interactive(encoded_video, input_prompt, vision_xattn_mask, **kwargs)
