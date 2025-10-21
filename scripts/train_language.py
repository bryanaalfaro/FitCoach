#!/usr/bin/env python3
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Finetune Stream-VLM on QEVD Fit-Coach workouts."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

import matplotlib.pyplot as plt

from src.constants import (
    FEEDBACK_BEGIN_TOKEN,
    FEEDBACK_END_TOKEN,
    VISION_TOKEN,
)
from src.fitness_datasets import load_dataset
from src.model_helpers import make_model

# For easy conversion of logits to text if using a debugger
from src.utils import logits_to_text
# This contains helpful functions for dataset preparation
from src.evaluators import InteractiveFeedbackEvaluator

IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stream-VLM on full workouts.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (defaults to CUDA if available).",
    )
    return parser.parse_args()


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Tokenizer could not encode token: {token}")
    return ids[-1]

def _append_tensors_with_padding(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Append tensors with padding to the same length. Currently only works for 2D tensors with the size of dim 1 = 1
    """
    max_length = max(t.shape[1] for t in tensors)
    padded_tensors = torch.zeros(len(tensors), max_length).to(tensors[0])
    for i, t in enumerate(tensors):
        padded_tensors[i:i+1, :t.shape[1]] = t
    return padded_tensors


@dataclass
class StreamingSequence:
    input_ids: List[int]
    attention_mask: List[int]
    vision_xattn_mask: List[int]
    loss_mask: List[int]


def build_streaming_sequence(
    tokenizer,
    system_prompt: str,
    responses: Sequence[str],
    response_timestamps: Sequence[float],
    video_timestamps: np.ndarray,
    special_token_ids: Dict[str, int],
    max_sequence_length: int | None = None,
) -> StreamingSequence:
    """Interleave <vision> tokens with timestamp-aligned feedback spans."""
    if len(video_timestamps) == 0:
        raise ValueError("Video features must contain at least one timestamp.")

    prompt_ids = tokenizer.encode(system_prompt, add_special_tokens=True)
    if tokenizer.eos_token_id is not None and prompt_ids[-1] == tokenizer.eos_token_id:
        prompt_ids = prompt_ids[:-1]

    input_ids: List[int] = list(prompt_ids) + [tokenizer.encode(VISION_TOKEN)[-1]]
    # attention_mask: List[int] = [1] * len(input_ids)
    # vision_xattn_mask: List[int] = [0] * len(input_ids)
    # ignore prompt tokens (prompt includes one vision token, may need to change this)
    loss_mask: List[int] = [0] * len(input_ids) 

    video_rel = video_timestamps - video_timestamps[0]
    video_rel = video_rel.astype(np.float64)

    response_ts = np.asarray(response_timestamps, dtype=np.float64)
    if response_ts.size:
        response_rel = response_ts - video_timestamps[0]
    else:
        response_rel = response_ts

    # breakpoint()

    frame_buckets: List[List[str]] = [[] for _ in range(len(video_rel))]
    if response_rel.size:
        for feedback, ts in zip(responses, response_rel):
            if feedback is None:
                continue
            feedback = feedback.strip()
            if not feedback:
                continue
            insert_idx = int(np.searchsorted(video_rel, ts, side="left"))
            insert_idx = min(max(insert_idx, 0), len(video_rel) - 1)
            frame_buckets[insert_idx].append(feedback)

    for frame_idx in range(len(video_rel)):
        input_ids.append(special_token_ids[VISION_TOKEN])
        # attention_mask.append(1)
        # vision_xattn_mask.append(2)
        loss_mask.append(1)

        for feedback in frame_buckets[frame_idx]:
            input_ids.append(special_token_ids[FEEDBACK_BEGIN_TOKEN])
            # attention_mask.append(1)
            # vision_xattn_mask.append(0)
            loss_mask.append(1)

            feedback_ids = tokenizer.encode(feedback, add_special_tokens=False)
            input_ids.extend(feedback_ids)
            # attention_mask.extend([1] * len(feedback_ids))
            # vision_xattn_mask.extend([0] * len(feedback_ids))
            loss_mask.extend([1] * len(feedback_ids))

            input_ids.append(special_token_ids[FEEDBACK_END_TOKEN])
            # attention_mask.append(1)
            # vision_xattn_mask.append(0)
            loss_mask.append(1)

        if max_sequence_length and len(input_ids) >= max_sequence_length:
            input_ids = input_ids[:max_sequence_length]
            # attention_mask = attention_mask[:max_sequence_length]
            # vision_xattn_mask = vision_xattn_mask[:max_sequence_length]
            loss_mask = loss_mask[:max_sequence_length]
            break
    attention_mask = InteractiveFeedbackEvaluator._get_attention_mask(input_ids)

    valid_video_indices = np.where(
        np.array(input_ids) == tokenizer.encode(VISION_TOKEN)[-1]
    )[0]
    vision_xattn_mask = np.array([0] * len(input_ids))
    vision_xattn_mask[valid_video_indices] = 1
    # re-unmask any vision tokens from the prompt (needed for sliding window)
    vision_xattn_mask[:len(prompt_ids)] = 0


    return StreamingSequence(
        input_ids=input_ids,
        attention_mask=attention_mask,
        vision_xattn_mask=vision_xattn_mask,
        loss_mask=loss_mask,
    )


def clip_timestamps(timestamps: np.ndarray, start_timestamp: float, end_timestamp: float) -> np.ndarray:
    """
    Clip timestamps to the given start and end timestamps. (Essentially does the same as _get_video_for_episode except 
    with the timestamps array instead of the video array.)

    :param timestamps:
        Array of timestamps.
    :param start_timestamp:
        Start timestamp.
    :param end_timestamp:
        End timestamp.

    """
    return timestamps[np.logical_and(timestamps > start_timestamp, timestamps <= end_timestamp)]

class WorkoutTrainingDataset(Dataset):
    """Lazily loads EfficientNet features + transcripts for training."""

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_sequence_length: int | None = None,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.special_token_ids = {
            VISION_TOKEN: _token_id(tokenizer, VISION_TOKEN),
            FEEDBACK_BEGIN_TOKEN: _token_id(tokenizer, FEEDBACK_BEGIN_TOKEN),
            FEEDBACK_END_TOKEN: _token_id(tokenizer, FEEDBACK_END_TOKEN),
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # usual path when not doing a sliding window
        if "segments" not in self.dataset[idx].keys():
            record = self.dataset[idx]
            video = np.load(record["efficientnet_features_path"])
            video = video.astype(np.float32)
            video_timestamps = np.load(record["efficientnet_timestamps_path"]).astype(np.float64)
            ep_start = record.get("exercise_start_timestamp", video_timestamps[0])
            ep_end = record.get("exercise_end_timestamp", video_timestamps[-1])
            video = InteractiveFeedbackEvaluator._get_video_for_episode(
                video, video_timestamps, ep_start, ep_end
            )
            video_timestamps = clip_timestamps(video_timestamps, ep_start, ep_end)

            responses = record["responses"]
            response_timestamps = record["response_timestamps"]

            seq = build_streaming_sequence(
                tokenizer=self.tokenizer,
                system_prompt=record["system"],
                responses=responses,
                response_timestamps=response_timestamps,
                video_timestamps=video_timestamps,
                special_token_ids=self.special_token_ids,
                max_sequence_length=self.max_sequence_length,
            )

            return {
                "video": video,
                "input_ids": np.array(seq.input_ids, dtype=np.int64),
                "attention_mask": np.array(seq.attention_mask, dtype=np.int64),
                "vision_xattn_mask": np.array(seq.vision_xattn_mask, dtype=np.int64),
                "loss_mask": np.array(seq.loss_mask, dtype=np.int64),
            }
        
        # sliding window dataset, do the same as above but for each segment and using the previous feedbacks as the system prompt
        record = self.dataset[idx]
        system_prompt = record["system"]
        skip_prompt = len(self.tokenizer.encode(system_prompt))
        prompt_append = VISION_TOKEN
        video = np.load(record["efficientnet_features_path"])
        video = video.astype(np.float32)
        video_timestamps = np.load(record["efficientnet_timestamps_path"]).astype(np.float64)
        video_segments = []
        input_ids_segments = []
        attention_mask_segments = []
        vision_xattn_mask_segments = []
        loss_mask_segments = []

        for segment in record["segments"]:
            ep_start = segment["exercise_start_timestamp"]
            ep_end = segment["exercise_end_timestamp"]
            video_segment = InteractiveFeedbackEvaluator._get_video_for_episode(
                video, video_timestamps, ep_start, ep_end
            )
            video_timestamps_segment = clip_timestamps(video_timestamps, ep_start, ep_end)
            video_segments.append(video_segment)
            seq_segment = build_streaming_sequence(
                tokenizer=self.tokenizer,
                system_prompt=system_prompt + prompt_append,
                responses=segment["responses"],
                response_timestamps=segment["response_timestamps"],
                video_timestamps=video_timestamps_segment,
                special_token_ids=self.special_token_ids,
                max_sequence_length=self.max_sequence_length,
            )
            input_ids_segments.append(seq_segment.input_ids)
            attention_mask_segments.append(seq_segment.attention_mask)
            vision_xattn_mask_segments.append(seq_segment.vision_xattn_mask)
            loss_mask_segments.append(seq_segment.loss_mask)
            prompt_append = self.tokenizer.decode(seq_segment.input_ids[skip_prompt:])
            skip_prompt = len(self.tokenizer.encode(system_prompt + prompt_append))
        
        return {
            "video": video_segments,
            "input_ids": input_ids_segments,
            "attention_mask": attention_mask_segments,
            "vision_xattn_mask": vision_xattn_mask_segments,
            "loss_mask": loss_mask_segments,
        }
            
            


class Collator:
    """Batch collator (currently assumes batch_size == 1)."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(batch) != 1:
            raise ValueError(
                "FullWorkoutCollator currently supports batch_size == 1. "
                "Increase support for padding if you need larger batches."
            )

        sample = batch[0]

        # Single windows:
        if isinstance(sample["video"], np.ndarray):
            video = torch.from_numpy(sample["video"]).unsqueeze(0)  # [1, T, C, H, W]
            out: Dict[str, torch.Tensor] = {
                "video": video,
                "input_ids": torch.from_numpy(sample["input_ids"]).unsqueeze(0),
                "attention_mask": torch.from_numpy(sample["attention_mask"]).unsqueeze(0),
                "vision_xattn_mask": torch.from_numpy(sample["vision_xattn_mask"]).unsqueeze(0),
                "loss_mask": torch.from_numpy(sample["loss_mask"]).unsqueeze(0),
            }
            return out
        # Otherwise, sliding window gives a list of each of the above keys for each segment - need to stack and pad
        video_list = list(map(torch.tensor, sample["video"]))
        max_frames = 0
        for v in video_list:
            max_frames = max(max_frames, v.shape[0])
        video = torch.zeros(len(video_list), max_frames, *video_list[0].shape[1:], dtype=torch.float32)
        for i, v in enumerate(video_list):
            video[i, :v.shape[0]] = v

        input_ids_list = list(map(torch.tensor, sample["input_ids"]))
        max_input_ids = 0
        for i in input_ids_list:
            max_input_ids = max(max_input_ids, i.shape[0])
        input_ids = self.pad_token_id * torch.ones(len(input_ids_list), max_input_ids, dtype=torch.int64)
        for i, ids in enumerate(input_ids_list):
            input_ids[i, :ids.shape[0]] = ids.int()

        attention_mask_list = list(map(torch.tensor, sample["attention_mask"]))
        max_attention_mask = 0
        for i in attention_mask_list:
            max_attention_mask = max(max_attention_mask, i.shape[0])
        attention_mask = torch.zeros(len(attention_mask_list), max_attention_mask, dtype=torch.int64)
        for i, mask in enumerate(attention_mask_list):
            attention_mask[i, :mask.shape[0]] = mask.int()

        vision_xattn_mask_list = list(map(torch.tensor, sample["vision_xattn_mask"]))
        max_vision_xattn_mask = 0
        for i in vision_xattn_mask_list:
            max_vision_xattn_mask = max(max_vision_xattn_mask, i.shape[0])
        vision_xattn_mask = torch.zeros(len(vision_xattn_mask_list), max_vision_xattn_mask, dtype=torch.int64)
        for i, mask in enumerate(vision_xattn_mask_list):
            vision_xattn_mask[i, :mask.shape[0]] = mask.int()
        
        loss_mask_list = list(map(torch.tensor, sample["loss_mask"]))
        max_loss_mask = 0
        for i in loss_mask_list:
            max_loss_mask = max(max_loss_mask, i.shape[0])
        loss_mask = torch.zeros(len(loss_mask_list), max_loss_mask, dtype=torch.int64)
        for i, mask in enumerate(loss_mask_list):
            loss_mask[i, :mask.shape[0]] = mask.int()

        # bc of memory issues, split this in half, usually 10 segments per video
        out1 = {
            "video": video_list[:5],
            "input_ids": input_ids_list[:5],
            "attention_mask": attention_mask_list,
            "vision_xattn_mask": vision_xattn_mask_list[:5],
            "loss_mask": loss_mask_list[:5],
        }
        out2 = {
            "video": video_list[5:],
            "input_ids": input_ids_list[5:],
            "attention_mask": attention_mask_list[5:],
            "vision_xattn_mask": vision_xattn_mask_list[5:],
            "loss_mask": loss_mask_list[5:],
        }

        return out1, out2


def save_checkpoint(model_wrapper, optimizer, step: int, epoch: int, ckpt_path: str) -> None:
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    payload = {
        "model_state": model_wrapper.model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    torch.save(payload, ckpt_path)


def load_checkpoint(model_wrapper, optimizer, ckpt_path: str) -> tuple[int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_wrapper.model.load_state_dict(ckpt["model_state"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)

def train() -> None:
    args = parse_args()
    config = _load_config(args.config)

    dataset_cfg = config["datasets"]["train"]
    training_cfg = config["training"]

    qevd = load_dataset(dataset_cfg["name"], **dataset_cfg["kwargs"])

    llama_path = config["model"]["llama2_7b_path"]
    model_kwargs = config["model"]["kwargs"]
    device_override = args.device
    model = make_model(
        llama_path,
        device=device_override,
        **model_kwargs,
    )
    model.train()

    training_dataset = WorkoutTrainingDataset(
        qevd,
        tokenizer=model.tokenizer,
        max_sequence_length=training_cfg.get("max_sequence_length"),
    )
    collator = Collator(model.tokenizer.pad_token_id or model.tokenizer.eos_token_id)
    dataloader = DataLoader(
        training_dataset,
        batch_size=training_cfg.get("batch_size", 1),
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=True,
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    start_epoch = 0
    global_step = 0
    if args.resume_from:
        start_epoch, global_step = load_checkpoint(model, optimizer, args.resume_from)
        tqdm.write(f"Resumed from {args.resume_from} at epoch {start_epoch}, step {global_step}.")

    grad_accum = training_cfg.get("gradient_accumulation_steps", 1)
    max_epochs = training_cfg.get("num_epochs", 1)
    max_steps = training_cfg.get("max_steps")
    log_every = training_cfg.get("log_every", 10)
    save_every = training_cfg.get("save_every", 1000)
    output_dir = training_cfg.get("output_dir", "./ckpts_full_workout")
    max_grad_norm = training_cfg.get("max_grad_norm", 1.0)

    device = model.device
    model_dtype = next(model.model.lang.parameters()).dtype

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    samples_seen = 0

    if max_steps:
        progress_total = max_steps
    else:
        steps_per_epoch = math.ceil(len(dataloader) / max(grad_accum, 1))
        progress_total = max_epochs * steps_per_epoch
    progress = tqdm(total=progress_total, desc="Training")
    micro_step = 0
    loss_history = []
    for epoch in range(start_epoch, max_epochs):
        for j, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                batch_size = batch["input_ids"].size(0)

                video = batch["video"].to(device=device, dtype=model_dtype, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                vision_xattn_mask = batch["vision_xattn_mask"].to(device, non_blocking=True)
                loss_mask = batch["loss_mask"].to(device, non_blocking=True)
                targets = input_ids.clone()
                targets_masked = targets.clone()
                targets_masked[~loss_mask.bool()] = IGNORE_INDEX
                outputs = model(
                    video=video,
                    input_ids=input_ids,
                    vision_xattn_mask=vision_xattn_mask,
                    attention_mask=attention_mask,
                    target_ids=targets_masked,
                )
                loss = outputs["loss"]
                loss_history.append(loss.detach().cpu().item())
                outputs = outputs["logits"].detach().cpu()
                
                if epoch == 0 and j == 0:
                    print('-' * 20)
                    print(f"Target: {model.tokenizer.batch_decode(targets)}")
                    print()
                    print(f"Output: {logits_to_text(outputs, model.tokenizer)}")
                    print('-' * 20)

                loss = loss / grad_accum
                loss.backward()

                running_loss += loss.item()
                micro_step += 1


                if micro_step % grad_accum == 0:
                    micro_step = 0
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if j % log_every == 0:
                        avg_loss = (running_loss / log_every) * grad_accum
                        tqdm.write(
                            f"Epoch {epoch} | Step {global_step} | "
                            f"Samples {samples_seen} | Loss {avg_loss:.4f}"
                        )
                        running_loss = 0.0
                        with open('train_output.txt', 'a') as f:
                            f.write(f"Epoch {epoch} | Step {global_step} | "
                                    f"Samples {samples_seen} | Loss {avg_loss:.4f}\n")
                            f.write('-' * 20)
                            f.write(f"Target: {model.tokenizer.batch_decode(targets)}\n")
                            f.write(f"Output: {logits_to_text(outputs, model.tokenizer)}\n")
                            f.write('-' * 20)
                    # For now can't save checkpoints, running low on storage
                    # if global_step % save_every == 0:
                    #     ckpt_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
                    #     save_checkpoint(model, optimizer, global_step, epoch, ckpt_path)

                    progress.update(1)

                    if max_steps and global_step >= max_steps:
                        break
                else:
                    continue

            if max_steps and global_step >= max_steps:
                break

            else:
                for k in range(2):
                    targets = []
                    outputs = []
                    batch_size = 1
                    loss = 0.0                    
                    sub_batch = batch[k]
                    for i in range(len(sub_batch["video"])):
                        video = sub_batch["video"][i].to(device=device, dtype=model_dtype, non_blocking=True).unsqueeze(0)
                        input_ids = sub_batch["input_ids"][i].to(device, non_blocking=True).unsqueeze(0)
                        attention_mask = sub_batch["attention_mask"][i].to(device, non_blocking=True).unsqueeze(0)
                        vision_xattn_mask = sub_batch["vision_xattn_mask"][i].to(device, non_blocking=True).unsqueeze(0)
                        loss_mask = sub_batch["loss_mask"][i].to(device, non_blocking=True).unsqueeze(0)
                        targets_segment = input_ids.clone()
                        targets_masked = targets_segment.clone()
                        targets_masked[~loss_mask.bool()] = IGNORE_INDEX
                        outputs_segment = model(
                            video=video,
                            input_ids=input_ids,
                            vision_xattn_mask=vision_xattn_mask,
                            attention_mask=attention_mask,
                            target_ids=targets_masked,
                        )
                        targets.append(targets_segment)
                        outputs.append(outputs_segment["logits"].detach().cpu())
                        loss += outputs_segment["loss"]/len(sub_batch["video"])

                    targets = _append_tensors_with_padding(targets)
                    # Outputs are logits
                    max_length = max(o.shape[1] for o in outputs)
                    outputs_padded = torch.zeros(len(outputs), max_length, outputs[0].shape[2]).to(outputs[0])
                    for i, o in enumerate(outputs):
                        outputs_padded[i, :o.shape[1]] = o[0]
                    outputs = outputs_padded

                    if epoch == 0 and j == 0:
                        print('-' * 20)
                        print(f"Target: {model.tokenizer.batch_decode(targets)}")
                        print()
                        print(f"Output: {logits_to_text(outputs, model.tokenizer)}")
                        print('-' * 20)

                    loss = loss / grad_accum
                    loss_history.append(loss.detach().cpu().item())
                    loss.backward()
                    running_loss += loss.item()
                    micro_step += 1
                    if micro_step % grad_accum == 0:
                        micro_step = 0
                        clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                        if j % log_every == 0:
                            avg_loss = (running_loss / log_every) * grad_accum
                            tqdm.write(
                                f"Epoch {epoch} | Step {global_step} | "
                                f"Samples {samples_seen} | Loss {avg_loss:.4f}"
                            )
                            running_loss = 0.0
                            with open('train_output.txt', 'a') as f:
                                f.write(f"Epoch {epoch} | Step {global_step} | "
                                        f"Samples {samples_seen} | Loss {avg_loss:.4f}\n")
                                f.write('-' * 20)
                                f.write(f"Target: {model.tokenizer.batch_decode(targets)}\n")
                                f.write(f"Output: {logits_to_text(outputs, model.tokenizer)}\n")
                                f.write('-' * 20)
                        running_loss = 0.0 # have to reset here because it's really 2 steps per video
            progress.update(1)

    final_ckpt = os.path.join(output_dir, "checkpoint_last.pt")
    save_checkpoint(model, optimizer, global_step, epoch, final_ckpt)
    tqdm.write(f"Training complete. Saved final checkpoint to {final_ckpt}")

    plt.plot(loss_history)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.savefig('loss_history.png')
    plt.close()


if __name__ == "__main__":
    train()
