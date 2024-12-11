# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Efficientnet Feature Extraction Script."""

import argparse
import glob
import os
from typing import Callable, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.constants import (
    EFFICIENTNET_FEATURES_DIR,
    FIT_COACH_BENCHMARK_DIR,
    FIT_COACH_TRAIN_DIR,
)
from src.utils import load_video_timestamps
from src.vision_modules.vision_model import Hypermodel


def get_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    _parser = argparse.ArgumentParser(
        description="EfficientNet Feature Extractor",
        usage="python scripts/extract_efficientnet_features.py [--options]",
    )
    _parser.add_argument(
        "-d", "--data_dir", default=None, required=True, help="Path to data directory"
    )
    _parser.add_argument(
        "-m",
        "--model_dir",
        required=True,
        help="Path to directory containing efficientnet model weights and label2int.json",
    )
    _parser.add_argument("-o", "--output_dir", default=None, help="Path to output features")
    _parser.add_argument(
        "--fit_coach_benchmark",
        action="store_true",
        help="Extract features for fit coach benchmark",
    )
    _parser.add_argument(
        "--fit_coach_dataset", action="store_true", help="Extract features for fit coach dataset"
    )
    _parser.add_argument("--overwrite", action="store_true", help="Overwrite pre-existing features")

    return _parser


def initialize_model() -> Hypermodel:
    """Initialize the EfficientNet model and load pre-trained weights"""
    hypermodel_parameters = {
        "num_global_classes": 3031,
        "num_frames_required": 4,
        "path_weights": os.path.join(args.model_dir, "efficientnet4Lite_1.8.3.checkpoint"),
        "gpus": [0],
        "half_precision": False,
    }

    _model = Hypermodel(**hypermodel_parameters)
    _model.initialize()

    return _model


def _register_features_hook() -> Dict:
    """Register a forward hook on last conv before relu activation"""
    _features = {}

    def get_features(name: str) -> Callable:
        """Apply a hook to extract features from named layer

        :param name:
            Layer to hook
        :return:
            Layer hookd
        """

        def hook(_, __, output: torch.Tensor):
            """Feature extraction hook

            :param _:
                Unused
            :param __:
                Unused
            :param output:
                Output tensor to extract features from
            """
            _features[name] = output.detach()

        return hook

    model.net.module[0].cnn[31][0].register_forward_hook(get_features("last_conv_before_relu"))

    return _features


def _downscale(frame: np.ndarray, max_length: int) -> np.ndarray:
    """
    Resize <frame> if either of its sides exceeds <max_length>. We preserve the aspect ratio of the
    frame. The given frame can be in portrait or landscape orientation. In both cases, we match
    the larger side with <max_length>

    Example:
        1. given max_length=640
           frame is w=1280, h=720
           and will be downscaled to w=640, h=360.

        2. given max_length=640
           frame is w=360, h=640
           and will not be rescaled.

    :param frame: numpy.array
        A frame to downscale
    :param max_length: int
        The target length of the longer side of the frame

    :return: numpy.array
        Resulting frame with the longer side not exceeding <max_length>

    """

    height, width, _ = frame.shape

    ratio = max_length / max(height, width)

    if ratio < 1:  # dont upscale
        target_size = (int(width * ratio), int(height * ratio))
        frame = cv2.resize(frame, target_size)

    return frame


def _resize(frame: np.ndarray) -> np.ndarray:
    """
    Resize a frame to not exceed self.resolution.
    The idea is to constrain the size of frames so that we do not accidentally send huge
    amounts of data through the queues that we won't need on the other end.

    Incoming frames could be in portrait or landscape orientation.
    To limit them in size, we take the larger side of <self.resolution>.
    Imagine a square with edge size being this value.
    We then resize the frame to fit into this square.
    This is not the most precise constraint, but it fulfills the
    requirement of making frames smaller, but still large enough for
    visualisation, video recording and the model.


    :param frame: numpy.array
        Single frame to resize

    :return: numpy.array
        Downscaled frame
    """
    height, width, _ = frame.shape
    max_length = max(height, width)
    frame = _downscale(frame, max_length)
    return frame


def _rotate(frame: np.ndarray) -> np.ndarray:
    """Rotate frame by 90 degrees and square pad with black borders.

    :param frame: numpy.array
        Single frame to rotate

    :return: numpy.array
        Frame rotated 90 degrees clockwise and squared-padded
    """
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width, _ = frame.shape
    square_size = max(height, width)

    pad_top = int((square_size - frame.shape[0]) / 2)
    pad_bottom = square_size - frame.shape[0] - pad_top
    pad_left = int((square_size - frame.shape[1]) / 2)
    pad_right = square_size - frame.shape[1] - pad_left
    frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
    return frame


def _process_video(filepath: str, fps_out: int = 16) -> np.ndarray:
    """Sample frames from video at fps_out (16) FPS and return features extracted at 4 Hz"""
    vidcap = cv2.VideoCapture(filepath)
    assert vidcap.isOpened()

    video_ts = load_video_timestamps(filepath.replace(".mp4", "_timestamps.npy"))
    video_ts_normalized = video_ts - video_ts[0]

    file_idx = int(filepath.split("/")[-1].split(".")[0])

    index_in = -1
    features_ds = []
    features_timestamps = []

    next_frame_timestamp = 0
    while True:
        success = vidcap.grab()
        if not success:
            break
        index_in += 1
        if video_ts_normalized[index_in] >= next_frame_timestamp:
            success, frame = vidcap.retrieve()

            # Ensure all videos are vertical
            frame = _resize(frame)
            if file_idx <= 145:
                frame = _rotate(frame)

            next_frame_timestamp += 1.0 / fps_out

            if not success:
                break

            if index_in == 0:
                # Repeat first frame to load internal model buffers
                for _ in range(12):
                    model(frame)

            out = model(frame)
            if out is not None:
                features_timestamps.append(video_ts[index_in])
                features_ds.append(features["last_conv_before_relu"].detach().cpu().numpy())
    return features_ds, np.array(features_timestamps, dtype=np.float64)


def _extract_features() -> None:
    """Loop through each video in the given directory and save extracted features"""

    all_filepaths = glob.glob(os.path.join(video_dir, "*.mp4"))

    pbar = tqdm(enumerate(all_filepaths), total=len(all_filepaths))

    for _, filepath in pbar:
        # Features file path
        filename = os.path.basename(filepath).split(".")[0]
        features_path = os.path.join(features_dir, filename + "_features.npy")
        feature_timestamps_path = os.path.join(features_dir, filename + "_timestamps.npy")

        if os.path.exists(features_path):
            if args.overwrite:
                print(f"Deleting; file already exists - {filename}")
                os.remove(features_path)
            else:
                print(f"Skipping; file already exists - {filename}")
                continue

        # Extract features
        features_ds, features_ts = _process_video(filepath)

        # Save features
        with open(features_path, "wb") as f:
            np.save(f, np.concatenate(features_ds, axis=0))
        with open(feature_timestamps_path, "wb") as f:
            np.save(f, features_ts)


if __name__ == "__main__":
    # Parse CLI arguments
    parser = get_parser()
    args = parser.parse_args()

    # Initialize model
    model = initialize_model()

    # Register feature extraction hook
    features = _register_features_hook()

    # Extract features from videos in data dir
    if not args.fit_coach_benchmark or args.fit_coach_dataset:
        features_dir = args.output_dir or os.path.join(args.data_dir, EFFICIENTNET_FEATURES_DIR)
        video_dir = args.data_dir
        os.makedirs(features_dir, exist_ok=True)
        _extract_features()

    # Extract FIT-COACH benchmark features
    if args.fit_coach_benchmark:
        model.reset_buffer()
        features_dir = os.path.join(
            args.data_dir, FIT_COACH_BENCHMARK_DIR, EFFICIENTNET_FEATURES_DIR, "long_range_videos"
        )
        os.makedirs(features_dir, exist_ok=True)
        video_dir = os.path.join(args.data_dir, FIT_COACH_BENCHMARK_DIR, "long_range_videos")
        _extract_features()

    # Extract FIT-COACH dataset features
    if args.fit_coach_dataset:
        model.reset_buffer()
        features_dir = os.path.join(
            args.data_dir, FIT_COACH_TRAIN_DIR, EFFICIENTNET_FEATURES_DIR, "long_range_videos"
        )
        os.makedirs(features_dir, exist_ok=True)
        video_dir = os.path.join(args.data_dir, FIT_COACH_TRAIN_DIR, "long_range_videos")
        _extract_features()
