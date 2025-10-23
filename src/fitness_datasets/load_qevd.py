# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""General Dataset Loading Helper Function."""

from datasets import Dataset

from .fitcoach import load_fit_coach_dataset
from .custom_dataset import load_custom_dataset


def load_dataset(dataset_name: str, **kwargs: dict) -> Dataset:
    """Load the requested dataset"""
    match dataset_name:
        case "fitcoach-dataset":
            return load_fit_coach_dataset(**kwargs)
        case "custom-dataset":
            return load_custom_dataset(**kwargs)
        case _:
            raise NotImplementedError(f"Dataset: {dataset_name}, not found.")