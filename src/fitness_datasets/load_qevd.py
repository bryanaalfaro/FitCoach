# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""General Dataset Loading Helper Function."""

from datasets import Dataset

from .fitcoach import load_fit_coach_dataset


def load_dataset(dataset_name: str, **kwargs: dict) -> Dataset:
    """Load the requested dataset"""
    if dataset_name == "fitcoach-dataset":
        return load_fit_coach_dataset(**kwargs)

    raise NotImplementedError(f"Dataset: {dataset_name}, not found.")
