# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Stream-VLM Baseline evaluation script for QEVD-Fit-Coach-Benchmark."""

import argparse

import yaml

from src.evaluation_helpers import evaluate_model
from src.fitness_datasets import load_dataset
from src.model_helpers import make_model

if __name__ == "__main__":
    # Parse command line arguments
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument(
        "--config", type=str, required=True, help="Path to the yaml config file."
    )
    config_args, _ = config_parser.parse_known_args()

    # Parse the config dict
    with open(config_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load and prepare datasets
    eval_dataset_name = config["datasets"]["test"]["name"]
    eval_kwargs = config["datasets"]["test"]["kwargs"]
    dataset = load_dataset(eval_dataset_name, **eval_kwargs)

    # Load model
    llama2_7b_path = config["model"]["llama2_7b_path"]
    model_kwargs = config["model"]["kwargs"]
    model = make_model(llama2_7b_path, **model_kwargs)

    # Run evaluator
    evaluator_name = config["evaluator"]["name"]
    evaluator_kwargs = config["evaluator"]["kwargs"]
    sampling_kwargs = config["evaluator"]["sampling_kwargs"]
    evaluate_model(model, dataset, evaluator_name, evaluator_kwargs, sampling_kwargs)
