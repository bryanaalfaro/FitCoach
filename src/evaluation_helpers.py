# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""Evaluator Loading Helper Functions."""

from typing import Callable, Union

from datasets import Dataset
from torch import nn

from src.evaluators import InteractiveFeedbackEvaluator, VideoOnlyEvaluator
from src.model_wrappers import BaseVLModelWrapper


def get_evaluator(evaluator_name: str) -> Callable:
    """
    :return:
        An evaluator of type VisionLanguageEvaluator.
    """
    match evaluator_name:
        case "interactive_feedback_evaluator":
            return InteractiveFeedbackEvaluator
        case "video_only_evaluator":
            return VideoOnlyEvaluator
        case _:
            raise NotImplementedError(f"Evaluator: {evaluator_name}, not found.")


def evaluate_model(
    model: Union[nn.Module, BaseVLModelWrapper],
    dataset: Dataset,
    evaluator_name: str,
    evaluator_kwargs: dict,
    sampling_kwargs: dict,
):
    """Evaluate a model by loading an evaluator

    :param model:
        The model to be evaluated
    :param dataset:
        The dataset for evaluation
    :param evaluator_name:
        The name of the evaluator class (currently only "interactive_feedback_evaluator" is
        supported)
    :param evaluator_kwargs:
        kwargs to be passed to the evaluator
    :param sampling_kwargs:
        kwargs to be passed to the generation call
    """
    evaluator = get_evaluator(evaluator_name)
    evaluator = evaluator(model, dataset, **evaluator_kwargs)
    evaluator(**sampling_kwargs)
