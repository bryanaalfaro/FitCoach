# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""QEVD-Fit-Coach Dataset and Benchmark Preparation Script."""

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from src.constants import (
    EFFICIENTNET_FEATURES_DIR,
    FIT_COACH_BENCHMARK_DIR,
    FIT_COACH_TRAIN_DIR,
)
from src.utils import load_video_timestamps


def get_feedback_span(
    feedbacks: list[str], feedback_timestamps: list[float]
) -> list[tuple[str, int, int]]:
    """
    Return a list of tuples including the feedback, start index, and end index, from a dense list of
    frame-aligned feedbacks.

    For example, given: ['', '', 'feedback_1', 'feedback_1', '', '', '', 'feedback_2',
    'feedback_2', '', ...].
    Return: [('feedback_1', 2, 3), ('feedback_2', 7, 8), ...]

    :param feedbacks:
        Dense list of feedbacks aligned with each video frame. Repeated feedbacks across frames
        represent feedback duration.
    :param feedback_timestamps:
        Timestamps for each feedback.

    :return:
        A list of tuples (string, int, int).
    """

    feedback_spans = []

    current_feedback = None
    start_idx = None
    for ii, feedback in enumerate(feedbacks):
        # Store the feedback and its start and end indices when encountering a new feedback
        if current_feedback is not None and current_feedback != feedback:
            feedback_spans.append((current_feedback, start_idx, ii - 1))
            current_feedback = None
            start_idx = None

        # If there's no current feedback, initialize it when encountering one
        if feedback and not current_feedback:
            current_feedback = feedback
            start_idx = ii

    assert len(feedback_spans) == len(
        feedback_timestamps
    ), "The number of feedbacks must equal the number of timestamps."

    return feedback_spans


def update_dataset(
    dataset: defaultdict[Any, list], updated_records: list[dict], record: dict, data_dir: str
) -> None:
    """
    Update processed segments with additional metadata and add to the dataset

    :param dataset:
        Dictionary to store final dataset records
    :param updated_records:
        List of updated records (single exercises or full workout)
    :param record:
        Original record corresponding to the full workout
    :param data_dir:
        Directory where data is stored
    """

    # Path to efficientnet features. REVIEW: Change name
    features_path = os.path.join(
        data_dir, EFFICIENTNET_FEATURES_DIR, record["long_range_video_file"]
    ).replace(".mp4", "_features.npy")

    for data_dict in updated_records:
        # Update paths
        data_dict.update(
            efficientnet_features_path=features_path,
            efficientnet_timestamps_path=features_path.replace("_features.npy", "_timestamps.npy"),
            video_path=os.path.join(data_dir, record["long_range_video_file"]),
            video_timestamps_path=os.path.join(data_dir, record["video_timestamps"]),
        )

        # Add to dataset
        for kk in data_dict:
            dataset[kk].append(data_dict[kk])


def prepare_single_exercise_segments(
    record: dict,
    exercises: list[str],
    feedback_spans: list[tuple[str, int, int]],
    transition_indices: list[int],
    data_dir: str,
) -> list[dict]:
    """
    Extract individual exercise segments by splitting workout videos based on transition feedbacks.

    :param record:
        Dictionary containing responses for the full workout.
    :param exercises:
        List of ordered exercises in the workout
    :param feedback_spans:
        List of (feedback, start index, end index) Tuples for each feedback.
    :param transition_indices:
        Indices where transitions occur between exercises.
    :param data_dir:
        Directory where data is stored.

    :return:
        A list of dictionaries representing individual exercise segments. Each dictionary
        contains system prompts responses, response timestamps, and other relevant information
        for each exercise.
    """
    # Load video timestamps
    video_timestamps = load_video_timestamps(os.path.join(data_dir, record["video_timestamps"]))
    print(f"Video timestamps file: {os.path.join(data_dir, record['video_timestamps'])}")
    feedback_timestamps = record["feedback_timestamps"]

    segments = []
    # Loop over all exercises transitions
    for ii in range(len(transition_indices) - 1):
        current_transition = feedback_spans[transition_indices[ii]]
        next_transition = feedback_spans[transition_indices[ii + 1]]

        # Start segment at the beginning of current transition feedback
        start_idx = current_transition[1]

        # End segment video halfway through the next transition feedback
        # (ignore the transition feedback)
        end_idx_video = next_transition[1] + (next_transition[2] - next_transition[1]) // 2

        # Update data_dict with exercise segment
        segment_feedbacks = [
            xx[0]
            for xx in feedback_spans[(transition_indices[ii] + 1) : transition_indices[ii + 1]]
        ]

        segment_feedback_timestamps = feedback_timestamps[
            (transition_indices[ii] + 1) : transition_indices[ii + 1]
        ]

        if (transition_indices[ii + 1] - transition_indices[ii]) > 1:
            segments.append(
                {
                    "exercise_name": exercises[ii],
                    "exercise_start_timestamp": video_timestamps[start_idx],
                    "exercise_end_timestamp": video_timestamps[end_idx_video],
                    "system": single_exercise_system_prompt(exercises[ii]),
                    "responses": segment_feedbacks,
                    "response_timestamps": segment_feedback_timestamps,
                }
            )

    return segments


def extract_exercise_name(transition_feedback: str) -> str:
    """Extract exercise name from a transition feedback.

    Example: First up are squats! => squats
    """
    return (
        transition_feedback.replace("First up are ", "")
        .replace("Moving on to ", "")
        .replace("!", "")
    )


def single_exercise_system_prompt(exercise_name: str) -> str:
    """System prompt for single exercise segments."""
    return (
        f"You are an expert fitness coaching AI who coaches users as they exercise. "
        f"You assess their performance, "
        f"count repetitions, and proactively provide feedback. "
        f"The user should be doing {exercise_name}."
    )


def workout_system_prompt(exercises: list[str]) -> str:
    """System prompt for multi-exercise workouts."""
    return (
        f"You are an expert fitness coaching AI guiding a user through a workout and "
        f"coaching them as they exercise. You assess their performance, count repetitions, and "
        f"proactively provide feedback.\n"
        f"Guide the user through the following exercises: "
        f"{'; '.join([str(idx + 1) + ') ' + exercise for idx, exercise in enumerate(exercises)])}."
        f"\nEach exercise should be done for 30 seconds."
    )


def load_fit_coach_dataset(
    data_root: str, split: str = "test", eval_mode: str = "single_exercise"
) -> Dataset:
    """Loads the QEVD-FIT-Coach training or benchmark dataset.

    :param data_root:
        Root directory of the dataset.
    :param split:
        Split to load. Options: "train", "test".
        "test" will load the QEVD-FIT-Coach-Benchmark dataset.
    :param eval_mode:
        Mode to use for evaluation ("full_workout" or "single_exercise").
        Defaults to "single_exercises".

    :return:
        A Hugging Face Datasets object.
    """

    data_dir = os.path.join(
        data_root, FIT_COACH_BENCHMARK_DIR if split == "test" else FIT_COACH_TRAIN_DIR
    )

    records_file = os.path.join(data_dir, "feedbacks_long_range.json")
    with open(records_file, "r", encoding="utf-8") as f:
        workout_records = json.load(f)

    # Create an empty dict to store processed records
    dataset = defaultdict(list)

    for orig_record in tqdm(workout_records):
        # Fix exercise name
        orig_record["feedbacks"] = [
            feedback.replace("armcrosschest", "deltoid stretch")
            for feedback in orig_record["feedbacks"]
        ]

        # Get feedback string and temporal span indices
        feedback_spans = get_feedback_span(
            orig_record["feedbacks"], orig_record["feedback_timestamps"]
        )

        # Find transition feedback indices
        transition_indices = np.where(np.array(orig_record["is_transition"]))[0].tolist()

        # Get list of exercises in workout - ignore final transition feedback marking session end
        exercises = [extract_exercise_name(feedback_spans[ii][0]) for ii in transition_indices][:-1]

        # Split workout videos into single exercise segments
        if eval_mode == "single_exercise":
            updated_records = prepare_single_exercise_segments(
                orig_record, exercises, feedback_spans, transition_indices, data_dir
            )

        # Keep full workout
        else:
            updated_records = [
                {
                    "system": workout_system_prompt(exercises),
                    "responses": [xx[0] for xx in feedback_spans],
                    "response_timestamps": orig_record["feedback_timestamps"],
                }
            ]

        # Update dataset with processed segments
        update_dataset(dataset, updated_records, orig_record, data_dir)

    return Dataset.from_dict(dataset)
