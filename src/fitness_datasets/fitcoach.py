# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""QEVD-Fit-Coach Dataset and Benchmark Preparation Script."""

import json
import os
from collections import defaultdict
from typing import Any
import random
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
    ), f"The number of feedbacks {len(feedbacks)} must equal the number of timestamps {len(feedback_timestamps)}."

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


def prepare_sliding_window_segments(
    record: dict,
    feedback_spans: list[tuple[str, int, int]],
    window_end_indices: list[int],
    video_timestamps: list[float],
) -> list[dict]:
    """
    Extract individual exercise segments by splitting workout videos based on transition feedbacks.

    :param record:
        Dictionary containing responses for the full workout.
    :param feedback_spans:
        List of (feedback, start index, end index) Tuples for each feedback.
    :param window_end_indices:
        End indices of each sliding window.
    :param video_timestamps:
        Timestamps of each video frame.

    :return:
        A list of dictionaries representing individual sliding window segments. Each dictionary
        contains responses, response timestamps, and other relevant information
        for each window.
    """

    feedback_timestamps = record["feedback_timestamps"]

    segments = []
    segment_feedbacks = []
    segment_feedback_timestamps = []
    window_idx = 0
    timestamp_start_idxs = [0] + list(window_end_indices[:-1]-1) # -1 because end is exclusive
    timestamp_start = video_timestamps[timestamp_start_idxs]
    timestamp_end_idxs = list(window_end_indices[:-1]) + [len(video_timestamps)-1]
    timestamp_end = video_timestamps[timestamp_end_idxs]
    feedback_counter = 0
    # Loop over all feedbacks and construct the dictionary for each window
    for feedback in feedback_spans:
        window_end_idx = window_end_indices[window_idx]
        feedback_start_idx = feedback[1]
        if feedback_start_idx >= window_end_idx:
            # move to next window and create the dictionary for the window that just ended
            # NOTE: because of the logic of this loop, the last window will not be created here,
            # needs to be done after the loop
            segments.append(
                {
                    "responses": segment_feedbacks,
                    "response_timestamps": segment_feedback_timestamps,
                    "exercise_start_timestamp": timestamp_start[window_idx],
                    "exercise_end_timestamp": timestamp_end[window_idx],
                }
            )
            segment_feedbacks = []
            segment_feedback_timestamps = []
            window_idx += 1
        
        segment_feedbacks.append(feedback[0])
        segment_feedback_timestamps.append(feedback_timestamps[feedback_counter])
        feedback_counter += 1

    # add the last window
    segments.append(
        {
            "responses": segment_feedbacks,
            "response_timestamps": segment_feedback_timestamps,
            "exercise_start_timestamp": timestamp_start[window_idx],
            "exercise_end_timestamp": timestamp_end[window_idx],
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
        "You are an expert fitness coaching AI guiding a user through a workout and "
        "coaching them as they exercise. You assess their performance, count repetitions, and "
        "proactively provide feedback.\n"
        "Guide the user through the following exercises: "
        f"{'; '.join([str(idx + 1) + ') ' + exercise for idx, exercise in enumerate(exercises)])}."
        "\nEach exercise should be done for 30 seconds."
    )

def sliding_window_system_prompt() -> str:
    """System prompt for sliding window workouts."""
    return (
        "You are an expert fitness coaching AI assisting a user with their workout and "
        "coaching them as they exercise. You assess their performance, count repetitions, and "
        "proactively provide feedback. Please also specify whenever you identify a transition between exercises.\n"
    )

def load_fit_coach_dataset(
    data_root: str, split: str = "test", eval_mode: str = "single_exercise", max_num_videos: int = None, shuffle_videos: bool = False, sliding_window_length: int = 20
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
    :param max_num_videos:
        Maximum number of videos to load.
    :param shuffle_videos:
        Whether to shuffle the videos. This is for when max_num_videos is specified, so that we can sample a random subset of the videos
        instead of loading the first max_num_videos videos. If max_num_videos is not specified, this is ignored.
    :param sliding_window_length:
        If using sliding window mode, this is the length of the sliding window in seconds.
    :return:
        A Hugging Face Datasets object.
    """

    data_dir = os.path.join(
        data_root, FIT_COACH_BENCHMARK_DIR if split == "test" else FIT_COACH_TRAIN_DIR
    )

    records_file = os.path.join(data_dir, "feedbacks_long_range.json")
    with open(records_file, "r", encoding="utf-8") as f:
        workout_records = json.load(f)

    if shuffle_videos:
        random.shuffle(workout_records)

    if max_num_videos is not None: 
        workout_records = workout_records[:max_num_videos]


    # Create an empty dict to store processed records
    dataset = defaultdict(list)

    for orig_record in tqdm(workout_records):
        # Fix exercise name
        orig_record["feedbacks"] = [
            feedback.replace("armcrosschest", "deltoid stretch")
            for feedback in orig_record["feedbacks"]
        ]

        # Get feedback string and temporal span indices
        try:
            feedback_spans = get_feedback_span(
                orig_record["feedbacks"], orig_record["feedback_timestamps"]
            )
        except AssertionError as e:
            print(f"[WARNING]: Error processing record {orig_record['long_range_video_file']}: {e}")
            continue
            

        # Find transition feedback indices
        transition_indices = np.where(np.array(orig_record["is_transition"]))[0].tolist()

        # Get list of exercises in workout - ignore final transition feedback marking session end
        exercises = [extract_exercise_name(feedback_spans[ii][0]) for ii in transition_indices][:-1]

        # Split workout videos into single exercise segments
        match eval_mode:
            case "single_exercise":
                updated_records = prepare_single_exercise_segments(
                    orig_record, exercises, feedback_spans, transition_indices, data_dir
                )
            case "full_workout":
                updated_records = [
                    {
                        "system": workout_system_prompt(exercises),
                        "responses": [xx[0] for xx in feedback_spans],
                        "response_timestamps": orig_record["feedback_timestamps"],
                    }
                ]
            case "full_workout_sliding_window":
                # Very similar to single exercise mode, but instead of having one separate entry for each exercise,
                # store it as a single entry for the entire workout
                video_timestamps = load_video_timestamps(os.path.join(data_dir, orig_record["video_timestamps"]))
                video_length = video_timestamps[-1] - video_timestamps[0]
                n_frames = len(video_timestamps)
                fps = n_frames / video_length
                sliding_window_length_frames = int(sliding_window_length * fps) # must be an int so windows may not be exactly the desired duration, but will be close
                num_windows = int(video_length // sliding_window_length) + 1# number of full-length windows, +1 to include last potentially shorter window
                # end indices of each window, last index will probably be greater than n_frames, but that's okay because it's for the partial last window
                window_end_indices = np.arange(1, num_windows+1) * sliding_window_length_frames
                segments = prepare_sliding_window_segments(
                    orig_record, feedback_spans, window_end_indices, video_timestamps
                )
                updated_records = [
                    {
                        "system": sliding_window_system_prompt(),
                        "responses": [xx[0] for xx in feedback_spans],
                        "response_timestamps": orig_record["feedback_timestamps"],
                        "segments" : segments
                    }
                ]
            case _:
                raise ValueError(f"Invalid eval_mode: {eval_mode}")

        # Update dataset with processed segments
        update_dataset(dataset, updated_records, orig_record, data_dir)

    return Dataset.from_dict(dataset)
