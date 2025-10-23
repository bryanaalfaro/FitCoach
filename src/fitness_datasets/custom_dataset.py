"""
Custom Dataset Loading Helper Function. At the moment, this is used to load
a custom dataset that only contains video and timestamp files (crucially, no
feedback json file such as the one in the QEVD dataset).
"""

import os
import glob
from datasets import Dataset


def workout_system_prompt():
    """System prompt for a general, unstructured workout."""
    return (
        "You are an expert fitness coaching AI guiding a user through a workout and "
        "coaching them as they exercise. Your main goal is to identify the user's current exercise and"
        "provide feedback on their performance. You may see multiple users in a video, in which case you"
        "should pick a user and focus on them."
    )

def load_custom_dataset(data_root: str, **kwargs) -> Dataset:
    """
    Load the custom dataset.
    """
    video_files = glob.glob(os.path.join(data_root, "*.mp4"))
    video_timestamps = glob.glob(os.path.join(data_root, "*.npy"))
    efficientnet_features = glob.glob(os.path.join(data_root, "efficientnet_features", "*features.npy"))
    efficientnet_timestamps = glob.glob(os.path.join(data_root, "efficientnet_features", "*timestamps.npy"))
    system = [workout_system_prompt() for _ in range(len(video_files))]
    assert len(video_files) == len(video_timestamps) == len(efficientnet_features) == len(efficientnet_timestamps) == len(system)
    dataset = Dataset.from_dict({
        "video_path": video_files,
        "video_timestamps_path": video_timestamps,
        "efficientnet_features_path": efficientnet_features,
        "efficientnet_timestamps_path": efficientnet_timestamps,
        "system": system,
    })
    return dataset