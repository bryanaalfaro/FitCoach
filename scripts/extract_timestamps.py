"""
Helper script to extract timestamps from a video file. Helpful when using a video not from the
QEVD dataset.
"""

import argparse
import glob
import os
import cv2
import numpy as np

def get_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    _parser = argparse.ArgumentParser(
        description="Extract Timestamps from Video",
        usage="python scripts/extract_timestamps.py [--options]",
    )
    _parser.add_argument(
        "-d", "--data_dir", required=True, help="Path to data directory"
    )
    return _parser

def load_video_timestamps(file_path: str) -> np.array:
    """ Return array of timestamps per video frame (in seconds) """
    vid = cv2.VideoCapture(file_path)
    assert vid.isOpened()
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid.release()
    return np.arange(0, frame_count) / fps

def extract_timestamps(data_dir: str) -> None:
    """Extract timestamps from all video files in the data directory."""
    all_filepaths = glob.glob(os.path.join(data_dir, "*.mp4"))
    for filepath in all_filepaths:
        video_ts = load_video_timestamps(filepath)
        np.save(filepath.replace(".mp4", "_timestamps.npy"), video_ts)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    extract_timestamps(args.data_dir)