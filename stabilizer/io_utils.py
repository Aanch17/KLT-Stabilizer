"""
io_utils.py

Handles all video reading/writing and input validation, so the rest of the
pipeline can work purely in terms of frames and numbers, and so all our
error handling / logging setup lives in one place.
"""

import os
import cv2
import logging


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def validate_input_path(path):
    """
    Raise a clear, specific error if the input path is unusable, instead
    of letting OpenCV fail silently later with a confusing error.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input video not found: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Input path is not a file: {path}")


def open_video(path):
    """
    Open a video file and return the cv2.VideoCapture object, raising a
    clear error if the file couldn't be read as a video (e.g. corrupted
    file, unsupported codec).
    """
    validate_input_path(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(
            f"Could not open '{path}' as a video. "
            "It may be corrupted or in an unsupported format."
        )
    return cap


def get_video_properties(path):
    """
    Return (fps, width, height, frame_count) for a video file.
    """
    cap = open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        fps = 30.0  # sensible fallback for videos with missing metadata

    if frame_count < 2:
        raise ValueError(
            f"Video '{path}' has too few frames ({frame_count}) to stabilize. "
            "Need at least 2 frames."
        )

    return fps, width, height, frame_count


def frame_generator(path):
    """
    Yield BGR frames from a video file, one at a time, in order.
    Using a generator (rather than loading the whole video into memory)
    keeps memory use flat regardless of video length.
    """
    cap = open_video(path)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


class VideoWriter:
    """
    Thin wrapper around cv2.VideoWriter with sane defaults, used as a
    context manager so the output file is always properly closed.
    """

    def __init__(self, path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise IOError(f"Could not open output file for writing: {path}")

    def write(self, frame):
        self.writer.write(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.release()