"""
motion_estimation.py

Estimates the frame-to-frame camera motion of a video using classical
feature tracking (Shi-Tomasi corner detection + Lucas-Kanade optical flow),
as covered in CSE3010 Module 4 (Motion Analysis).

The core idea:
    1. Detect a set of "good" corner points in the current frame.
    2. Track where those exact points move to in the next frame using
       KLT optical flow.
    3. From the matched point pairs (before, after), estimate the single
       affine transform (translation + rotation + slight scale) that best
       explains how the whole frame moved.

This gives one (dx, dy, da) motion estimate per frame transition, which
trajectory.py will later accumulate and smooth.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("stabilizer.motion")

# Parameters for Shi-Tomasi corner detection (cv2.goodFeaturesToTrack)
FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=30,
    blockSize=3,
)

# Parameters for Lucas-Kanade optical flow (cv2.calcOpticalFlowPyrLK)
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

MIN_TRACKED_POINTS = 10  # below this, we consider tracking to have failed


def detect_features(gray_frame):
    """
    Detect good-to-track corner points in a grayscale frame.

    Returns an (N, 1, 2) float32 array of point coordinates, or None if
    no features could be found (e.g. a completely blank frame).
    """
    points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **FEATURE_PARAMS)
    return points


def track_features(prev_gray, curr_gray, prev_points):
    """
    Track prev_points from prev_gray into curr_gray using KLT optical flow.

    Returns (good_prev_points, good_curr_points) -- only the points that
    were tracked successfully, matched up pair-for-pair.
    """
    curr_points, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **LK_PARAMS
    )

    status = status.reshape(-1)
    good_prev = prev_points[status == 1]
    good_curr = curr_points[status == 1]

    return good_prev, good_curr


def estimate_transform(prev_points, curr_points):
    """
    Estimate the affine transform (translation + rotation + uniform scale)
    that best maps prev_points onto curr_points, using a robust estimator
    (RANSAC under the hood via cv2.estimateAffinePartial2D).

    Returns (dx, dy, da) where:
        dx, dy -- translation in pixels
        da     -- rotation angle in radians

    Returns None if a transform could not be estimated (too few points).
    """
    if prev_points is None or curr_points is None:
        return None
    if len(prev_points) < MIN_TRACKED_POINTS:
        return None

    transform_matrix, _inliers = cv2.estimateAffinePartial2D(prev_points, curr_points)

    if transform_matrix is None:
        return None

    dx = transform_matrix[0, 2]
    dy = transform_matrix[1, 2]
    da = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

    return dx, dy, da


def compute_frame_to_frame_motion(frame_generator):
    """
    Given a generator/iterable of BGR frames (in order), compute the
    (dx, dy, da) motion for every consecutive frame pair.

    Yields one (dx, dy, da) tuple per transition (so for N frames, this
    yields N-1 tuples). If tracking fails on a given frame, features are
    re-detected fresh from that frame and the motion for that transition
    is reported as (0, 0, 0) with a warning logged -- this keeps the
    pipeline robust to occasional bad frames instead of crashing.
    """
    prev_gray = None
    prev_points = None

    for frame in frame_generator:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            prev_points = detect_features(prev_gray)
            continue

        if prev_points is None or len(prev_points) < MIN_TRACKED_POINTS:
            prev_points = detect_features(prev_gray)

        if prev_points is None:
            logger.warning("No trackable features found in a frame; reporting zero motion.")
            yield (0.0, 0.0, 0.0)
            prev_gray = gray
            continue

        good_prev, good_curr = track_features(prev_gray, gray, prev_points)
        transform = estimate_transform(good_prev, good_curr)

        if transform is None:
            logger.warning("Motion estimation failed for a frame pair; reporting zero motion.")
            transform = (0.0, 0.0, 0.0)

        yield transform

        # Prepare for next iteration -- re-detect features fresh each time
        # for robustness (cheap, and avoids drift from stale point sets).
        prev_gray = gray
        prev_points = detect_features(prev_gray)