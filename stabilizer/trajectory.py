"""
trajectory.py

Turns the per-frame (dx, dy, da) motion estimates from motion_estimation.py
into:
    1. The raw camera trajectory (cumulative sum of the motions) -- this is
       the actual, shaky path the camera took through the video.
    2. A smoothed version of that trajectory -- the path the camera
       "should" have taken if it had been held perfectly steady.

The difference between (2) and (1), frame by frame, is exactly the
correction that frame_correction.py needs to apply.
"""

import numpy as np


def accumulate_trajectory(motions):
    """
    Given a list/array of (dx, dy, da) frame-to-frame motions, compute the
    cumulative trajectory: trajectory[i] = sum of all motions up to frame i.

    motions: array-like of shape (N, 3)
    returns: numpy array of shape (N, 3)
    """
    motions = np.asarray(motions, dtype=np.float64)
    trajectory = np.cumsum(motions, axis=0)
    return trajectory


def smooth_trajectory(trajectory, radius=15):
    """
    Smooth the trajectory with a simple moving-average filter.

    For each point, we average all trajectory values within `radius`
    frames on either side. A larger radius means a smoother (but less
    responsive) result -- radius controls the trade-off between
    "very stable" and "still follows intentional camera movement".

    trajectory: numpy array of shape (N, 3)
    radius: int, half-width of the averaging window in frames
    returns: numpy array of shape (N, 3), same shape as input
    """
    n_frames = trajectory.shape[0]
    smoothed = np.copy(trajectory)

    window_size = 2 * radius + 1

    # Pad the trajectory at both ends by repeating the edge values, so the
    # moving average is well-defined even near the start/end of the video.
    padded = np.pad(trajectory, ((radius, radius), (0, 0)), mode="edge")

    kernel = np.ones(window_size) / window_size

    for axis in range(trajectory.shape[1]):  # dx, dy, da independently
        smoothed[:, axis] = np.convolve(padded[:, axis], kernel, mode="valid")[:n_frames]

    return smoothed


def compute_corrections(trajectory, smoothed_trajectory):
    """
    Compute the per-frame ABSOLUTE correction transform needed to turn the
    raw, shaky trajectory into the smoothed trajectory.

    Frame (i+1)'s raw pixel content naturally sits at cumulative position
    trajectory[i] (how far it has drifted from frame 0, per the tracked
    motion). We want it to instead appear at smoothed_trajectory[i]. The
    shift that accomplishes this is simply the difference between the two:

        correction[i] = smoothed_trajectory[i] - trajectory[i]

    Note this does NOT include the raw motion itself -- adding it back in
    would double-count the frame's own natural displacement and amplify
    jitter instead of cancelling it.

    returns: numpy array of shape (N, 3)
    """
    corrections = smoothed_trajectory - trajectory
    return corrections