"""
manual_check_full_pipeline.py

Verifies the FULL pipeline actually reduces jitter, by re-running motion
estimation on both the shaky input and the stabilized output, and
comparing how much frame-to-frame motion (jitter) remains in each.

Run with: python tests/manual_check_full_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stabilizer import io_utils, motion_estimation

def jitter_score(path):
    motions = list(motion_estimation.compute_frame_to_frame_motion(io_utils.frame_generator(path)))
    motions = np.array(motions)
    return np.std(motions[:, 0]), np.std(motions[:, 1])

before = jitter_score("shaky_test.mp4")
after = jitter_score("stable_test.mp4")

print(f"BEFORE stabilization -> dx std: {before[0]:.2f}px, dy std: {before[1]:.2f}px")
print(f"AFTER  stabilization -> dx std: {after[0]:.2f}px, dy std: {after[1]:.2f}px")
print(f"Jitter reduction: dx {100*(1-after[0]/before[0]):.1f}%, dy {100*(1-after[1]/before[1]):.1f}%")