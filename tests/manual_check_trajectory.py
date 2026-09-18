"""
manual_check_trajectory.py

Sanity check for trajectory.py: simulate a camera that pans slowly and
intentionally (dx=1 per frame) while also jittering randomly, then check
that smoothing removes the jitter while keeping the intentional pan.

Run with: python tests/manual_check_trajectory.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stabilizer.trajectory import accumulate_trajectory, smooth_trajectory, compute_corrections

np.random.seed(1)

n_frames = 100
intentional_pan = np.tile([1.0, 0.0, 0.0], (n_frames, 1))   # steady 1px/frame pan
jitter = np.random.normal(0, 3, (n_frames, 3))               # random shake noise
motions = intentional_pan + jitter

trajectory = accumulate_trajectory(motions)
smoothed = smooth_trajectory(trajectory, radius=15)
corrections = compute_corrections(trajectory, smoothed)

raw_jitter = np.std(np.diff(trajectory[:, 0]))
smoothed_jitter = np.std(np.diff(smoothed[:, 0]))

print(f"Raw trajectory dx jitter (std dev):      {raw_jitter:.3f}")
print(f"Smoothed trajectory dx jitter (std dev): {smoothed_jitter:.3f}")
print(f"Jitter reduced by: {100 * (1 - smoothed_jitter / raw_jitter):.1f}%")

pan_rate = (smoothed[-1, 0] - smoothed[0, 0]) / (n_frames - 1)
print(f"Smoothed trajectory still tracks the pan (~1.0 px/frame expected): {pan_rate:.3f}")

print(f"\nSample corrections (first 5 frames):\n{corrections[:5]}")