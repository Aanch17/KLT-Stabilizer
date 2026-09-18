"""
manual_check_motion.py

Quick sanity check for motion_estimation.py: create two frames with a
KNOWN shift between them, and check whether the module recovers that
exact shift. Run with: python tests/manual_check_motion.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from stabilizer.motion_estimation import detect_features, track_features, estimate_transform

# Create a synthetic frame with some texture (random rectangles) so it
# actually has corners to detect and track.
np.random.seed(0)
frame1 = np.zeros((300, 400, 3), dtype=np.uint8)
for _ in range(40):
    pt1 = tuple(np.random.randint(0, 400, 2))
    color = tuple(int(c) for c in np.random.randint(50, 255, 3))
    cv2.rectangle(frame1, pt1, (pt1[0] + 20, pt1[1] + 20), color, -1)

# Shift frame1 by a KNOWN amount (dx=10, dy=5) to create frame2.
M = np.float32([[1, 0, 10], [0, 1, 5]])
frame2 = cv2.warpAffine(frame1, M, (400, 300))

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

points = detect_features(gray1)
print("Detected points:", 0 if points is None else len(points))

good_prev, good_curr = track_features(gray1, gray2, points)
print("Tracked points:", len(good_prev))

result = estimate_transform(good_prev, good_curr)
print("Estimated (dx, dy, da):", result)
print("Expected: dx=10, dy=5, da=0")