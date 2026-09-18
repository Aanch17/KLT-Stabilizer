"""
manual_check_frame_correction.py

Visual sanity check for frame_correction.py. Creates a textured frame,
applies a correction (shift), and saves the before/after images side by
side so you can SEE that:
  1. The warp actually shifts the image content
  2. fix_border() successfully hides the resulting black edges

Run with: python tests/manual_check_frame_correction.py
Then open tests/frame_correction_check.png to view the result.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from stabilizer.frame_correction import warp_frame, fix_border, stabilize_frame

# Build a textured test frame
np.random.seed(3)
frame = np.zeros((300, 400, 3), dtype=np.uint8)
for _ in range(30):
    pt1 = tuple(np.random.randint(0, 380, 2))
    color = tuple(int(c) for c in np.random.randint(50, 255, 3))
    cv2.rectangle(frame, pt1, (pt1[0] + 20, pt1[1] + 20), color, -1)

# Apply a large correction so the border effect is very visible
warped = warp_frame(frame, dx=25, dy=15, da=0.03)
corrected = stabilize_frame(frame, dx=25, dy=15, da=0.03, zoom_ratio=0.08)

# Stack them side by side: original | warped (with black borders) | corrected (borders hidden)
combined = np.hstack([frame, warped, corrected])

output_path = os.path.join(os.path.dirname(__file__), "frame_correction_check.png")
cv2.imwrite(output_path, combined)
print(f"Saved comparison image to: {output_path}")
print("Left = original | Middle = warped (should show black borders) | Right = corrected (borders hidden)")