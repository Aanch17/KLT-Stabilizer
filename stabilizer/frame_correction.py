"""
frame_correction.py

Applies the per-frame (dx, dy, da) corrections computed by trajectory.py
to the actual video frames, and cleans up the black borders that
stabilization introduces.

Why borders appear: when we shift/rotate a frame to cancel out shake, part
of the frame gets pushed outside the image boundary, leaving black areas
where there's no image data. We fix this by slightly zooming in (cropping
a border margin and rescaling back up), which trades a small amount of
field-of-view for a border-free result.
"""

import cv2
import numpy as np


def build_transform_matrix(dx, dy, da):
    """
    Build a 2x3 affine transform matrix from a (dx, dy, da) correction.

    da is a small rotation angle in radians.
    """
    cos_a = np.cos(da)
    sin_a = np.sin(da)

    matrix = np.array([
        [cos_a, -sin_a, dx],
        [sin_a, cos_a, dy],
    ], dtype=np.float64)

    return matrix


def warp_frame(frame, dx, dy, da):
    """
    Apply the (dx, dy, da) correction to a single frame via an affine warp.
    """
    h, w = frame.shape[:2]
    matrix = build_transform_matrix(dx, dy, da)
    warped = cv2.warpAffine(frame, matrix, (w, h))
    return warped


def fix_border(frame, zoom_ratio=0.04):
    """
    Crop a thin margin from all sides of the frame and rescale back up to
    the original size, to hide the black borders introduced by warping.

    zoom_ratio: fraction of the frame to crop from each side (0.04 = 4%,
    a reasonable default that hides most stabilization artifacts without
    losing too much field of view).
    """
    h, w = frame.shape[:2]

    # Slightly enlarge the frame around its center, which has the same
    # visual effect as cropping a margin and rescaling back up.
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + zoom_ratio)
    zoomed = cv2.warpAffine(frame, matrix, (w, h))
    return zoomed


def stabilize_frame(frame, dx, dy, da, zoom_ratio=0.04):
    """
    Full per-frame correction: warp to cancel shake, then fix borders.
    """
    warped = warp_frame(frame, dx, dy, da)
    result = fix_border(warped, zoom_ratio=zoom_ratio)
    return result