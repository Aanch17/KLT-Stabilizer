"""
Unit tests for stabilizer.motion_estimation

Run with: pytest tests/
"""

import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stabilizer.motion_estimation import detect_features, track_features, estimate_transform


def _make_textured_frame(width=400, height=300, seed=0):
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(20):
        pt1 = (int(rng.integers(0, width - 20)), int(rng.integers(0, height - 20)))
        color = tuple(int(c) for c in rng.integers(50, 255, 3))
        cv2.rectangle(frame, pt1, (pt1[0] + 20, pt1[1] + 20), color, -1)
    return frame


def test_detect_features_finds_points_on_textured_frame():
    frame = _make_textured_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points = detect_features(gray)
    assert points is not None
    assert len(points) > 0


def test_detect_features_none_on_blank_frame():
    blank = np.zeros((300, 400), dtype=np.uint8)
    points = detect_features(blank)
    assert points is None


def test_estimate_transform_recovers_known_translation():
    frame1 = _make_textured_frame(seed=1)
    M = np.float32([[1, 0, 15], [0, 1, -7]])
    frame2 = cv2.warpAffine(frame1, M, (frame1.shape[1], frame1.shape[0]))

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    points = detect_features(gray1)
    good_prev, good_curr = track_features(gray1, gray2, points)
    result = estimate_transform(good_prev, good_curr)

    assert result is not None
    dx, dy, da = result
    assert abs(dx - 15) < 1.0
    assert abs(dy - (-7)) < 1.0
    assert abs(da) < 0.05


def test_estimate_transform_none_with_too_few_points():
    tiny_prev = np.array([[[10.0, 10.0]]], dtype=np.float32)
    tiny_curr = np.array([[[11.0, 11.0]]], dtype=np.float32)
    result = estimate_transform(tiny_prev, tiny_curr)
    assert result is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))