"""
Unit tests for stabilizer.trajectory

Run with: pytest tests/
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stabilizer.trajectory import accumulate_trajectory, smooth_trajectory, compute_corrections


def test_accumulate_trajectory_simple_sum():
    motions = np.array([
        [1.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
    ])
    trajectory = accumulate_trajectory(motions)
    expected = np.array([
        [1.0, 2.0, 0.0],
        [2.0, 4.0, 0.0],
        [3.0, 6.0, 0.0],
    ])
    assert np.allclose(trajectory, expected)


def test_smooth_trajectory_preserves_shape():
    trajectory = np.random.rand(50, 3)
    smoothed = smooth_trajectory(trajectory, radius=5)
    assert smoothed.shape == trajectory.shape


def test_smooth_trajectory_reduces_noise():
    # A perfectly linear trend plus random jitter -- smoothing should
    # reduce the frame-to-frame variance without changing the overall trend.
    n = 200
    trend = np.linspace(0, 100, n)
    np.random.seed(0)
    noisy = trend + np.random.normal(0, 5, n)
    trajectory = np.stack([noisy, noisy, np.zeros(n)], axis=1)

    smoothed = smooth_trajectory(trajectory, radius=10)

    raw_jitter = np.std(np.diff(trajectory[:, 0]))
    smoothed_jitter = np.std(np.diff(smoothed[:, 0]))

    assert smoothed_jitter < raw_jitter, "Smoothing should reduce frame-to-frame jitter"

    # Overall trend should still be roughly preserved (first vs last point)
    assert abs(smoothed[-1, 0] - trend[-1]) < 15


def test_smooth_trajectory_constant_input_unchanged():
    # Smoothing a perfectly constant trajectory should return it unchanged
    trajectory = np.ones((30, 3)) * 5.0
    smoothed = smooth_trajectory(trajectory, radius=8)
    assert np.allclose(smoothed, trajectory)


def test_compute_corrections_zero_when_already_smooth():
    # If the raw trajectory already equals the "smoothed" trajectory,
    # the correction should be exactly zero everywhere.
    trajectory = np.random.rand(20, 3)
    corrections = compute_corrections(trajectory, trajectory)
    assert np.allclose(corrections, 0.0)


def test_compute_corrections_matches_difference():
    trajectory = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    smoothed = np.array([[8.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    corrections = compute_corrections(trajectory, smoothed)
    expected = np.array([[-2.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    assert np.allclose(corrections, expected)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))