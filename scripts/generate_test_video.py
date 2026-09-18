"""
generate_test_video.py

Generates a synthetic "shaky" video for testing the stabilization pipeline
without needing real camera footage. Draws a static textured scene and
simulates camera shake by randomly jittering the crop window each frame.

Usage:
    python scripts/generate_test_video.py --output shaky_test.mp4
"""

import argparse
import cv2
import numpy as np


def build_scene(width, height):
    """Build a large, richly textured static 'world' image to shoot our
    synthetic shaky camera at (needs texture for corner detection to work)."""
    rng = np.random.default_rng(42)
    scene = np.full((height, width, 3), 30, dtype=np.uint8)

    for _ in range(120):
        x, y = rng.integers(0, width), rng.integers(0, height)
        size = rng.integers(15, 60)
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        shape = rng.integers(0, 3)
        if shape == 0:
            cv2.rectangle(scene, (x, y), (x + size, y + size), color, -1)
        elif shape == 1:
            cv2.circle(scene, (x, y), size // 2, color, -1)
        else:
            cv2.line(scene, (x, y), (x + size, y + size), color, 4)

    return scene


def generate(output_path, n_frames=150, fps=30, width=640, height=360, shake_amplitude=8):
    scene_width, scene_height = width + 100, height + 100
    scene = build_scene(scene_width, scene_height)

    rng = np.random.default_rng(1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Smooth "intentional" pan across the scene, plus high-frequency jitter
    cx, cy = 50, 50
    pan_dx, pan_dy = 0.3, 0.1

    for i in range(n_frames):
        cx += pan_dx
        cy += pan_dy

        jitter_x = rng.normal(0, shake_amplitude)
        jitter_y = rng.normal(0, shake_amplitude)

        x = int(np.clip(cx + jitter_x, 0, scene_width - width))
        y = int(np.clip(cy + jitter_y, 0, scene_height - height))

        frame = scene[y:y + height, x:x + width].copy()
        writer.write(frame)

    writer.release()
    print(f"Synthetic shaky video written to {output_path} ({n_frames} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic shaky test video.")
    parser.add_argument("--output", default="shaky_test.mp4")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--shake", type=float, default=8.0, help="Shake amplitude in pixels.")
    args = parser.parse_args()
    generate(args.output, n_frames=args.frames, shake_amplitude=args.shake)