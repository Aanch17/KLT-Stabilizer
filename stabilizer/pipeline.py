"""
pipeline.py

Orchestrates the full video stabilization pipeline:

    Pass 1 (read-only):  estimate frame-to-frame motion for the whole video
    Analysis:            accumulate + smooth the trajectory, compute corrections
    Pass 2 (read+write):  re-read the video and write out corrected frames

Two passes over the video are used (rather than holding every frame in
memory) so memory usage stays low even for long videos -- this is the
"resource efficiency" non-functional requirement in the project report.
"""

import logging
import numpy as np

from . import io_utils
from . import motion_estimation
from . import trajectory as traj
from . import frame_correction

logger = logging.getLogger("stabilizer.pipeline")


def run(input_path, output_path, smoothing_radius=15, zoom_ratio=0.04,
        trajectory_plot_path=None):
    """
    Run the full stabilization pipeline on input_path, writing the
    stabilized result to output_path.

    smoothing_radius: half-width (in frames) of the trajectory smoothing window
    zoom_ratio: fraction of the frame cropped to hide stabilization borders
    trajectory_plot_path: if given, saves a before/after trajectory plot here
                           (useful evidence for the project report)
    """
    fps, width, height, frame_count = io_utils.get_video_properties(input_path)
    logger.info(
        "Input video: %s (%dx%d, %.2f fps, %d frames)",
        input_path, width, height, fps, frame_count,
    )

    # ---- Pass 1: estimate motion for every frame transition ----
    logger.info("Pass 1/2: estimating frame-to-frame motion...")
    motions = list(
        motion_estimation.compute_frame_to_frame_motion(io_utils.frame_generator(input_path))
    )
    motions = np.array(motions, dtype=np.float64)
    logger.info("Estimated motion for %d frame transitions.", len(motions))

    # ---- Analysis: trajectory + smoothing + corrections ----
    trajectory = traj.accumulate_trajectory(motions)
    smoothed_trajectory = traj.smooth_trajectory(trajectory, radius=smoothing_radius)
    corrections = traj.compute_corrections(trajectory, smoothed_trajectory)

    if trajectory_plot_path:
        _save_trajectory_plot(trajectory, smoothed_trajectory, trajectory_plot_path)

    # ---- Pass 2: apply corrections and write output ----
    logger.info("Pass 2/2: applying corrections and writing output video...")
    with io_utils.VideoWriter(output_path, fps, width, height) as writer:
        frame_gen = io_utils.frame_generator(input_path)

        # The first frame has no preceding correction (it's the reference
        # frame), so it's written through unchanged.
        first_frame = next(frame_gen)
        writer.write(frame_correction.fix_border(first_frame, zoom_ratio=zoom_ratio))

        for i, frame in enumerate(frame_gen):
            dx, dy, da = corrections[i]
            stabilized = frame_correction.stabilize_frame(frame, dx, dy, da, zoom_ratio=zoom_ratio)
            writer.write(stabilized)

    logger.info("Done. Stabilized video written to %s", output_path)


def _save_trajectory_plot(trajectory, smoothed_trajectory, path):
    """
    Save a plot comparing the raw (shaky) trajectory to the smoothed
    trajectory, for the dx and dy components -- good visual evidence for
    the project report's "Results" section.
    """
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just save to file
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(trajectory[:, 0], label="Raw (shaky)", alpha=0.7)
    axes[0].plot(smoothed_trajectory[:, 0], label="Smoothed", linewidth=2)
    axes[0].set_ylabel("Cumulative dx (pixels)")
    axes[0].legend()
    axes[0].set_title("Camera Trajectory: Raw vs Smoothed")

    axes[1].plot(trajectory[:, 1], label="Raw (shaky)", alpha=0.7)
    axes[1].plot(smoothed_trajectory[:, 1], label="Smoothed", linewidth=2)
    axes[1].set_ylabel("Cumulative dy (pixels)")
    axes[1].set_xlabel("Frame number")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Trajectory plot saved to %s", path)