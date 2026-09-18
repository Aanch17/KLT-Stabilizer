"""
cli.py

Command-line interface for the video stabilizer.

Usage:
    python main.py --input shaky.mp4 --output stable.mp4
    python main.py --input shaky.mp4 --output stable.mp4 --plot trajectory.png
    python main.py --input shaky.mp4 --output stable.mp4 --smoothing-radius 25
"""

import argparse
import logging
import sys

from . import io_utils
from . import pipeline

logger = logging.getLogger("stabilizer.cli")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Stabilize a shaky video using classical feature-tracking based motion estimation."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input (shaky) video.")
    parser.add_argument("--output", "-o", required=True, help="Path to write the stabilized video to.")
    parser.add_argument(
        "--smoothing-radius", type=int, default=15,
        help="Trajectory smoothing window half-width in frames (default: 15). "
             "Higher = smoother but less responsive to intentional camera movement.",
    )
    parser.add_argument(
        "--zoom-ratio", type=float, default=0.04,
        help="Fraction of the frame cropped to hide stabilization borders (default: 0.04).",
    )
    parser.add_argument(
        "--plot", default=None,
        help="Optional path to save a raw-vs-smoothed trajectory plot (e.g. trajectory.png).",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    io_utils.setup_logging(level=getattr(logging, args.log_level))

    try:
        pipeline.run(
            input_path=args.input,
            output_path=args.output,
            smoothing_radius=args.smoothing_radius,
            zoom_ratio=args.zoom_ratio,
            trajectory_plot_path=args.plot,
        )
    except (FileNotFoundError, ValueError, IOError) as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()