# Video Stabilization System

A command-line video stabilizer built with classical computer vision
techniques (feature tracking, optical flow, trajectory smoothing) — no
deep learning models involved. Built as a CSE3010 (Computer Vision) course
project.

## Overview

Shaky handheld video is stabilized by:

1. Tracking feature points across frames (Shi-Tomasi corners + Lucas-Kanade
   optical flow)
2. Estimating the camera's frame-to-frame motion from those tracked points
3. Reconstructing and smoothing the camera's overall trajectory
4. Warping each frame to compensate for the difference between its raw and
   smoothed position
5. Cropping slightly to hide the resulting border artifacts

See [`statement.md`](statement.md) for the full problem statement and scope.

## Features

- Fully command-line driven — no GUI required
- Adjustable smoothing strength and border-crop ratio
- Diagnostic trajectory plot (raw vs smoothed camera path) for analysis
- Synthetic shaky-video generator for testing without real footage
- Automated unit test suite (`pytest`) — 10 tests across motion estimation
  and trajectory modules
- Robust error handling for missing/corrupt input files and short videos
- Validated on both synthetic test video and real handheld phone footage

## Technologies Used

- Python 3.9+
- OpenCV (`opencv-python`) — feature detection, optical flow, warping
- NumPy — trajectory math
- Matplotlib — diagnostic plots
- pytest — unit testing

## Project Structure

video-stabilizer/
├── main.py # CLI entry point
├── requirements.txt
├── statement.md # Problem statement, scope, target users
├── stabilizer/
│ ├── motion_estimation.py # Feature tracking + per-frame motion estimation
│ ├── trajectory.py # Trajectory accumulation + smoothing + corrections
│ ├── frame_correction.py # Frame warping + border handling
│ ├── io_utils.py # Video I/O, validation, logging
│ ├── pipeline.py # Orchestrates the full pipeline
│ └── cli.py # argparse CLI
├── scripts/
│ └── generate_test_video.py # Generates a synthetic shaky test video
└── tests/
├── test_motion_estimation.py
└── test_trajectory.py

## Setup & Installation

**Prerequisite:** Python 3.9 or newer installed.

1. Clone the repository:

```bash
   git clone https://github.com/<github-username>/<repo-name>.git
   cd <repo-name>
```

2. (Recommended) Create and activate a virtual environment:

```bash
   python -m venv venv
   venv\Scripts\Activate.ps1        # On Windows PowerShell
   # source venv/bin/activate       # On Mac/Linux
```

3. Install dependencies:

```bash
   pip install -r requirements.txt
```

## Running the Project

### 1. Generate a test video (optional, if you don't have footage handy)

```bash
python scripts/generate_test_video.py --output shaky_test.mp4 --frames 150 --shake 8
```

This creates a synthetic shaky video you can use to try the pipeline
immediately, without needing a real shaky recording.

### 2. Stabilize a video

```bash
python main.py --input shaky_test.mp4 --output stable_test.mp4
```

With the optional diagnostic trajectory plot:

```bash
python main.py --input shaky_test.mp4 --output stable_test.mp4 --plot trajectory.png
```

### CLI Options

| Flag                 | Description                                                      | Default  |
| -------------------- | ---------------------------------------------------------------- | -------- |
| `--input`, `-i`      | Path to the input (shaky) video                                  | required |
| `--output`, `-o`     | Path to write the stabilized video                               | required |
| `--smoothing-radius` | Trajectory smoothing window half-width, in frames                | 15       |
| `--zoom-ratio`       | Fraction of the frame cropped to hide stabilization borders      | 0.04     |
| `--plot`             | Path to save a raw-vs-smoothed trajectory plot (e.g. `traj.png`) | none     |
| `--log-level`        | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`           | INFO     |

## Testing

Run the automated test suite:

```bash
pip install pytest   # if not already installed
pytest tests/ -v
```

All 10 tests should pass, covering feature detection, KLT tracking,
transform estimation, trajectory accumulation, and smoothing correctness.

To manually verify stabilization quality on any video, compare the
frame-to-frame motion jitter (standard deviation) before and after
stabilization — see the project report for the full methodology and
measured results (97% jitter reduction on synthetic test footage, 25% on
real handheld footage where most motion was intentional panning).

## Notes on Parameter Tuning

- **`--smoothing-radius`**: higher values produce smoother output but can
  make the camera feel less responsive to genuinely intentional panning.
  Lower values preserve intentional motion better but remove less shake.
- **`--zoom-ratio`**: higher values more reliably hide black borders but
  crop away more of the original frame.
