# Problem Statement

Handheld and casually recorded video footage is almost always affected by
unwanted camera shake, caused by natural hand tremor, walking motion, or an
unstable recording surface. This shake makes footage visually unpleasant
and harder to analyze, both for casual viewing and for downstream computer
vision tasks (e.g. tracking, surveillance, or feature matching), which
assume a relatively stable camera.

This project addresses the problem: **given a shaky input video, can we
automatically produce a visually stabilized version of it, using only
classical, hand-engineered computer vision techniques** (no deep learning
models), consistent with the geometric and motion-analysis concepts taught
in CSE3010 Computer Vision (Module 4: Motion Analysis — background
modeling, optical flow, KLT tracking).

## Scope of the Project

The project implements a full digital video stabilization pipeline:

- Detecting trackable feature points in each video frame (Shi-Tomasi
  corner detection)
- Tracking those points across consecutive frames using Lucas-Kanade (KLT)
  optical flow
- Estimating the frame-to-frame camera motion (translation + rotation)
  from the tracked point correspondences
- Reconstructing the camera's cumulative trajectory across the whole video
- Smoothing that trajectory to separate intentional camera movement from
  high-frequency shake
- Warping each frame to correct for the difference between its raw and
  smoothed trajectory position
- Compensating for the resulting border artifacts via a controlled zoom/crop
- Exposing the entire pipeline as a command-line tool, along with
  diagnostic trajectory plots for evaluation

**Out of scope:** rolling-shutter correction, real-time/live-camera
stabilization, and deep-learning-based stabilization methods — the focus
is on the classical geometric/motion-analysis pipeline covered in this
course.

## Target Users

- Students and hobbyists who record handheld video (phone/camera) and want
  a free, transparent, script-based way to stabilize footage without
  relying on a black-box mobile app or paid software
- Anyone wanting to understand _how_ stabilization works internally, since
  every stage of the pipeline is inspectable and explainable

## High-Level Features

1. Command-line video stabilization: `python main.py --input shaky.mp4 --output stable.mp4`
2. Adjustable smoothing strength via a `--smoothing-radius` parameter
3. Adjustable border-cropping via a `--zoom-ratio` parameter
4. Optional diagnostic trajectory plot (`--plot`) comparing raw vs
   smoothed camera paths, for analysis and reporting
5. Synthetic shaky-video generator (`scripts/generate_test_video.py`) for
   testing the pipeline without needing real camera footage
6. Automated unit tests (`pytest`) covering the trajectory and
   motion-estimation modules

## Validation

The pipeline was validated two ways:

- On a synthetic shaky video with a known shake amplitude, it reduced
  frame-to-frame jitter by ~97% (dx) and ~91% (dy).
- On real handheld phone footage, jitter reduction was more modest
  (~25% dx, ~8% dy) — because most of the real camera motion was an
  intentional pan rather than shake, and the smoothing correctly left
  intentional motion mostly untouched rather than flattening everything.
