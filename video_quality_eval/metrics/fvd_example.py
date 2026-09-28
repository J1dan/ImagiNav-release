"""Minimal example to compute FVD using the local frechet_video_distance code.

Usage:
  conda activate vggt
  python ImagiNav/video_quality_eval/metrics/fvd_example.py

This script:
 - loads MP4s from `ImagiNav/video_quality_eval/datasets/reference_videos` and
   `ImagiNav/video_quality_eval/datasets/synthetic_videos`
 - samples/ pads to 15 frames per video
 - repeats videos to build a batch of 16 (I3D expects batch=16)
 - calls `compute_fvd` from the project's metrics module
"""

import os
import glob
from typing import List

import cv2
import numpy as np
import torch

# Import compute_fvd directly from the file to avoid importing the top-level
# ImagiNav package (which pulls many heavy deps). This makes the example
# self-contained and runnable in the environment.
import importlib.util
import pathlib

_fvd_path = pathlib.Path(__file__).resolve().parent / "fvd.py"
_spec = importlib.util.spec_from_file_location("local_fvd", str(_fvd_path))
_local_fvd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_fvd)
compute_fvd = _local_fvd.compute_fvd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
REF_DIR = os.path.join(DATA_DIR, "reference_videos")
SYN_DIR = os.path.join(DATA_DIR, "synthetic_videos")

NUM_FRAMES = 15
BATCH_SIZE = 16
RESIZE_SCALE = 0.5


def load_video_frames(
    path: str,
    num_frames: int = NUM_FRAMES,
    resize: tuple = None,
    resize_scale: float = RESIZE_SCALE,
):
    cap = cv2.VideoCapture(path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    # sample num_frames uniformly across the video
    indices = np.linspace(0, max(total - 1, 0), num=num_frames, dtype=int)
    idx_set = set(indices.tolist())
    cur = 0
    taken = 0
    while cur < total and taken < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize is not None:
                frame = cv2.resize(frame, resize)
            elif resize_scale is not None:
                h, w = frame.shape[:2]
                new_w = max(int(w * resize_scale), 1)
                new_h = max(int(h * resize_scale), 1)
                frame = cv2.resize(frame, (new_w, new_h))
            frames.append(frame)
            taken += 1
        cur += 1
    cap.release()

    # If video shorter, pad by repeating last frame
    if len(frames) == 0:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    arr = np.stack(frames, axis=0).astype(np.uint8)
    return arr


def make_batch_from_dir(
    directory: str,
    batch_size: int = BATCH_SIZE,
    num_frames: int = NUM_FRAMES,
    resize_scale: float = RESIZE_SCALE,
):
    files = sorted(glob.glob(os.path.join(directory, "*.mp4")))
    if len(files) == 0:
        raise RuntimeError(f"No mp4 files found in {directory}")
    videos = []
    for p in files:
        v = load_video_frames(p, num_frames=num_frames, resize=None, resize_scale=resize_scale)
        if v is not None:
            videos.append(v)
    if len(videos) == 0:
        raise RuntimeError(f"Could not load any videos from {directory}")
    # Repeat videos until we have batch_size
    while len(videos) < batch_size:
        videos.append(videos[-1].copy())
    videos = videos[:batch_size]
    # Stack -> [N, T, H, W, 3]
    batch = np.stack(videos, axis=0).astype(np.uint8)
    return batch


def main():
    ref_batch = make_batch_from_dir(REF_DIR)
    syn_batch = make_batch_from_dir(SYN_DIR)

    # convert to torch float in 0-1 and permute to [N, T, C, H, W]
    ref_t = torch.from_numpy(ref_batch).float() / 255.0
    ref_t = ref_t.permute(0, 1, 4, 2, 3)
    syn_t = torch.from_numpy(syn_batch).float() / 255.0
    syn_t = syn_t.permute(0, 1, 4, 2, 3)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['i3d','videomae'], default='videomae', help='Model to use for FVD (i3d|videomae)')
    parser.add_argument('--half-precision', action='store_true', help='Load model in half precision (float16) to reduce VRAM')
    args = parser.parse_args()

    print("Computing FVD (this may download model weights the first time and take a while)...")
    score = compute_fvd(ref_t, syn_t, input_range="0_1", model_name=args.model, half_precision=args.half_precision)
    print(f"FVD: {score:.2f}")


if __name__ == "__main__":
    main()
