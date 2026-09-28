"""Utility helpers for loading and aligning video tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import cv2


def load_video_tensor(
    video_path: Path,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Load a video file as a float tensor in range [0, 1].

    Returns tensor of shape [T, C, H, W] in RGB order.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1
    if target_fps and fps and fps > 0:
        frame_interval = max(1, int(round(fps / target_fps)))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_hw is not None:
                h, w = resize_hw
                frame_rgb = cv2.resize(frame_rgb, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from video: {video_path}")

    arr = np.stack(frames, axis=0)  # [T, H, W, C]
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return tensor


def align_video_tensors(
    ref_video: torch.Tensor,
    gen_video: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two videos to the same [T, C, H, W] shape.

    - Temporally: Checks if frame counts match exactly. Raises ValueError if not.
    - Spatially: RESIZES the Reference video to match the Generated video's resolution.
    """
    # 1. Strict Temporal Check
    if ref_video.shape[0] != gen_video.shape[0]:
        raise ValueError(
            f"Temporal mismatch: Reference has {ref_video.shape[0]} frames, "
            f"but Generated has {gen_video.shape[0]} frames. "
            "Frame counts must be identical for pairwise metrics."
        )

    ref = ref_video
    gen = gen_video

    # 2. Spatial Resize (Downsample Ref to match Gen)
    # ref shape: [T, C, H, W]
    if ref.shape[2:] != gen.shape[2:]:
        target_h, target_w = gen.shape[2], gen.shape[3]
        
        # F.interpolate works on [Batch, Channel, Height, Width]
        # We treat Time (T) as the Batch dimension here, which is perfectly valid.
        ref = F.interpolate(
            ref, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )

    return ref, gen