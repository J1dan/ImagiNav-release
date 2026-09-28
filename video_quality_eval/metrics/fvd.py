"""FVD metric wrapper."""

from __future__ import annotations

from typing import Optional, Union, List

import numpy as np
import torch


def _normalize_input(videos: torch.Tensor, input_range: str) -> torch.Tensor:
    if input_range == "0_1":
        return videos
    if input_range == "-1_1":
        return (videos + 1.0) / 2.0
    raise ValueError(f"Unsupported input_range: {input_range}")


def _slice_into_clips(video_list: List[torch.Tensor], clip_length: int = 16, stride: int = 16) -> torch.Tensor:
    """Slice videos into fixed-length clips using sliding window.
    
    Args:
        video_list: List of [T, C, H, W] tensors
        clip_length: Number of frames per clip (default 16 for VideoMAE)
        stride: Step size between clips (default 16 for non-overlapping)
    
    Returns:
        Stacked tensor of clips [N_clips, clip_length, C, H, W]
    """
    all_clips = []
    
    for video in video_list:
        T, C, H, W = video.shape
        
        if T < clip_length:
            # Pad short videos by repeating last frame
            padding = clip_length - T
            last_frame = video[-1:].expand(padding, -1, -1, -1)
            video = torch.cat([video, last_frame], dim=0)
            T = clip_length
        
        # Extract non-overlapping clips
        num_clips = (T - clip_length) // stride + 1
        for i in range(num_clips):
            start_idx = i * stride
            clip = video[start_idx:start_idx + clip_length]
            all_clips.append(clip)
        
        # Handle remainder: add back-aligned clip for the last 16 frames
        last_start = T - clip_length
        if last_start > (num_clips - 1) * stride:
            # Only add if it doesn't duplicate the last clip
            clip = video[last_start:last_start + clip_length]
            all_clips.append(clip)
    
    if not all_clips:
        raise ValueError("No clips extracted from video list")
    
    # Stack all clips into single batch
    return torch.stack(all_clips, dim=0)


def _resolve_videomae_ckpt(ckpt_path: Optional[str]) -> Optional[str]:
    """Return a valid local path to the VideoMAE checkpoint.

    Resolution order:
    1. Explicit path provided via config -> use as-is (must exist).
    2. None -> download via huggingface_hub into ~/.cache/huggingface (the
       directory that is bind-mounted into the Docker container, so the file
       persists across container restarts).

    This replaces cdfvd's built-in requests.get downloader, which saves the
    file into the container's site-packages directory (ephemeral) and can
    silently save HTML error pages instead of the real weights.
    """
    import logging
    log = logging.getLogger(__name__)

    if ckpt_path is not None:
        import os
        path = os.path.expanduser(ckpt_path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"VideoMAE checkpoint not found at configured path: {path}"
            )
        log.info(f"Using VideoMAE checkpoint from config: {path}")
        return path

    # Auto-download into the HuggingFace cache (persisted via Docker volume mount)
    try:
        from huggingface_hub import hf_hub_download
        log.info(
            "Downloading VideoMAE checkpoint via huggingface_hub "
            "(will be cached in ~/.cache/huggingface across container restarts)..."
        )
        resolved = hf_hub_download(
            repo_id="OpenGVLab/InternVideoMAE_models",
            filename="mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth",
        )
        log.info(f"VideoMAE checkpoint ready at: {resolved}")
        return resolved
    except Exception as e:
        log.warning(
            f"huggingface_hub download failed ({e}). "
            "Falling back to cdfvd's built-in downloader (ckpt_path=None). "
            "If FVD then fails with an UnpicklingError, the checkpoint file is "
            "corrupt — set fvd.ckpt_path in your eval YAML to a manually "
            "downloaded copy of vit_g_hybrid_pt_1200e_ssv2_ft.pth."
        )
        return None  # let cdfvd try its own download


def compute_fvd(
    reference_videos: Union[torch.Tensor, List[torch.Tensor]],
    generated_videos: Union[torch.Tensor, List[torch.Tensor]],
    input_range: str = "0_1",
    device: Optional[str] = None,
    model_name: str = "videomae",
    half_precision: bool = True,
    ckpt_path: Optional[str] = None,
) -> float:
    """Compute Fréchet Video Distance (FVD) using the local cdfvd implementation.

    Args:
        reference_videos: Tensor [N, T, C, H, W] or List of [T, C, H, W] tensors
        generated_videos: Tensor [N, T, C, H, W] or List of [T, C, H, W] tensors
        input_range: "0_1" or "-1_1" indicating current tensor range
        device: Optional device string (e.g., "cuda" or "cpu"). If None, defaults to cuda if available.
        model_name: Which feature extractor to use in cdfvd ('videomae' or 'i3d').
        half_precision: Whether to load the model in half precision (float16).
        ckpt_path: Optional explicit path to the VideoMAE checkpoint file.
            If None (default), the checkpoint is downloaded automatically via
            huggingface_hub into ~/.cache/huggingface, which is the Docker
            bind-mount and therefore persists across container restarts.
            Set this in your eval YAML under fvd.ckpt_path to skip download.
    """
    import logging
    log = logging.getLogger(__name__)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Optimized path for disk-backed uint8 numpy/memmap inputs
    if input_range == "0_255_uint8":
        ref_np = reference_videos
        gen_np = generated_videos

        # If torch tensors were provided, convert to uint8 numpy in [N, T, H, W, C]
        if isinstance(ref_np, torch.Tensor):
            ref_np = ref_np.detach().cpu().numpy()
        if isinstance(gen_np, torch.Tensor):
            gen_np = gen_np.detach().cpu().numpy()

        # Ensure correct dtype
        if isinstance(ref_np, np.ndarray) and ref_np.dtype != np.uint8:
            ref_np = ref_np.astype(np.uint8, copy=False)
        if isinstance(gen_np, np.ndarray) and gen_np.dtype != np.uint8:
            gen_np = gen_np.astype(np.uint8, copy=False)

        try:
            import logging
            log = logging.getLogger(__name__)
            log.info("Using content-debiased FVD (cd-fvd) for FVD computation (uint8 memmap path)")

            resolved_ckpt = _resolve_videomae_ckpt(ckpt_path)

            # PyTorch >= 2.6 changed torch.load default to weights_only=True, which breaks
            # the VideoMAE checkpoint loader inside cdfvd (a trusted third-party package).
            # Monkey-patch torch.load to restore the old behaviour for this call only.
            import functools
            _original_torch_load = torch.load
            torch.load = functools.partial(_original_torch_load, weights_only=False)
            try:
                from cdfvd import fvd as _fvd
                evaluator = _fvd.cdfvd(model_name, ckpt_path=resolved_ckpt, device=device, half_precision=half_precision)
            finally:
                torch.load = _original_torch_load

            score = evaluator.compute_fvd(ref_np, gen_np)

            try:
                evaluator.offload_model_to_cpu()
            except Exception:
                pass

            log.info("cdfvd produced score: %s", score)
            return float(score)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    # Handle list inputs - slice into clips
    if isinstance(reference_videos, list):
        log.info(f"Slicing {len(reference_videos)} reference videos into 16-frame clips...")
        reference_videos = _slice_into_clips(reference_videos, clip_length=16, stride=16)
        log.info(f"Extracted {reference_videos.shape[0]} reference clips")
    
    if isinstance(generated_videos, list):
        log.info(f"Slicing {len(generated_videos)} generated videos into 16-frame clips...")
        generated_videos = _slice_into_clips(generated_videos, clip_length=16, stride=16)
        log.info(f"Extracted {generated_videos.shape[0]} generated clips")

    ref = _normalize_input(reference_videos.float(), input_range).to(device)
    gen = _normalize_input(generated_videos.float(), input_range).to(device)

    # Use pip-installed content-debiased FVD (cd-fvd) exclusively.
    try:
        log.info("Using content-debiased FVD (cd-fvd) for FVD computation")

        resolved_ckpt = _resolve_videomae_ckpt(ckpt_path)

        # PyTorch >= 2.6 changed torch.load default to weights_only=True, which breaks
        # the VideoMAE checkpoint loader inside cdfvd (a trusted third-party package).
        # Monkey-patch torch.load to restore the old behaviour for this call only.
        import functools
        _original_torch_load = torch.load
        torch.load = functools.partial(_original_torch_load, weights_only=False)
        try:
            from cdfvd import fvd as _fvd
            # cdfvd expects uint8 numpy arrays [N, T, H, W, C] in 0..255
            evaluator = _fvd.cdfvd(model_name, ckpt_path=resolved_ckpt, device=device, half_precision=half_precision)
        finally:
            torch.load = _original_torch_load

        def _to_uint8_np(t: torch.Tensor) -> np.ndarray:
            t_np = t.detach().cpu().numpy()
            out = (np.round(t_np * 255.0)).astype(np.uint8)
            out = np.transpose(out, (0, 1, 3, 4, 2))
            return out

        ref_np = _to_uint8_np(ref)
        gen_np = _to_uint8_np(gen)

        score = evaluator.compute_fvd(ref_np, gen_np)

        # free GPU memory if model was on CUDA
        try:
            evaluator.offload_model_to_cpu()
        except Exception:
            pass

        log.info("cdfvd produced score: %s", score)
        return float(score)
    except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
