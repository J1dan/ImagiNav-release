"""LPIPS metric for video frames."""

from __future__ import annotations

from typing import Optional

import torch


def _to_minus_one_one(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 2.0 - 1.0


def compute_lpips_video(
    gt_video: torch.Tensor,
    gen_video: torch.Tensor,
    input_range: str = "0_1",
    net: str = "alex",
    device: Optional[str] = None,
) -> float:
    """Compute LPIPS across corresponding frames and return the mean score.

    Args:
        gt_video: Tensor [T, C, H, W] or [N, T, C, H, W]
        gen_video: Tensor [T, C, H, W] or [N, T, C, H, W]
        input_range: "0_1" or "-1_1" indicating current tensor range
        net: LPIPS backbone (e.g., "alex", "vgg")
        device: Optional device string
    """
    import logging
    log = logging.getLogger(__name__)
    
    if gt_video.dim() == 5:
        gt = gt_video.reshape(-1, *gt_video.shape[2:])
    else:
        gt = gt_video

    if gen_video.dim() == 5:
        gen = gen_video.reshape(-1, *gen_video.shape[2:])
    else:
        gen = gen_video

    t = min(gt.shape[0], gen.shape[0])
    gt = gt[:t]
    gen = gen[:t]

    if input_range == "0_1":
        gt = _to_minus_one_one(gt)
        gen = _to_minus_one_one(gen)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gt = gt.to(device)
    gen = gen.to(device)
    
    log.info(f"Computing LPIPS on {t} frames using {net} network...")

    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        metric = LearnedPerceptualImagePatchSimilarity(net_type=net).to(device)
        metric.eval()
        with torch.no_grad():
            score = metric(gen, gt)
        result = float(score.detach().cpu().item())
        log.info(f"LPIPS score: {result:.4f}")
        return result
    except Exception:
        pass

    try:
        import lpips

        loss_fn = lpips.LPIPS(net=net).to(device)
        loss_fn.eval()
        with torch.no_grad():
            score = loss_fn(gen, gt)
        result = float(score.mean().detach().cpu().item())
        log.info(f"LPIPS score: {result:.4f}")
        return result
    except Exception as e:
        raise ImportError(
            "LPIPS requires either torchmetrics or the lpips package. "
            "Install with `pip install torchmetrics` or `pip install lpips`."
        ) from e


def compute_pixel_metrics(
    ref_video: torch.Tensor,
    gen_video: torch.Tensor,
    input_range: str = "0_1",
    device: Optional[str] = None,
) -> tuple[float, float]:
    """Compute PSNR and SSIM metrics across video frames.
    
    Args:
        ref_video: Reference tensor [T, C, H, W] in range [0, 1]
        gen_video: Generated tensor [T, C, H, W] in range [0, 1]
        input_range: "0_1" or "-1_1" indicating current tensor range
        device: Optional device string
        
    Returns:
        Tuple of (psnr, ssim) float values
    """
    import logging
    log = logging.getLogger(__name__)
    
    # Normalize to 0-1 range if needed
    if input_range == "-1_1":
        ref_video = (ref_video + 1.0) / 2.0
        gen_video = (gen_video + 1.0) / 2.0
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref_video = ref_video.to(device)
    gen_video = gen_video.to(device)
    
    try:
        from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
        
        log.info(f"Computing PSNR and SSIM on {ref_video.shape[0]} frames...")
        
        with torch.no_grad():
            psnr = peak_signal_noise_ratio(gen_video, ref_video, data_range=1.0)
            ssim = structural_similarity_index_measure(gen_video, ref_video, data_range=1.0)
        
        psnr_val = float(psnr.detach().cpu().item())
        ssim_val = float(ssim.detach().cpu().item())
        
        log.info(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
        
        return psnr_val, ssim_val
    except ImportError as e:
        raise ImportError(
            "PSNR/SSIM requires torchmetrics. Install with `pip install torchmetrics`."
        ) from e
