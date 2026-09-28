"""
Evaluator for Video Quality (ImagiNav)

Implements the pipeline described by the user:
- Load dataset JSON with entries {"image_path": "images/xxx.jpg", "instruction": "..."} or {"video_path": "videos/xxx.mp4", "caption": "..."}
- For each entry:
    - Use pre-generated video if present; otherwise generate via LTX-Video
  - Extract motion primitives from generated video using VGGT
    - Compute motion fidelity (Levenshtein distance) vs primitives parsed from instruction
    - Optionally compute LPIPS per sample and FVD across dataset when reference videos are provided
  - Send video + prompt to Gemini to get three human-like scores:
      Instruction following (natural movement), Safety, Physical coherence
  - Save per-sample and aggregate results to JSON under `results/`

Notes:
- If Gemini API key not present, a deterministic fallback scorer will be used.
- Uses VGGT inference helpers from `ImagiNav/vggt/imaginav_scripts/infer_move.py`

"""

import json
import os
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

import torch

from ImagiNav.core.config import ImagiNavConfig
from ImagiNav.models.reasoner import GeminiReasoner

# Import VGGT helper functions from infer_move.py
from ImagiNav.vggt.imaginav_scripts.infer_move import (
    ensure_device_and_dtype,
    sorted_image_list,
    get_motion_trajectory,
    extract_frames_from_video,
    VGGT,
)

from ImagiNav.video_quality_eval.metrics import (
    motion_fidelity_score,
    compute_lpips_video,
    compute_pixel_metrics,
    compute_fvd,
    load_video_tensor,
    align_video_tensors,
    extract_video_motion,
    extract_video_labels_and_trajectory,
    ensure_vggt_metadata_entries,
    compute_rpe,
    parse_prompt_primitives,
    gemini_score_video,
)
from ImagiNav.video_quality_eval.utils.config_utils import load_yaml_config, resolve_path, apply_imaginav_overrides

logger = logging.getLogger("imaginav.video_eval")


class DiskBuffer:
    """Disk-backed buffer for storing FVD clips as uint8 [T, H, W, C]."""

    def __init__(self, temp_dir: Path, name: str, shape_example: tuple):
        """
        Args:
            shape_example: Tuple (T, H, W, C) of a SINGLE clip.
        """
        self.dir = temp_dir
        self.file_path = self.dir / f"{name}.dat"
        self.dtype = np.uint8
        self.shape_example = shape_example
        self.count = 0
        self.fp = open(self.file_path, "wb")

    def add(self, clip_tensor: torch.Tensor) -> None:
        """Write a single clip to disk.

        Expects tensor [T, C, H, W] in range 0-1.
        Converts to [T, H, W, C] uint8 0-255 and writes raw bytes.
        """
        clip = clip_tensor.permute(0, 2, 3, 1)
        clip_np = (clip * 255.0).byte().cpu().numpy()
        self.fp.write(clip_np.tobytes())
        self.count += 1

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass

    def get_memmap(self):
        """Returns a memory-mapped numpy array of the full dataset."""
        self.close()
        if self.count == 0:
            return None
        full_shape = (self.count,) + self.shape_example
        return np.memmap(
            self.file_path,
            dtype=self.dtype,
            mode="r",
            shape=full_shape,
        )


class VideoQualityEvaluator:
    def __init__(
        self,
        config: Optional[ImagiNavConfig] = None,
        work_dir: Optional[str] = None,
        device: Optional[str] = None,
        analysis_subsample_step: Optional[int] = None,
        config_path: Optional[Path] = None,
    ):
        """Initialize evaluator.

        Parameters:
            analysis_subsample_step: If provided, overrides the local video-quality config. If None, the evaluator
            will attempt to read `ImagiNav/configs/video_quality_eval_config.yaml` (or `config_path` if provided).
        """
        self.config = config or ImagiNavConfig()
        if work_dir is not None:
            self.work_dir = Path(work_dir)
        else:
            self.work_dir = Path(__file__).parent / "video_eval_results"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.work_dir / "results"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Configure file logging to the results directory (avoids duplicate handlers)
        logfile = self.results_dir / "eval.log"
        # Add file handler only if not already present for this file
        existing_fh = False
        for h in logger.handlers:
            try:
                if getattr(h, 'baseFilename', None) and Path(h.baseFilename).resolve() == logfile.resolve():
                    existing_fh = True
                    break
            except Exception:
                continue
        if not existing_fh:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(fh)
            logger.info(f"Logging evaluator output to: {logfile}")

        self.config_path = config_path or (Path(__file__).parent / "configs" / "video_quality_eval.yaml")
        self.eval_cfg = load_yaml_config(self.config_path)
        apply_imaginav_overrides(self.config, self.eval_cfg.get("imaginav_overrides", {}))
        config_base = self.config_path.parent if self.config_path else Path(__file__).parent

        # Metadata & scoring settings
        self.vggt_metadata_path = resolve_path(
            self.eval_cfg.get("dataset", {}).get("vggt_metadata_json"),
            config_base,
        )
        # Enable/disable metrics
        metrics_cfg = self.eval_cfg.get("enable_metrics", {})
        self.enable_motion_fidelity = bool(metrics_cfg.get("motion_fidelity", True))
        self.enable_gemini_scoring = bool(metrics_cfg.get("gemini_scoring", True))
        self.enable_lpips = bool(metrics_cfg.get("lpips", True))
        self.enable_psnr = bool(metrics_cfg.get("psnr", True))
        self.enable_ssim = bool(metrics_cfg.get("ssim", True))
        self.enable_fvd = bool(metrics_cfg.get("fvd", True))
        
        self.motion_compare_mode = self.eval_cfg.get("motion_fidelity", {}).get("compare_against", "primitives")
        self.prompt_max_primitives = int(self.eval_cfg.get("motion_fidelity", {}).get("prompt_max_primitives", 2))
        self.motion_yaw_thresh_deg = float(self.eval_cfg.get("motion_fidelity", {}).get("yaw_thresh_deg", 2.0))
        self.motion_forward_thresh_m = float(self.eval_cfg.get("motion_fidelity", {}).get("forward_thresh_m", 0.012))
        self.motion_min_consecutive_frames = int(self.eval_cfg.get("motion_fidelity", {}).get("min_consecutive_frames", 5))

        # Video-quality-specific settings (isolated config)
        self.analysis_subsample_step = 1
        # FVD defaults
        self.fvd_model_name = "videomae"
        self.fvd_half_precision = True
        self.fvd_ckpt_path: Optional[str] = None  # None → auto-download via huggingface_hub
        # If explicit param provided, use it. Otherwise try to load local config file.
        if analysis_subsample_step is not None:
            self.analysis_subsample_step = int(analysis_subsample_step)
        else:
            local_cfg = self.eval_cfg.get("motion_fidelity", {}) if self.eval_cfg else {}
            self.analysis_subsample_step = int(local_cfg.get("analysis_subsample_step", 1))
            fvd_cfg = self.eval_cfg.get("fvd", {}) if self.eval_cfg else {}
            self.fvd_model_name = str(fvd_cfg.get("model_name", self.fvd_model_name))
            self.fvd_half_precision = bool(fvd_cfg.get("half_precision", self.fvd_half_precision))
            # Optional explicit checkpoint path — avoids cdfvd's internal download
            # which saves into the ephemeral container filesystem and can produce
            # corrupt HTML files when the network request is redirected.
            raw_ckpt = fvd_cfg.get("ckpt_path")
            if raw_ckpt:
                import os
                self.fvd_ckpt_path = os.path.expanduser(str(raw_ckpt))

        # Logging
        logging.basicConfig(level=logging.INFO)

        # Initialize models
        self.logger = logger

        # Device configuration
        config_device = self.eval_cfg.get("device")
        if device is not None:
            # CLI argument takes precedence
            device_str = device
        elif config_device is not None:
            # Use config file device
            device_str = config_device
        else:
            # Auto-detect
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device_str = device_str
        self.logger.info(f"Using device: {self.device_str}")

        # 2) VGGT model
        self.logger.info("Loading VGGT model for motion extraction...")
        self.vggt_device, self.vggt_dtype = ensure_device_and_dtype(self.device_str)
        self.vggt_model = VGGT.from_pretrained(self.config.navigation.vggt_checkpoint).to(self.vggt_device)
        self.vggt_model.eval()

        # 3) Gemini reasoner (for scoring)
        self.logger.info("Initializing Gemini reasoner (may require env var)")
        try:
            self.reasoner = GeminiReasoner(self.config.reasoner, logger=self.logger)
            # create client if missing (the class has commented out client init)
            api_key = os.getenv(self.config.reasoner.api_key_env)
            if api_key:
                try:
                    import google.genai as genai
                    self.reasoner.client = genai.Client(api_key=api_key)
                except Exception as e:
                    self.logger.warning(f"Could not initialize genai client: {e}")
            else:
                self.logger.warning("GEMINI API key not set; scorer will use fallback heuristics")
        except Exception as e:
            self.logger.warning(f"GeminiReasoner init failed: {e}. A fallback scorer will be used.")
            self.reasoner = None

    def evaluate_dataset(self, dataset_json: Path, dataset_root: Optional[Path] = None, limit: Optional[int] = None) -> Path:
        dataset_json = Path(dataset_json)
        dataset_root = Path(dataset_root) if dataset_root is not None else dataset_json.parent.parent

        self.logger.info(f"Loading dataset JSON: {dataset_json} (resolving relative paths from: {dataset_json.parent})")
        with open(dataset_json, 'r') as f:
            data = json.load(f)

        results_path = self.results_dir / "results.json"
        existing_results: List[Dict[str, Any]] = []
        existing_results_by_path: Dict[str, Dict[str, Any]] = {}
        pre_result_metric_keys: set[str] = set()

        if results_path.exists():
            try:
                with open(results_path, "r") as rf:
                    loaded = json.load(rf)
                if isinstance(loaded, list):
                    existing_results = loaded
                    for entry in existing_results:
                        if not isinstance(entry, dict):
                            continue
                        vp = entry.get("video_path")
                        if vp:
                            existing_results_by_path[str(vp)] = entry
                        for k in entry.keys():
                            if k not in ("instruction", "video_path"):
                                pre_result_metric_keys.add(k)
            except Exception as e:
                self.logger.warning(f"Failed to load existing results.json: {e}")

        results = existing_results
        overwritten_result_keys: set[str] = set()
        appended_result_keys: set[str] = set()
        updated_result_keys: set[str] = set()
        any_result_updates = False

        fvd_ref_videos: List[torch.Tensor] = []
        fvd_gen_videos: List[torch.Tensor] = []

        # Disk-backed buffers for FVD clips (initialized lazily)
        ref_buffer: Optional[DiskBuffer] = None
        gen_buffer: Optional[DiskBuffer] = None

        if self.vggt_metadata_path is not None:
            if self.vggt_metadata_path.exists():
                self.logger.info(f"Using VGGT metadata: {self.vggt_metadata_path}")
            else:
                self.logger.info(f"VGGT metadata file not found; it will be created: {self.vggt_metadata_path}")

        def _resolve_rel_path(rel: str | Path) -> Path:
            relp = Path(rel)
            if relp.is_absolute():
                return relp

            # Try reference_videos subtree first (common layout for original videos)
            candidates = [
                dataset_root / "reference_videos" / relp,
                dataset_root / relp,
                dataset_root.parent / relp,
                dataset_json.parent.parent / relp,
            ]

            tried = []
            for cand in candidates:
                exists = cand.exists()
                tried.append((str(cand), exists))
                if exists:
                    self.logger.debug(f"Resolved relative path '{rel}' -> '{cand}'")
                    return cand.resolve()

            # fallback to dataset_root
            fallback = dataset_root / relp
            self.logger.debug(
                "Could not resolve '%s' in candidates. Tried: %s. Falling back to dataset_root: '%s'",
                rel,
                tried,
                str(fallback),
            )
            return fallback.resolve()

        # Temporary directory for intermediate files (frames, memmaps, etc).
        # Config option `temp_dir` can be set in the config YAML; if omitted or null,
        # we create a system tempdir via mkdtemp (previous behavior).
        cfg_temp = self.eval_cfg.get("temp_dir") if self.eval_cfg else None
        if cfg_temp:
            temp_dir = Path(os.path.expanduser(str(cfg_temp))).resolve()
            temp_dir.mkdir(parents=True, exist_ok=True)
            created_temp_dir = False
            self.logger.info(f"Using configured temporary dir: {temp_dir}")
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="imaginav_eval_"))
            created_temp_dir = True
            self.logger.info(f"Using temporary dir: {temp_dir}")

        # Import clip slicer for FVD
        from ImagiNav.video_quality_eval.metrics.fvd import _slice_into_clips

        try:
            count = 0
            for idx, entry in enumerate(data):
                if limit is not None and count >= limit:
                    break
                instruction = entry.get('instruction') or entry.get('caption', '')
                video_rel = entry.get('gen_video_path') or entry.get('video_path') or entry.get('video')

                if not video_rel:
                    # Fail-fast: dataset entries must include a generated or reference video path
                    raise RuntimeError(f"Dataset entry at index {idx} missing 'gen_video_path'/'video_path'/'video': {entry}")

                video_path = Path(video_rel)
                if not video_path.is_absolute():
                    video_path = _resolve_rel_path(video_rel)
                if not video_path.exists():
                    self.logger.warning(
                        f"Generated video not found: {video_path}. Run eval_video_gen.py first."
                    )
                    continue

                self.logger.info(f"Processing video: {video_path}  instruction: {instruction}")

                # 2. Motion fidelity
                motion_score = None
                generated_prims = []
                generated_labels = []
                generated_trajectory = []
                reference_labels = None
                reference_trajectory = None
                motion_rpe = None
                motion_rpe_trans = None
                motion_rpe_rot = None
                
                if self.enable_motion_fidelity:
                    motion = extract_video_motion(
                        video_path,
                        temp_dir,
                        self.vggt_model,
                        self.vggt_device,
                        self.vggt_dtype,
                        target_fps=self.config.navigation.video_fps,
                        analysis_subsample_step=self.analysis_subsample_step,
                        yaw_thresh_deg=self.motion_yaw_thresh_deg,
                        forward_thresh_m=self.motion_forward_thresh_m,
                        min_consecutive_frames=self.motion_min_consecutive_frames,
                    )
                    generated_prims = motion.get("primitives", [])
                    generated_labels = motion.get("labels", [])
                    generated_trajectory = motion.get("trajectory", [])
                    if generated_trajectory:
                        self.logger.info(
                            "Generated trajectory extracted: %s (steps=%d)",
                            video_path.name,
                            len(generated_trajectory),
                        )

                    # Prefer parsing the generated/reference video filename when available to capture explicit motion markers
                    prompt_prims = parse_prompt_primitives(
                        instruction,
                        max_primitives=self.prompt_max_primitives,
                        ref_name=video_path.name if video_path is not None else None,
                    )
                    ref_video_rel = (
                        entry.get("reference_video")
                        or entry.get("reference_video_path")
                        or entry.get("gt_video_path")
                    )
                    if ref_video_rel:
                        ref_path = Path(ref_video_rel)
                        if not ref_path.is_absolute():
                            ref_path = _resolve_rel_path(ref_video_rel)

                        if self.vggt_metadata_path is not None:
                            meta_entry, meta_flags = ensure_vggt_metadata_entries(
                                ref_path,
                                self.vggt_metadata_path,
                                temp_dir,
                                self.vggt_model,
                                self.vggt_device,
                                self.vggt_dtype,
                                target_fps=self.config.navigation.video_fps,
                                analysis_subsample_step=self.analysis_subsample_step,
                                yaw_thresh_deg=self.motion_yaw_thresh_deg,
                                forward_thresh_m=self.motion_forward_thresh_m,
                            )
                            reference_labels = meta_entry.get("labels")
                            reference_trajectory = meta_entry.get("trajectory")
                            if meta_flags.get("trajectory_updated"):
                                self.logger.info(
                                    "Reference trajectory extracted and saved to metadata: %s",
                                    ref_path.name,
                                )
                            else:
                                self.logger.info(
                                    "Reference trajectory loaded from metadata: %s",
                                    ref_path.name,
                                )
                            if meta_flags.get("labels_updated"):
                                self.logger.info(
                                    "Reference labels extracted and saved to metadata: %s",
                                    ref_path.name,
                                )
                            else:
                                self.logger.info(
                                    "Reference labels loaded from metadata: %s",
                                    ref_path.name,
                                )
                        else:
                            reference_labels, reference_trajectory = extract_video_labels_and_trajectory(
                                ref_path,
                                temp_dir,
                                self.vggt_model,
                                self.vggt_device,
                                self.vggt_dtype,
                                target_fps=self.config.navigation.video_fps,
                                analysis_subsample_step=self.analysis_subsample_step,
                                yaw_thresh_deg=self.motion_yaw_thresh_deg,
                                forward_thresh_m=self.motion_forward_thresh_m,
                            )

                        if self.motion_compare_mode == "labels" and reference_labels:
                            motion_score = motion_fidelity_score(reference_labels, generated_labels)

                    if reference_trajectory and generated_trajectory:
                        motion_rpe = compute_rpe(reference_trajectory, generated_trajectory)
                        if motion_rpe is not None:
                            motion_rpe_trans = motion_rpe.get("rpe_trans_m")
                            motion_rpe_rot = motion_rpe.get("rpe_rot_deg")
                            if motion_rpe_trans is not None and motion_rpe_rot is not None:
                                self.logger.info(
                                    "Sample %d: rpe_trans_m=%.6f rpe_rot_deg=%.6f",
                                    idx,
                                    float(motion_rpe_trans),
                                    float(motion_rpe_rot),
                                )

                    if motion_score is None:
                        motion_score = motion_fidelity_score(prompt_prims, generated_prims)

                    # Log motion fidelity per-sample to terminal and file
                    try:
                        self.logger.info(
                            "Sample %d: motion_fidelity=%.4f prompt_prims=%s generated_prims=%s",
                            idx,
                            float(motion_score),
                            prompt_prims,
                            generated_prims,
                        )
                    except Exception:
                        # Protect logging from causing failures
                        self.logger.debug("Failed to log motion fidelity for sample %d", idx)

                # 3. Gemini scoring
                gemini_scores = None
                if self.enable_gemini_scoring:
                    gemini_scores = gemini_score_video(
                    instruction,
                    video_path,
                    self.reasoner,
                    self.config.reasoner,
                    logger=self.logger,
                )

                # 4. Optional LPIPS/PSNR/SSIM/FVD metrics (if reference videos are provided)
                lpips_score = None
                psnr_score = None
                ssim_score = None
                
                if self.enable_lpips or self.enable_psnr or self.enable_ssim or self.enable_fvd:
                    ref_video_rel = (
                        entry.get('reference_video')
                        or entry.get('reference_video_path')
                        or entry.get('gt_video_path')
                    )
                    if ref_video_rel:
                        ref_path = Path(ref_video_rel)
                        if not ref_path.is_absolute():
                            ref_path = _resolve_rel_path(ref_video_rel)
                        if ref_path.exists():
                            try:
                                ref_video = load_video_tensor(ref_path, target_fps=self.config.navigation.video_fps)
                                gen_video = load_video_tensor(video_path, target_fps=self.config.navigation.video_fps)
                                # Temporal align by truncation to min length
                                t = min(ref_video.shape[0], gen_video.shape[0])
                                ref_video = ref_video[:t]
                                gen_video = gen_video[:t]
                                # Spatial align (resize ref to gen if needed)
                                ref_video, gen_video = align_video_tensors(ref_video, gen_video)
                                
                                if self.enable_lpips:
                                    lpips_score = compute_lpips_video(ref_video, gen_video, input_range="0_1", device=self.device_str)
                                
                                if self.enable_psnr or self.enable_ssim:
                                    psnr_val, ssim_val = compute_pixel_metrics(ref_video, gen_video, input_range="0_1", device=self.device_str)
                                    if self.enable_psnr:
                                        psnr_score = psnr_val
                                    if self.enable_ssim:
                                        ssim_score = ssim_val
                                
                                if self.enable_fvd:
                                    # Slice into clips immediately and stream to disk
                                    ref_clips = _slice_into_clips([ref_video], clip_length=16, stride=16)
                                    gen_clips = _slice_into_clips([gen_video], clip_length=16, stride=16)

                                    if ref_buffer is None:
                                        _, _, gen_h, gen_w = gen_video.shape
                                        shape_ex = (16, gen_h, gen_w, 3)
                                        ref_buffer = DiskBuffer(temp_dir, "ref_clips", shape_ex)
                                        gen_buffer = DiskBuffer(temp_dir, "gen_clips", shape_ex)

                                    for i in range(ref_clips.shape[0]):
                                        ref_buffer.add(ref_clips[i])
                                        gen_buffer.add(gen_clips[i])
                            except Exception as e:
                                # Fail-fast on metric computation errors to surface missing deps or corrupt files
                                raise RuntimeError(f"Metric computation failed for {ref_path}: {e}")
                        else:
                            raise RuntimeError(f"Reference video not found: {ref_path}")

                # 5. Persist results for this sample (merge without clobbering unrelated metrics)
                result_key = str(video_path)
                updates: Dict[str, Any] = {}
                if self.enable_motion_fidelity:
                    updates.update({
                        "prompt_primitives": prompt_prims,
                        "generated_primitives": generated_prims,
                        "generated_labels": generated_labels,
                        "reference_labels": reference_labels,
                        "motion_fidelity": motion_score,
                        "motion_rpe": motion_rpe,
                        "motion_rpe_trans_m": motion_rpe_trans,
                        "motion_rpe_rot_deg": motion_rpe_rot,
                    })
                if gemini_scores is not None:
                    updates["gemini_scores"] = gemini_scores
                if lpips_score is not None:
                    updates["lpips"] = float(lpips_score)
                if psnr_score is not None:
                    updates["psnr"] = float(psnr_score)
                if ssim_score is not None:
                    updates["ssim"] = float(ssim_score)

                if not updates:
                    count += 1
                    continue

                existing_entry = existing_results_by_path.get(result_key)
                if existing_entry is None:
                    new_entry = {
                        "instruction": instruction,
                        "video_path": result_key,
                    }
                    for k, v in updates.items():
                        if v is None:
                            continue
                        new_entry[k] = v
                        appended_result_keys.add(k)
                        updated_result_keys.add(k)
                    results.append(new_entry)
                    existing_results_by_path[result_key] = new_entry
                    any_result_updates = True
                else:
                    existing_entry["instruction"] = instruction
                    existing_entry["video_path"] = result_key
                    for k, v in updates.items():
                        if v is None:
                            continue
                        if k in existing_entry:
                            overwritten_result_keys.add(k)
                        else:
                            appended_result_keys.add(k)
                        existing_entry[k] = v
                        updated_result_keys.add(k)
                    any_result_updates = True

                count += 1
        except Exception as e:
            self.logger.exception("Evaluation loop failed: %s", e)
        finally:
            # Close any open buffers but do not remove user-configured temp dirs.
            try:
                if ref_buffer:
                    ref_buffer.close()
                if gen_buffer:
                    gen_buffer.close()
            except Exception:
                self.logger.debug("Failed to close buffers during cleanup", exc_info=True)

        # Final results saved
        final_path = results_path
        if any_result_updates:
            with open(results_path, "w") as outf:
                json.dump(results, outf, indent=2)
            self.logger.info(f"Evaluation complete. Results saved to {final_path}")
        else:
            self.logger.info(f"Evaluation complete. Results unchanged at {final_path}")

        # Aggregate metrics
        aggregate_metrics = {}

        # Compute average LPIPS if available
        lpips_scores = [r.get('lpips') for r in results if r.get('lpips') is not None]
        if lpips_scores:
            aggregate_metrics['lpips_mean'] = float(sum(lpips_scores) / len(lpips_scores))
            aggregate_metrics['lpips_std'] = float(torch.tensor(lpips_scores).std().item())
        
        # Compute average PSNR if available
        psnr_scores = [r.get('psnr') for r in results if r.get('psnr') is not None]
        if psnr_scores:
            aggregate_metrics['psnr_mean'] = float(sum(psnr_scores) / len(psnr_scores))
            aggregate_metrics['psnr_std'] = float(torch.tensor(psnr_scores).std().item())
        
        # Compute average SSIM if available
        ssim_scores = [r.get('ssim') for r in results if r.get('ssim') is not None]
        if ssim_scores:
            aggregate_metrics['ssim_mean'] = float(sum(ssim_scores) / len(ssim_scores))
            aggregate_metrics['ssim_std'] = float(torch.tensor(ssim_scores).std().item())
        
        # Compute average motion_fidelity if available
        motion_scores = [r.get('motion_fidelity') for r in results if r.get('motion_fidelity') is not None]
        if motion_scores:
            aggregate_metrics['motion_fidelity_mean'] = float(sum(motion_scores) / len(motion_scores))
            aggregate_metrics['motion_fidelity_std'] = float(torch.tensor(motion_scores).std().item())

        rpe_trans_scores = [r.get('motion_rpe_trans_m') for r in results if r.get('motion_rpe_trans_m') is not None]
        if rpe_trans_scores:
            aggregate_metrics['motion_rpe_trans_m_mean'] = float(sum(rpe_trans_scores) / len(rpe_trans_scores))
            aggregate_metrics['motion_rpe_trans_m_std'] = float(torch.tensor(rpe_trans_scores).std().item())

        rpe_rot_scores = [r.get('motion_rpe_rot_deg') for r in results if r.get('motion_rpe_rot_deg') is not None]
        if rpe_rot_scores:
            aggregate_metrics['motion_rpe_rot_deg_mean'] = float(sum(rpe_rot_scores) / len(rpe_rot_scores))
            aggregate_metrics['motion_rpe_rot_deg_std'] = float(torch.tensor(rpe_rot_scores).std().item())

        # Compute FVD
        if self.enable_fvd:
            if ref_buffer is None or gen_buffer is None or ref_buffer.count == 0:
                self.logger.warning(
                    "FVD is enabled but no reference-video clips were accumulated. "
                    "FVD will be SKIPPED. Make sure each dataset entry has a "
                    "'reference_video', 'reference_video_path', or 'gt_video_path' field "
                    "pointing to an existing file."
                )
            else:
                self.logger.info(
                    "Computing FVD on %d clips using disk-backed memory mapping...",
                    ref_buffer.count,
                )
                try:
                    ref_mmap = ref_buffer.get_memmap()
                    gen_mmap = gen_buffer.get_memmap()
                    fvd_score = compute_fvd(
                        ref_mmap,
                        gen_mmap,
                        input_range="0_255_uint8",
                        model_name=self.fvd_model_name,
                        half_precision=self.fvd_half_precision,
                        device=self.device_str,
                        ckpt_path=self.fvd_ckpt_path,
                    )
                    aggregate_metrics['fvd'] = float(fvd_score)
                    self.logger.info("FVD score: %.4f", fvd_score)
                except Exception as e:
                    self.logger.warning(f"FVD computation failed: {e}")
        
        agg_path = self.results_dir / "aggregate_metrics.json"
        existing_agg: Dict[str, Any] = {}
        existing_agg_keys: set[str] = set()
        if agg_path.exists():
            try:
                with open(agg_path, "r") as aggf:
                    loaded = json.load(aggf)
                if isinstance(loaded, dict):
                    existing_agg = loaded
                    existing_agg_keys = set(existing_agg.keys())
            except Exception as e:
                self.logger.warning(f"Failed to load existing aggregate_metrics.json: {e}")

        overwritten_agg_keys: set[str] = set()
        appended_agg_keys: set[str] = set()

        if aggregate_metrics:
            for k, v in aggregate_metrics.items():
                if k in existing_agg:
                    overwritten_agg_keys.add(k)
                else:
                    appended_agg_keys.add(k)
                existing_agg[k] = v

            with open(agg_path, "w") as aggf:
                json.dump(existing_agg, aggf, indent=2)
            self.logger.info(f"Aggregate metrics saved to {agg_path}")
        else:
            self.logger.info(f"Aggregate metrics unchanged at {agg_path}")

        untouched_result_keys = pre_result_metric_keys - updated_result_keys
        untouched_agg_keys = existing_agg_keys - set(aggregate_metrics.keys())

        if overwritten_result_keys or appended_result_keys or untouched_result_keys:
            self.logger.info(
                "Results metrics updated. Overwritten: %s | Appended: %s | Untouched: %s",
                sorted(overwritten_result_keys),
                sorted(appended_result_keys),
                sorted(untouched_result_keys),
            )
        if overwritten_agg_keys or appended_agg_keys or untouched_agg_keys:
            self.logger.info(
                "Aggregate metrics updated. Overwritten: %s | Appended: %s | Untouched: %s",
                sorted(overwritten_agg_keys),
                sorted(appended_agg_keys),
                sorted(untouched_agg_keys),
            )
        return final_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate video quality for a dataset")
    parser.add_argument("--dataset", type=str, required=False, default=None, help="Path to dataset JSON file (optional; read from config if omitted)")
    parser.add_argument("--dataset-root", type=str, help="Root folder for dataset relative paths (defaults to dataset file parent)")
    parser.add_argument("--out", type=str, help="Output results directory (defaults to video_eval_results)")
    parser.add_argument("--limit", type=int, help="Limit number of items to evaluate")
    parser.add_argument("--device", type=str, help="Device for VGGT (cpu/cuda)")
    parser.add_argument("--analysis-subsample-step", type=int, help="Subsample factor for VGGT analysis (overrides local config)")
    parser.add_argument("--config", type=str, help="Path to local video_quality_eval config YAML (optional)")
    args = parser.parse_args()

    # If --dataset is omitted, try to read it from the YAML config (prefer --config if provided)
    dataset_arg = args.dataset
    if not dataset_arg:
        cfg_path = Path(args.config) if args.config else (Path(__file__).parent / "configs" / "video_quality_eval.yaml")
        cfg_dict = load_yaml_config(cfg_path)
        dataset_arg = cfg_dict.get("dataset", {}).get("dataset_json")
        if not dataset_arg:
            parser.error("--dataset not provided and 'dataset.dataset_json' not found in the config YAML")
        # resolve relative to config base
        cfg_base = cfg_path.parent
        resolved = resolve_path(dataset_arg, cfg_base)
        if not Path(resolved).exists():
            parser.error(f"Dataset JSON referenced in config does not exist: {resolved}")
        dataset_arg = str(resolved)

    cfg = ImagiNavConfig()
    evaluator = VideoQualityEvaluator(
        cfg,
        work_dir=args.out,
        device=args.device,
        analysis_subsample_step=args.analysis_subsample_step,
        config_path=Path(args.config) if args.config else None,
    )
    evaluator.evaluate_dataset(Path(dataset_arg), dataset_root=Path(args.dataset_root) if args.dataset_root else None, limit=args.limit)
