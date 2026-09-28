"""Generate evaluation videos from first-frame images using LTX-Video.
Refactored to match GPU isolation logic from test_imagination_module.py
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from pathlib import Path

# ==============================================================================
# PHASE 1: PRE-IMPORT SETUP (Learn from test_imagination_module.py)
# We must set CUDA_VISIBLE_DEVICES before importing torch or ImagiNav
# ==============================================================================

def setup_gpu_env(config_path: Path):
    """
    Reads the config YAML strictly to find the 'device' setting.
    Sets CUDA_VISIBLE_DEVICES so bitsandbytes loads on the correct card.
    Returns: True if we remapped the device, False otherwise.
    """
    try:
        with open(config_path, 'r') as f:
            raw_cfg = yaml.safe_load(f)
        
        # Check standard locations for device in your config structure
        device_str = raw_cfg.get("generation", {}).get("device")
            
        # If user requested a specific GPU index (e.g., "cuda:1")
        if device_str and isinstance(device_str, str) and device_str.startswith("cuda:"):
            parts = device_str.split(":")
            if len(parts) > 1 and parts[1].isdigit():
                gpu_idx = parts[1]
                
                # CRITICAL: Isolate the GPU so it appears as 'cuda:0' to the script
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
                print(f"[Pre-Check] Found device '{device_str}' in config.")
                print(f"[Pre-Check] Setting CUDA_VISIBLE_DEVICES={gpu_idx}")
                return True
                
    except Exception as e:
        print(f"[Pre-Check] Warning: Could not pre-parse config for device: {e}")
        
    return False

# Parse args immediately to get config path
parser = argparse.ArgumentParser(description="Generate evaluation videos")
parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
# We parse known args only, in case other scripts call this differently
args, unknown = parser.parse_known_args()

# Execute GPU setup BEFORE any other imports
IS_REMAPPED = setup_gpu_env(Path(args.config))


# ==============================================================================
# PHASE 2: STANDARD IMPORTS & LOGIC
# Now safe to import torch, ImagiNav, etc.
# ==============================================================================

import json
import logging
import random
import shutil
from typing import Any, Dict, Optional

from ImagiNav.core.config import ImagiNavConfig
from ImagiNav.models.imagination import LTXVideoGenerator
from ImagiNav.video_quality_eval.utils.config_utils import load_yaml_config, resolve_path

logger = logging.getLogger("imaginav.eval_video_gen")

def _resolve_rel_path(rel: str | Path, dataset_json: Path, dataset_root: Path) -> Path:
    relp = Path(rel)
    if relp.is_absolute():
        return relp
    candidates = [
        dataset_root / relp,
        dataset_json.parent / relp,
        dataset_json.parent.parent / relp,
        dataset_root.parent / relp,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return (dataset_root / relp).resolve()

def _select_expert_type(ref_video_path: Path, rng) -> str:
    if not ref_video_path:
        raise ValueError("No reference video path provided")
    name = ref_video_path.name.upper()
    if "TURN_RIGHT" in name:
        return "right"
    if "TURN_LEFT" in name:
        return "left"
    if "MOVE_FORWARD" in name:
        return rng.choice(["left", "right"])
    raise ValueError(f"Could not determine expert type from: {ref_video_path}")

def _mirror_gen_video_rel(ref_video_rel: str, synthetic_videos_dir: str = "synthetic_videos") -> str:
    ref_path = Path(ref_video_rel)
    parts = list(ref_path.parts)
    subdir_parts = []
    if "reference_videos" in parts:
        idx = parts.index("reference_videos")
        subdir_parts = parts[idx + 1:-1]
    else:
        if ref_path.parent != Path("."):
            subdir_parts = list(ref_path.parent.parts)
    gen_name = f"gen-{ref_path.name}"
    if subdir_parts:
        return str(Path(synthetic_videos_dir) / Path(*subdir_parts) / gen_name)
    return str(Path(synthetic_videos_dir) / gen_name)


def _derive_output_dataset_json(gen_dataset_json: Path, synthetic_videos_dir: str) -> Path:
    suffix = Path(synthetic_videos_dir).name
    stem = gen_dataset_json.stem
    if stem.endswith(f"_{suffix}"):
        return gen_dataset_json
    return gen_dataset_json.with_name(f"{stem}_{suffix}{gen_dataset_json.suffix}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    config_path = Path(args.config)
    eval_cfg = load_yaml_config(config_path)
    config_base = config_path.parent

    # 1. Load Base System Config (mimicking test_imagination_module.py)
    default_config_path = Path("ImagiNav/configs/default_config.yaml")
    if not default_config_path.exists():
        default_config_path = Path("configs/default_config.yaml")
    
    logger.info(f"Loading base system config from: {default_config_path}")
    sys_config = ImagiNavConfig.from_yaml(default_config_path)

    # 2. Apply Overrides
    gen_cfg = eval_cfg.get("generation", {})
    dataset_cfg = eval_cfg.get("dataset", {})

    # --- DEVICE SETTING (The Fix) ---
    if IS_REMAPPED:
        # Since we set CUDA_VISIBLE_DEVICES to a single index, 
        # that GPU becomes "cuda:0" to the process.
        sys_config.imagination.device = "cuda:0"
        logger.info(f"Device was remapped via env var. Using internal device: {sys_config.imagination.device}")
    else:
        # Fallback if no specific index was found (e.g. cpu or just 'cuda')
        sys_config.imagination.device = gen_cfg.get("device", "cuda")

    # --- Dimensions ---
    if gen_cfg.get("base_model"):
        sys_config.imagination.base_model = gen_cfg["base_model"]

    if gen_cfg.get("full_checkpoint"):
        full_checkpoint = resolve_path(gen_cfg.get("full_checkpoint"), config_base)
        if not full_checkpoint or not full_checkpoint.exists():
            raise RuntimeError(f"Configured full checkpoint does not exist: {gen_cfg.get('full_checkpoint')}")
        sys_config.imagination.full_checkpoint = str(full_checkpoint)

    video_dims = gen_cfg.get("video_dims", [480, 256, 121])
    sys_config.imagination.width = video_dims[0]
    sys_config.imagination.height = video_dims[1]
    sys_config.imagination.num_frames = video_dims[2]

    # --- LoRA ---
    lora_mode = gen_cfg.get("lora_mode", "dual")
    if lora_mode == "uni":
        logger.info("generation.lora_mode='uni' is accepted as an alias for 'single'")
        lora_mode = "single"
    if lora_mode not in {"dual", "single", "none"}:
        raise RuntimeError(
            "generation.lora_mode must be one of: dual, single, uni, none "
            f"(got {lora_mode!r})"
        )
    sys_config.imagination.lora_mode = lora_mode

    if lora_mode == "none":
        logger.info("LoRA mode is 'none': base-model ablation — no LoRA weights will be loaded")
    elif lora_mode == "single":
        lora_uni = resolve_path(gen_cfg.get("lora_uni") or gen_cfg.get("lora_turn_left"), config_base)
        if not lora_uni:
            raise RuntimeError(
                "generation.lora_uni or generation.lora_turn_left is required "
                "when lora_mode='single'/'uni' but was not set"
            )
        if not lora_uni.exists():
            raise RuntimeError(f"Configured single LoRA file does not exist: {lora_uni}")
        sys_config.imagination.model_path = str(lora_uni)
        logger.info(f"Single LoRA mode: using one adapter for all videos — {lora_uni}")
    else:
        lora_left = resolve_path(gen_cfg.get("lora_turn_left"), config_base)
        lora_right = resolve_path(gen_cfg.get("lora_turn_right"), config_base)

        # Validate LoRA paths: fail-fast if either is missing or doesn't exist
        missing: list[str] = []
        if not lora_left:
            missing.append("generation.lora_turn_left (not set or could not be resolved)")
        elif not lora_left.exists():
            missing.append(str(lora_left))

        if not lora_right:
            missing.append("generation.lora_turn_right (not set or could not be resolved)")
        elif not lora_right.exists():
            missing.append(str(lora_right))

        if missing:
            raise RuntimeError("Configured LoRA files are missing or invalid:\n" + "\n".join(missing))

        sys_config.imagination.left_forward_model_path = str(lora_left)
        sys_config.imagination.right_model_path = str(lora_right)

    sys_config.imagination.num_inference_steps = int(gen_cfg.get("inference_steps", 50))
    sys_config.imagination.guidance_scale = float(gen_cfg.get("guidance_scale", 3.8))

    # 3. Initialize Generator
    logger.info("Initializing LTXVideoGenerator...")
    generator = LTXVideoGenerator(sys_config.imagination, logger=logger)

    # 4. Loop and Generate
    dataset_json = resolve_path(dataset_cfg.get("dataset_json"), config_base)
    gen_dataset_json = resolve_path(gen_cfg.get("gen_dataset_json") or dataset_cfg.get("dataset_json"), config_base)
    dataset_root = resolve_path(dataset_cfg.get("dataset_root"), config_base) or gen_dataset_json.parent
    synthetic_videos_dir = gen_cfg.get("synthetic_videos_dir", "synthetic_videos")
    overwrite = bool(gen_cfg.get("overwrite", False))
    limit = gen_cfg.get("limit")
    seed = gen_cfg.get("seed")
    negative_prompt = gen_cfg.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted, collision")
    random_seed = gen_cfg.get("random_seed", 42)
    rng = random.Random(random_seed)

    with open(gen_dataset_json, "r") as f:
        data = json.load(f)

    updated_gen_paths = False
    count = 0
    
    logger.info(f"Starting generation loop for {len(data)} entries...")

    for entry in data:
        if limit is not None and count >= limit:
            break

        # Path Resolution
        instruction = entry.get("instruction") or entry.get("caption", "")
        ref_video_rel = entry.get("reference_video") or entry.get("reference_video_path") or entry.get("gt_video_path")
        
        gen_video_rel = entry.get("gen_video_path") or entry.get("video_path")
        if not gen_video_rel:
            if ref_video_rel:
                gen_video_rel = _mirror_gen_video_rel(ref_video_rel, synthetic_videos_dir)
                entry["gen_video_path"] = gen_video_rel
                updated_gen_paths = True
            else:
                continue

        gen_video_path = _resolve_rel_path(gen_video_rel, gen_dataset_json, dataset_root)
        if gen_video_path.exists() and not overwrite:
            logger.info(f"[{count}] Skipping existing generated video: {gen_video_path} (overwrite={overwrite})")
            count += 1
            continue
            
        first_frame_rel = entry.get("first_frame_path") or entry.get("first_frame")
        if not first_frame_rel:
            continue
        first_frame_path = _resolve_rel_path(first_frame_rel, gen_dataset_json, dataset_root)
        if not first_frame_path.exists():
            continue

        ref_video_path = _resolve_rel_path(ref_video_rel, gen_dataset_json, dataset_root) if ref_video_rel else None
        # For base-model ablation (lora_mode='none') or uni mode, expert_type is irrelevant — no LoRA switching
        if lora_mode in {"none", "uni"}:
            expert_type = None
        else:
            expert_type = _select_expert_type(ref_video_path, rng)
        per_seed = seed if seed is not None else rng.randint(0, 2**31 - 1)

        try:
            logger.info(f"[{count}] Generating: {instruction} (Expert: {expert_type})")
            
            temp_video_path = generator.generate_video(
                image_path=str(first_frame_path),
                prompt=instruction,
                output_dir=gen_video_path.parent, 
                negative_prompt=negative_prompt,
                seed=per_seed,
                expert_type=expert_type
            )

            # Move file
            temp_video_path = Path(temp_video_path)
            gen_video_path.parent.mkdir(parents=True, exist_ok=True)
            if temp_video_path.resolve() != gen_video_path.resolve():
                shutil.move(str(temp_video_path), str(gen_video_path))
                
        except Exception as e:
            logger.error(f"Generation failed for {first_frame_path}: {e}")
            continue

        count += 1

    logger.info("Video generation completed.")
    generator.cleanup()

    if updated_gen_paths:
        out_path = _derive_output_dataset_json(gen_dataset_json, synthetic_videos_dir)
        with open(out_path, "w") as outf:
            json.dump(data, outf, indent=2)
        logger.info(f"Wrote dataset with mirrored gen_video_path: {out_path}")

if __name__ == "__main__":
    main()
