"""
Test script for imagination.py module using default_config.yaml

This imports the LTXVideoGenerator class and tests it with real inputs.
"""

import argparse
import os
import sys

# Parse args BEFORE importing torch-related modules
parser = argparse.ArgumentParser(description="Test imagination module")
parser.add_argument("--gpu", type=int, default=None, help="GPU device ID to use")
parser.add_argument("--expert", type=str, default="left", choices=["left", "right"], help="Which expert to use for generation (left/right)")
args = parser.parse_args()

# Set GPU before any torch imports - MUST happen before importing imagination.py
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Setting CUDA_VISIBLE_DEVICES={args.gpu}")


import logging
from pathlib import Path

from ImagiNav.core.config import ImagiNavConfig
from ImagiNav.models.imagination import LTXVideoGenerator

# ============ Configuration ============
CONFIG_PATH = "ImagiNav/configs/default_config.yaml"
IMAGE_PATH = "ImagiNav/LTX-Video-Trainer/datasets/validation_images/IMG_4370_2808_2929_MOVE_FORWARD_TURN_LEFT_firstframe.png"
PROMPT = "POV, Dolly forward and pan left towards the bushes on the left."
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted, collision"
OUTPUT_DIR = Path("test_outputs/imagination_module_test")
SEED = 42

# ============ Setup Logging ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_imagination")

if args.gpu is not None:
    logger.info(f"Using GPU: {args.gpu}")

# ============ Load Config ============
logger.info(f"Loading config from: {CONFIG_PATH}")
config = ImagiNavConfig.from_yaml(CONFIG_PATH)
# Apply requested device and LoRA overrides
if args.gpu is not None:
    # When CUDA_VISIBLE_DEVICES is set, the visible device index is 0
    config.imagination.device = "cuda:0"
    logger.info(f"Set imagination.device -> {config.imagination.device} (via --gpu)")


# Force num_frames to 121 for testing (override default_config without editing it)
config.imagination.num_frames = 121
logger.info(f"Overriding num_frames -> {config.imagination.num_frames}")

logger.info(f"Imagination config:")
logger.info(f"  base_model: {config.imagination.base_model}")
logger.info(f"  lora_mode: {config.imagination.lora_mode}")
logger.info(f"  left_forward_model_path: {config.imagination.left_forward_model_path}")
logger.info(f"  right_model_path: {config.imagination.right_model_path}")
logger.info(f"  num_inference_steps: {config.imagination.num_inference_steps}")
logger.info(f"  guidance_scale: {config.imagination.guidance_scale}")
logger.info(f"  height: {config.imagination.height}")
logger.info(f"  width: {config.imagination.width}")
logger.info(f"  num_frames: {config.imagination.num_frames}")

# ============ Initialize Generator ============
logger.info("Initializing LTXVideoGenerator...")
generator = LTXVideoGenerator(config=config.imagination, logger=logger)

# ============ Generate Video ============
logger.info(f"\n{'='*50}")
logger.info(f"Generating video...")
logger.info(f"  Image: {IMAGE_PATH}")
logger.info(f"  Prompt: {PROMPT}")
logger.info(f"{'='*50}")

video_path = generator.generate_video(
    image_path=IMAGE_PATH,
    prompt=PROMPT,
    output_dir=OUTPUT_DIR,
    negative_prompt=NEGATIVE_PROMPT,
    seed=SEED,
    expert_type=args.expert,
)

logger.info(f"\n✅ Video saved to: {video_path}")

# ============ Cleanup ============
generator.cleanup()
logger.info("Test complete!")
