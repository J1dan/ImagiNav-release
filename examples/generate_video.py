"""Generate one imagined navigation video from a reference image."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Reference PNG or JPEG")
    parser.add_argument("--prompt", required=True, help="POV-style motion prompt")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default_config.yaml",
        help="ImagiNav YAML configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "demo",
        help="Directory for the generated video",
    )
    parser.add_argument("--expert", choices=("left", "right"), default="left")
    parser.add_argument("--gpu", type=int, help="CUDA device exposed to the process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--negative-prompt",
        default="worst quality, inconsistent motion, blurry, jittery, distorted, collision",
    )
    return parser.parse_args()


def _resolve_config_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT.parent / path).resolve())


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Import GPU libraries only after CUDA_VISIBLE_DEVICES has been configured.
    from ImagiNav.core.config import ImagiNavConfig
    from ImagiNav.models.imagination import LTXVideoGenerator

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("imaginav.example")

    image_path = args.image.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {image_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    config = ImagiNavConfig.from_yaml(str(config_path))
    for field_name in (
        "model_path",
        "left_forward_model_path",
        "right_model_path",
        "full_checkpoint",
    ):
        field_value = getattr(config.imagination, field_name)
        if field_value:
            setattr(config.imagination, field_name, _resolve_config_path(field_value))
    if args.gpu is not None:
        config.imagination.device = "cuda:0"

    generator = LTXVideoGenerator(config=config.imagination, logger=logger)
    try:
        video_path = generator.generate_video(
            image_path=str(image_path),
            prompt=args.prompt,
            output_dir=output_dir,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            expert_type=args.expert,
        )
    finally:
        generator.cleanup()

    logger.info("Video saved to %s", video_path)


if __name__ == "__main__":
    main()
