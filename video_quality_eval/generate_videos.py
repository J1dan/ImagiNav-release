"""Offline video generation for video quality eval datasets."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from ImagiNav.core.config import ImagiNavConfig
from ImagiNav.models.imagination import LTXVideoGenerator

logger = logging.getLogger("imaginav.video_gen")


def _apply_overrides(cfg: ImagiNavConfig, overrides: Dict[str, Any]) -> None:
    for section in ["imagination", "navigation", "logging", "reasoner"]:
        if section in overrides and hasattr(cfg, section):
            section_obj = getattr(cfg, section)
            for key, value in overrides[section].items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)


def generate_videos_from_config(config_path: Path) -> Path:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Simplified: expect a single `dataset_path` key pointing to the dataset.json file
    # Paths inside dataset.json will be resolved relative to dataset.json's parent directory.
    imag_cfg = ImagiNavConfig()
    _apply_overrides(imag_cfg, cfg)

    dataset_json = Path(cfg.get("dataset_path") or cfg.get("dataset_json"))
    if dataset_json is None:
        raise ValueError("Config must include 'dataset_path' pointing to a dataset JSON file")
    if not dataset_json.exists():
        raise ValueError(f"dataset_path does not exist: {dataset_json}")
    dataset_root = dataset_json.parent
    logger.info(f"Dataset JSON: {dataset_json} (paths inside resolved relative to: {dataset_root})")
    output_dir = Path(cfg.get("output_dir") or "video_eval_generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_dataset_json = Path(cfg.get("output_dataset_json") or output_dir / "dataset_with_videos.json")
    limit = cfg.get("limit")
    seed = cfg.get("seed")
    negative_prompt = cfg.get("negative_prompt", "blurry, low quality, static, stationary")
    overwrite = bool(cfg.get("overwrite", False))

    logger.info("Initializing LTX-Video generator...")
    generator = LTXVideoGenerator(imag_cfg.imagination, logger=logger)

    with open(dataset_json, "r") as f:
        data = json.load(f)

    out_items = []
    count = 0
    for idx, entry in enumerate(data):
        if limit is not None and count >= limit:
            break

        image_rel = entry.get("image_path") or entry.get("image")
        instruction = entry.get("instruction") or entry.get("caption", "")
        if not image_rel:
            logger.warning("Skipping entry with no image_path")
            continue

        # Resolve image path flexibly: try dataset parent, its parent, and dataset_root
        def _resolve_rel(rel: str | Path) -> Path:
            r = Path(rel)
            if r.is_absolute():
                return r
            candidates = [
                dataset_json.parent / r,
                dataset_json.parent.parent / r,
                dataset_root / r,
                dataset_root.parent / r,
            ]
            for cand in candidates:
                if cand.exists():
                    logger.debug(f"Resolved relative path '{rel}' -> '{cand}'")
                    return cand.resolve()
            logger.debug(f"Could not resolve '{rel}' in candidates. Falling back to dataset parent: '{dataset_json.parent / r}'")
            return (dataset_json.parent / r).resolve()

        image_path = _resolve_rel(image_rel)
        if not image_path.exists():
            logger.warning(f"Image not found (tried dataset parent/grandparent/dataset_root): {image_path}; skipping")
            continue

        run_dir = output_dir / f"run_{count:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        video_path = run_dir / f"02_generated_video.{imag_cfg.imagination.output_format}"
        if video_path.exists() and not overwrite:
            logger.info(f"Video already exists, skipping generation: {video_path}")
        else:
            logger.info(f"Generating video for {image_rel}")
            try:
                video_path = generator.generate_video(
                    str(image_path),
                    instruction,
                    run_dir,
                    negative_prompt=negative_prompt,
                    seed=seed,
                )
            except Exception as e:
                logger.error(f"Generation failed for {image_rel}: {e}")
                continue

        out_item = dict(entry)
        try:
            out_item["video_path"] = str(video_path.relative_to(output_dir))
        except ValueError:
            out_item["video_path"] = str(video_path)
        out_item["caption"] = instruction
        out_items.append(out_item)
        count += 1

    with open(output_dataset_json, "w") as f:
        json.dump(out_items, f, indent=2)

    logger.info(f"Saved dataset with videos to: {output_dataset_json}")
    return output_dataset_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate videos offline for evaluation datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to generation config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_videos_from_config(Path(args.config))


if __name__ == "__main__":
    main()
