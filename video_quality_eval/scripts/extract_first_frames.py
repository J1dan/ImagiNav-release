"""Extract first-frame images from reference videos."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import cv2

from ImagiNav.video_quality_eval.utils.config_utils import load_yaml_config, resolve_path

logger = logging.getLogger("imaginav.extract_first_frames")


def _resolve_rel_path(rel: str | Path, dataset_json: Path, dataset_root: Path) -> Path:
    relp = Path(rel)
    if relp.is_absolute():
        return relp
    
    # Try multiple candidate locations
    candidates = [
        dataset_root / relp,
        dataset_root / "reference_videos" / relp,  # Added: check in reference_videos subdirectory
        dataset_json.parent / relp,
        dataset_json.parent.parent / relp,
        dataset_root.parent / relp,
    ]
    
    for cand in candidates:
        if cand.exists():
            logger.debug(f"Resolved '{rel}' -> '{cand}'")
            return cand.resolve()
    
    # Fallback
    return (dataset_root / relp).resolve()


def _mirror_first_frame_path(ref_video_rel: str, first_frame_dir: str = "first_frame_images") -> str:
    """Mirror reference_videos sub-structure under first_frame_images.

    Example:
      reference_videos/apartment_clips/IMG_123.mp4
      -> first_frame_images/apartment_clips/IMG_123_firstframe.png
    """
    ref_path = Path(ref_video_rel)
    parts = list(ref_path.parts)
    subdir_parts: list[str] = []
    
    if "reference_videos" in parts:
        idx = parts.index("reference_videos")
        subdir_parts = parts[idx + 1:-1]
    else:
        if ref_path.parent != Path("."):
            subdir_parts = list(ref_path.parent.parts)
    
    first_frame_name = f"{ref_path.stem}_firstframe.png"
    
    if subdir_parts:
        return str(Path(first_frame_dir) / Path(*subdir_parts) / first_frame_name)
    return str(Path(first_frame_dir) / first_frame_name)


def _extract_first_frame(video_path: Path, output_path: Path, overwrite: bool = False) -> bool:
    if output_path.exists() and not overwrite:
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return False

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        logger.warning(f"No frames in video: {video_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        logger.warning(f"Failed to write first frame: {output_path}")
        return False
    return True


def extract_first_frames_from_config(config_path: Path) -> None:
    cfg = load_yaml_config(config_path)
    config_base = config_path.parent

    dataset_cfg = cfg.get("dataset", {})
    dataset_json = resolve_path(dataset_cfg.get("dataset_json"), config_base)
    if dataset_json is None:
        raise ValueError("Config must include dataset.dataset_json")
    if not dataset_json.exists():
        raise ValueError(f"dataset_json does not exist: {dataset_json}")

    dataset_root = resolve_path(dataset_cfg.get("dataset_root"), config_base) or dataset_json.parent
    default_first_frame_dir = resolve_path(dataset_cfg.get("first_frame_dir"), config_base)

    overwrite = bool(cfg.get("generation", {}).get("overwrite_first_frames", False))

    with open(dataset_json, "r") as f:
        data = json.load(f)

    converted = False
    for entry in data:
        # Accept media_path / caption style datasets and normalize them to reference_video
        if not any(k in entry for k in ("reference_video", "reference_video_path", "gt_video_path")):
            if "media_path" in entry:
                entry["reference_video"] = entry["media_path"]
                converted = True

        ref_video_rel = (
            entry.get("reference_video")
            or entry.get("reference_video_path")
            or entry.get("gt_video_path")
        )
        if not ref_video_rel:
            logger.warning("Skipping entry with no reference_video_path or media_path")
            continue

        # Ensure there is a place to write first frames
        first_frame_rel = entry.get("first_frame_path") or entry.get("first_frame")
        if not first_frame_rel:
            if default_first_frame_dir is None:
                logger.warning("No first_frame_path and no dataset.first_frame_dir configured; skipping")
                continue
            # Mirror the subdirectory structure from reference_videos
            first_frame_rel = _mirror_first_frame_path(ref_video_rel, str(default_first_frame_dir.name if default_first_frame_dir else "first_frame_images"))
            entry["first_frame_path"] = first_frame_rel
            converted = True

        ref_video_path = _resolve_rel_path(ref_video_rel, dataset_json, dataset_root)
        if not ref_video_path.exists():
            logger.warning(f"Reference video not found: {ref_video_path}")
            continue

        first_frame_path = _resolve_rel_path(entry["first_frame_path"], dataset_json, dataset_root)
        if _extract_first_frame(ref_video_path, first_frame_path, overwrite=overwrite):
            logger.info(f"Wrote first frame: {first_frame_path}")

    # If conversion happened, write out a new dataset JSON alongside the original
    if converted:
        out_path = dataset_json.parent / (dataset_json.stem + "_with_firstframes.json")
        with open(out_path, "w") as outf:
            json.dump(data, outf, indent=2)
        logger.info(f"Wrote converted dataset JSON with first_frame_path to: {out_path}")

    logger.info("First-frame extraction completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract first-frame images from reference videos")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    extract_first_frames_from_config(Path(args.config))


if __name__ == "__main__":
    main()
