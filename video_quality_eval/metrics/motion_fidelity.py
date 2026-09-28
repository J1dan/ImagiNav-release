"""Motion fidelity metric based on normalized Levenshtein distance."""

from __future__ import annotations

from pathlib import Path
import re
import json
import math
from typing import Dict, List, Optional, Tuple, Any

_ACTION_ALIASES = {
    "F": "forward",
    "TL": "left",
    "TR": "right",
    "S": "static",
    "rotate_left": "left",
    "rotate_right": "right",
}


def normalize_action_labels(actions: List[str]) -> List[str]:
    """Normalize action labels to canonical strings."""
    return [_ACTION_ALIASES.get(a, a) for a in actions]


def levenshtein_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """Compute Levenshtein (edit) distance between two lists of strings."""
    if seq_a == seq_b:
        return 0
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    len_a = len(seq_a)
    len_b = len(seq_b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[len_a][len_b]


def run_length_encode_labels(labels: List[str]) -> List[str]:
    """Run-length-encode label list and drop consecutive repeats (return primitives sequence)."""
    if not labels:
        return []
    result = [labels[0]]
    for l in labels[1:]:
        if l != result[-1]:
            result.append(l)
    return result


from ImagiNav.vggt.imaginav_scripts.motion_utils import (
    classify_raw_steps_from_trajectory,
)

# Backwards-compatible wrapper
def classify_steps_from_trajectory(trajectory: List[dict], yaw_thresh_deg: float = 2.0, forward_thresh_m: float = 0.012) -> List[str]:
    return classify_raw_steps_from_trajectory(trajectory, yaw_thresh_deg, forward_thresh_m)



def get_clean_motion_sequence(labels: List[str], min_len: int = 5) -> List[List]:
    """Clean per-frame label sequence into RLE runs with denoising.

    Steps:
    1. Run-length encode labels into [(label, length), ...].
    2. Iteratively merge short segments (< min_len) into neighbors until stable.
    3. Merge adjacent runs with the same label.

    Returns a list of [label, length].
    """
    if not labels:
        return []

    # Initial RLE
    runs = []
    curr = labels[0]
    cnt = 1
    for l in labels[1:]:
        if l == curr:
            cnt += 1
        else:
            runs.append([curr, cnt])
            curr = l
            cnt = 1
    runs.append([curr, cnt])

    # Denoise: merge short segments
    changed = True
    while changed:
        changed = False
        new_runs = []
        i = 0
        while i < len(runs):
            label, length = runs[i]
            if length < min_len and len(runs) > 1:
                changed = True
                # Merge into previous if exists
                if new_runs:
                    new_runs[-1][1] += length
                # Else merge into next
                elif i + 1 < len(runs):
                    runs[i + 1][1] += length
                else:
                    new_runs.append([label, length])
            else:
                new_runs.append([label, length])
            i += 1
        runs = new_runs

    # Final pass: merge adjacent identical labels
    final = []
    if not runs:
        return []
    curr_label, curr_len = runs[0]
    for label, length in runs[1:]:
        if label == curr_label:
            curr_len += length
        else:
            final.append([curr_label, curr_len])
            curr_label, curr_len = label, length
    final.append([curr_label, curr_len])
    return final


def _parse_primitives_from_filename(ref_name: str, max_primitives: int = 2) -> List[str]:
    """Parse motion primitives from a reference video filename.

    Recognizes only the canonical markers: MOVE_FORWARD -> 'F', TURN_LEFT -> 'TL', TURN_RIGHT -> 'TR'.
    Returns labels in the order they appear in the filename (left-to-right)."""
    if not ref_name:
        return []
    txt = str(ref_name).upper()
    patterns = [
        (r"TURN_LEFT", "TL"),
        (r"TURN_RIGHT", "TR"),
        (r"MOVE_FORWARD", "F"),
        (r"FORWARD", "F"),
    ]
    occ: List[tuple[int, str]] = []
    for pat, label in patterns:
        for m in re.finditer(pat, txt):
            occ.append((m.start(), label))
    occ.sort(key=lambda x: x[0])
    labels = [lab for _, lab in occ]
    # remove consecutive duplicates
    compact: List[str] = []
    for l in labels:
        if not compact or compact[-1] != l:
            compact.append(l)
    return compact[:max_primitives]


def parse_prompt_primitives(instruction: str, max_primitives: int = 2, ref_name: Optional[str] = None) -> List[str]:
    """Extract up to `max_primitives` motion primitives using only the reference filename.

    NOTE: Per request, we no longer parse the instruction text. If `ref_name` is provided and contains
    canonical markers (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT) they are returned in file-order. If no
    markers are found, returns the default ["F"].
    """
    # If a reference filename is provided, use it. Do not parse the instruction string.
    if ref_name:
        from_file = _parse_primitives_from_filename(ref_name, max_primitives=max_primitives)
        if from_file:
            return from_file

    # Default fallback when no ref_name markers are available
    return ["F"]


def load_vggt_metadata(metadata_path: Path) -> Dict[str, Dict]:
    """Load VGGT metadata JSON and index by multiple keys for lookup."""
    import json

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"VGGT metadata not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        data = json.load(f)

    indexed: Dict[str, Dict] = {}
    for item in data:
        keys = []
        media_path = item.get("media_path")
        filename = item.get("filename")
        if media_path:
            keys.append(str(media_path))
            keys.append(Path(media_path).name)
        if filename:
            keys.append(str(filename))
            keys.append(Path(filename).name)
        for k in keys:
            indexed[k] = item
    return indexed


def _load_vggt_metadata_list(metadata_path: Path) -> List[Dict[str, Any]]:
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return []
    with open(metadata_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"VGGT metadata JSON must be a list: {metadata_path}")


def _index_vggt_metadata_list(metadata_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in metadata_list:
        keys = []
        media_path = item.get("media_path")
        filename = item.get("filename")
        if media_path:
            keys.append(str(media_path))
            keys.append(Path(media_path).name)
        if filename:
            keys.append(str(filename))
            keys.append(Path(filename).name)
        for k in keys:
            indexed[k] = item
    return indexed


def _clean_trajectory_for_json(trajectory_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    for step in trajectory_list:
        clean_step: Dict[str, Any] = {}
        for k, v in step.items():
            if isinstance(v, (int, float)):
                clean_step[k] = float(v)
            else:
                try:
                    clean_step[k] = float(v)
                except Exception:
                    clean_step[k] = v
        clean.append(clean_step)
    return clean


def extract_video_labels_and_trajectory(
    video_path: Path,
    temp_dir: Path,
    vggt_model,
    device,
    dtype,
    target_fps: float | None = None,
    analysis_subsample_step: int = 6,
    yaw_thresh_deg: float = 2.0,
    forward_thresh_m: float = 0.012,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extract per-frame labels and relative trajectory from a video using VGGT.

    Returns:
        labels: per-frame labels (length N-1)
        trajectory: list of dicts containing yaw_deg, forward_m, lateral_m, etc.
    """
    from ImagiNav.vggt.imaginav_scripts.infer_move import (
        extract_frames_from_video,
        sorted_image_list,
        get_motion_trajectory,
    )

    if analysis_subsample_step < 1:
        raise ValueError("analysis_subsample_step must be >= 1")

    video_path = Path(video_path)
    frames_dir = Path(temp_dir) / (video_path.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    target_fps = target_fps or 24.0
    extract_frames_from_video(video_path, frames_dir, target_fps)

    imgs = sorted_image_list(frames_dir)
    if len(imgs) < 2:
        return [], []

    if analysis_subsample_step > 1:
        imgs = imgs[::analysis_subsample_step]

    if len(imgs) < 2:
        return [], []

    trajectory = get_motion_trajectory(vggt_model, device, dtype, imgs)
    labels = classify_steps_from_trajectory(trajectory, yaw_thresh_deg, forward_thresh_m)
    return labels, trajectory


def ensure_vggt_metadata_entries(
    video_path: Path,
    metadata_path: Path,
    temp_dir: Path,
    vggt_model,
    device,
    dtype,
    target_fps: float | None = None,
    analysis_subsample_step: int = 6,
    yaw_thresh_deg: float = 2.0,
    forward_thresh_m: float = 0.012,
) -> tuple[Dict[str, Any], Dict[str, bool]]:
    """Ensure that VGGT metadata has `labels` and `trajectory` for a given video.

    If missing, compute and append/update the metadata file.
    """
    metadata_path = Path(metadata_path)
    metadata_list = _load_vggt_metadata_list(metadata_path)
    indexed = _index_vggt_metadata_list(metadata_list)

    key = str(Path(video_path))
    entry = indexed.get(key) or indexed.get(Path(video_path).name)
    if entry is None:
        entry = {
            "media_path": str(Path(video_path)),
            "filename": Path(video_path).name,
        }
        metadata_list.append(entry)

    needs_labels = "labels" not in entry
    needs_traj = "trajectory" not in entry

    if needs_labels or needs_traj:
        labels, trajectory = extract_video_labels_and_trajectory(
            video_path,
            temp_dir,
            vggt_model,
            device,
            dtype,
            target_fps=target_fps,
            analysis_subsample_step=analysis_subsample_step,
            yaw_thresh_deg=yaw_thresh_deg,
            forward_thresh_m=forward_thresh_m,
        )
        if needs_labels:
            entry["labels"] = labels
        if needs_traj:
            entry["trajectory"] = _clean_trajectory_for_json(trajectory)

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata_list, f, indent=2)

    return entry, {"labels_updated": needs_labels, "trajectory_updated": needs_traj}


def compute_rpe(
    reference_trajectory: List[Dict[str, Any]],
    generated_trajectory: List[Dict[str, Any]],
) -> Optional[Dict[str, float]]:
    """Compute standard Relative Pose Error (RPE) with split translation/rotation.

    Returns per-step averages:
      - rpe_trans_m: Euclidean distance in meters (forward/lateral).
      - rpe_rot_deg: Absolute yaw error in degrees (wrapped to [-180, 180]).
    """
    if not reference_trajectory or not generated_trajectory:
        return None

    n = min(len(reference_trajectory), len(generated_trajectory))
    if n == 0:
        return None

    trans_err_sum = 0.0
    rot_err_sum = 0.0
    for i in range(n):
        ref = reference_trajectory[i]
        gen = generated_trajectory[i]

        ref_f = float(ref.get("forward_m", 0.0))
        ref_l = float(ref.get("lateral_m", 0.0))
        gen_f = float(gen.get("forward_m", 0.0))
        gen_l = float(gen.get("lateral_m", 0.0))

        df = ref_f - gen_f
        dl = ref_l - gen_l
        trans_err_sum += math.sqrt(df * df + dl * dl)

        ref_y = math.radians(float(ref.get("yaw_deg", 0.0)))
        gen_y = math.radians(float(gen.get("yaw_deg", 0.0)))
        dy = ref_y - gen_y
        dy = (dy + math.pi) % (2.0 * math.pi) - math.pi
        rot_err_sum += abs(math.degrees(dy))

    return {
        "rpe_trans_m": trans_err_sum / float(n),
        "rpe_rot_deg": rot_err_sum / float(n),
    }


def extract_video_motion(
    video_path: Path,
    temp_dir: Path,
    vggt_model,
    device,
    dtype,
    target_fps: float | None = None,
    analysis_subsample_step: int = 6,
    yaw_thresh_deg: float = 2.0,
    forward_thresh_m: float = 0.012,
    min_consecutive_frames: int = 5,
) -> Dict[str, Any]:
    """Extract motion primitives and per-frame labels from a video using VGGT."""
    raw_labels, trajectory = extract_video_labels_and_trajectory(
        video_path,
        temp_dir,
        vggt_model,
        device,
        dtype,
        target_fps=target_fps,
        analysis_subsample_step=analysis_subsample_step,
        yaw_thresh_deg=yaw_thresh_deg,
        forward_thresh_m=forward_thresh_m,
    )
    if not raw_labels and not trajectory:
        return {"primitives": [], "labels": [], "trajectory": []}

    clean_runs = get_clean_motion_sequence(raw_labels, min_len=min_consecutive_frames)
    motion_segments = [r for r in clean_runs if r[0] != 'S']

    if not motion_segments:
        return {"primitives": ["S"], "labels": raw_labels, "trajectory": trajectory}

    if len(motion_segments) > 2:
        indexed = [(i, label, length) for i, (label, length) in enumerate(motion_segments)]
        top2 = sorted(indexed, key=lambda x: x[2], reverse=True)[:2]
        top2_sorted = sorted(top2, key=lambda x: x[0])
        selected = [t[1] for t in top2_sorted]
    else:
        selected = [r[0] for r in motion_segments]

    mapped = []
    for lab in selected:
        if lab in ("TL", "TR", "F"):
            mapped.append(lab)
        else:
            if lab.startswith("TURN"):
                mapped.append("TL" if "LEFT" in lab else "TR")
            elif "FORWARD" in lab:
                mapped.append("F")
            else:
                mapped.append(lab)

    compact = []
    for m in mapped:
        if not compact or compact[-1] != m:
            compact.append(m)

    if not compact:
        compact = ["S"]

    return {"primitives": compact, "labels": raw_labels, "trajectory": trajectory}


def extract_video_primitives(
    video_path: Path,
    temp_dir: Path,
    vggt_model,
    device,
    dtype,
    target_fps: float | None = None,
    analysis_subsample_step: int = 6,
    yaw_thresh_deg: float = 2.0,
    forward_thresh_m: float = 0.012,
    min_consecutive_frames: int = 5,
) -> List[str]:
    """Extract motion primitives from a video file using VGGT.

    Behaviour:
    - Extract frames at `target_fps` (if provided) using `extract_frames_from_video`.
    - Optionally subsample frames for motion analysis by `analysis_subsample_step` (e.g., 6).
    - Run VGGT on the subsampled frames, denoise labels, and return up to two primitives.

    Returns run-length-encoded sequence of labels (drop consecutive repeats and remove 'S' unless it's the only label).
    This function uses utilities from `ImagiNav.vggt.imaginav_scripts.infer_move`.
    """
    motion = extract_video_motion(
        video_path,
        temp_dir,
        vggt_model,
        device,
        dtype,
        target_fps=target_fps,
        analysis_subsample_step=analysis_subsample_step,
        yaw_thresh_deg=yaw_thresh_deg,
        forward_thresh_m=forward_thresh_m,
        min_consecutive_frames=min_consecutive_frames,
    )
    return motion.get("primitives", [])


def motion_fidelity_score(prompt_actions: List[str], generated_actions: List[str]) -> float:
    """Compute normalized motion fidelity score in [0, 1].

    Score = 1 - (EditDistance / MaxLength)
    """
    prompt_norm = normalize_action_labels(prompt_actions)
    gen_norm = normalize_action_labels(generated_actions)

    max_len = max(len(prompt_norm), len(gen_norm))
    if max_len == 0:
        return 1.0

    dist = levenshtein_distance(prompt_norm, gen_norm)
    score = 1.0 - (dist / float(max_len))
    return max(0.0, min(1.0, score))
