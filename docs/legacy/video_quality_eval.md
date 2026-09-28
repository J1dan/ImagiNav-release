# Legacy video-quality evaluation notes

The current evaluation procedure is in [ImagiNav's README](../../README.md#offline-video-quality).
This archived guide may describe older commands and is retained for historical
reference only.

This package evaluates generated navigation videos with motion fidelity, RPE,
LPIPS, PSNR, SSIM, and content-debiased FVD. See the [main guide](../../README.md)
for repository setup, checkpoints, training, and simulator integration.

## Prerequisites

Complete the main installation first. The examples below assume:

```bash
export IMAGINAV_ROOT=/absolute/path/to/ImagiNav
export PYTHONPATH="$(dirname "$IMAGINAV_ROOT")${PYTHONPATH:+:$PYTHONPATH}"
export IMAGINAV_PYTHON="$IMAGINAV_ROOT/LTX-Video-Trainer/.venv/bin/python"
export IMAGINAV_HF="$IMAGINAV_ROOT/LTX-Video-Trainer/.venv/bin/hf"
```

Generation and VGGT motion extraction require a CUDA GPU for practical runs.
The default configuration disables Gemini scoring, so a Google API key is not
required unless `enable_metrics.gemini_scoring` is enabled.

## Download evaluation data

Download only the evaluation subset:

```bash
mkdir -p "$IMAGINAV_ROOT/video_quality_eval/datasets"

"$IMAGINAV_HF" download J1dan/imaginav-dataset \
  --repo-type dataset \
  --include "eval_clips/**" \
  --local-dir "$IMAGINAV_ROOT/video_quality_eval/datasets"

ln -sfn eval_clips \
  "$IMAGINAV_ROOT/video_quality_eval/datasets/reference_videos"
```

The required source manifest is expected at:

```text
video_quality_eval/datasets/reference_videos/dataset-20260131.json
```

Paths in `video_quality_eval/configs/*.yaml` are resolved relative to the
`video_quality_eval` package directory.

## Run the pipeline

Run all commands from the directory containing the `ImagiNav` clone:

```bash
cd "$IMAGINAV_ROOT/.."
CONFIG=ImagiNav/video_quality_eval/configs/video_quality_eval.yaml
```

### 1. Extract first frames

```bash
"$IMAGINAV_PYTHON" -m ImagiNav.video_quality_eval.scripts.extract_first_frames \
  --config "$CONFIG"
```

This creates `first_frame_images/` and
`dataset-20260131_with_firstframes.json` beside the source manifest.

### 2. Generate videos

Download the AC-MoE checkpoints described in the main README, then run:

```bash
"$IMAGINAV_PYTHON" -m ImagiNav.video_quality_eval.eval_video_gen \
  --config "$CONFIG"
```

With the default config, generated videos are written under
`synthetic_videos_full/` and the enriched manifest is named
`dataset-20260131_with_firstframes_full.json`.

The generation config supports:

- `lora_mode: dual` for left/forward and right adapters
- `lora_mode: single` for one shared adapter (`uni` is accepted as an alias)
- `lora_mode: none` for the base LTX model

### 3. Compute metrics

```bash
"$IMAGINAV_PYTHON" -m ImagiNav.video_quality_eval.evaluator \
  --config "$CONFIG" \
  --dataset ImagiNav/video_quality_eval/datasets/reference_videos/dataset-20260131_with_firstframes_full.json \
  --out ImagiNav/video_quality_eval/results/full
```

The existing `run_eval.sh` helper uses its original dataset-first interface. To
use it, activate the LTX environment and pass the optional dataset root and
output directory:

```bash
source "$IMAGINAV_ROOT/LTX-Video-Trainer/.venv/bin/activate"
ImagiNav/video_quality_eval/run_eval.sh \
  ImagiNav/video_quality_eval/datasets/reference_videos/dataset-20260131_with_firstframes_full.json \
  ImagiNav/video_quality_eval/datasets/reference_videos \
  ImagiNav/video_quality_eval/results/full
```

## Configuration

The default full-set configuration is `configs/video_quality_eval.yaml`. The
half-set configuration limits generation to 355 examples and writes to a
separate output directory.

Important sections:

- `dataset`: source manifest, dataset root, first-frame output, and VGGT cache
- `generation`: device, output directory, sampling, and adapter paths
- `enable_metrics`: per-metric switches
- `motion_fidelity`: action parsing and trajectory thresholds
- `fvd`: feature extractor, precision, and optional checkpoint path

If `fvd.ckpt_path` is null, the VideoMAE checkpoint is downloaded through
`huggingface_hub` and cached normally. Set it to a local checkpoint for offline
runs.

## Outputs

`results.json` contains per-sample fields such as:

- `video_path`
- `prompt_primitives` and `generated_primitives`
- `motion_fidelity`
- `rpe_translation` and `rpe_rotation_deg` when trajectories are available
- `lpips`, `psnr`, and `ssim`
- `gemini_scores` when enabled

`aggregate_metrics.json` contains dataset-level summaries, including FVD when
enough valid video pairs are available.

The evaluator accepts `instruction` or `caption` as the text field and checks
generated paths in this order: `gen_video_path`, `video_path`, then `video`.

Example entry:

```json
{
  "caption": "Dolly forward and pan left.",
  "reference_video": "LaunchPad_clips/example.mp4",
  "first_frame_path": "first_frame_images/LaunchPad_clips/example_firstframe.png",
  "gen_video_path": "synthetic_videos_full/LaunchPad_clips/gen-example.mp4"
}
```

Motion fidelity is one minus normalized Levenshtein distance between the
expected and inferred action sequences:

```text
score = 1 - edit_distance(expected, generated) / max(len(expected), len(generated))
```
