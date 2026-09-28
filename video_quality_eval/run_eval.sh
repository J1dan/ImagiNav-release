#!/usr/bin/env bash
# Simple helper to run the evaluator
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <dataset.json> [dataset_root]"
  exit 1
fi
DATASET=$1
ROOT=${2:-$(dirname "$DATASET")}
OUTDIR=${3:-results}
python -m ImagiNav.video_quality_eval.evaluator --dataset "$DATASET" --dataset-root "$ROOT" --out "$OUTDIR"
