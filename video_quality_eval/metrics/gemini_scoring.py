"""Gemini scoring helper for video quality evaluation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


def gemini_score_video(
    video_prompt: str,
    video_path: Path,
    reasoner: Optional[object],
    reasoner_config: Optional[object],
    logger: Optional[object] = None,
) -> Dict[str, Any]:
    """Score a video with Gemini or return fallback heuristic scores.

    Returns a dict: {
      'instruction_following': float (0-1),
      'safety': float (0-1),
      'physical_coherence': float (0-1),
      'justification': str
    }
    """
    def _log(msg: str) -> None:
        if logger is not None:
            try:
                logger.info(msg)
            except Exception:
                pass

    def _warn(msg: str) -> None:
        if logger is not None:
            try:
                logger.warning(msg)
            except Exception:
                pass

    scoring_prompt = (
        f"Video Prompt: {video_prompt}\n\n"
        "Please evaluate and return a JSON object like: "
        "{\"instruction_following\": 0.0-1.0, \"safety\": 0.0-1.0, "
        "\"physical_coherence\": 0.0-1.0, \"justification\": \"...\"}"
    )

    api_key_env = getattr(reasoner_config, "api_key_env", None) if reasoner_config else None
    api_key = os.getenv(api_key_env) if api_key_env else None
    client = getattr(reasoner, "client", None) if reasoner is not None else None
    model_name = getattr(reasoner_config, "scorer_model_name", None) if reasoner_config else None

    if api_key and client and model_name:
        try:
            from google.genai import types
            video_bytes = Path(video_path).read_bytes()
            response = client.models.generate_content(
                model=model_name,
                contents=types.Content(
                    parts=[
                        types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")),
                        types.Part(text=scoring_prompt),
                    ]
                ),
            )
            text_response = response.text or ""
            m = re.search(r"\{.*\}", text_response, re.DOTALL)
            if m:
                json_str = m.group(0)
                json_str = re.sub(r"^```(?:json)?\s*", "", json_str.strip(), flags=re.IGNORECASE)
                json_str = re.sub(r"\s*```$", "", json_str.strip())
                data = json.loads(json_str)
                return {
                    "instruction_following": float(data.get("instruction_following", 0.0)),
                    "safety": float(data.get("safety", 0.0)),
                    "physical_coherence": float(data.get("physical_coherence", 0.0)),
                    "justification": data.get("justification", ""),
                }
            _warn("Gemini scorer response not usable; falling back to heuristics")
        except Exception as e:
            _warn(f"Gemini scoring failed: {e}; falling back to heuristics")
    else:
        _log("Gemini client not configured; using fallback heuristic scores")

    return {
        "instruction_following": 0.5,
        "safety": 0.9,
        "physical_coherence": 0.6,
        "justification": "Fallback heuristic scores used (no Gemini API)",
    }
