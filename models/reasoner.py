"""
Gemini Reasoner Module

Converts (Image + Instruction) → Video Generation Prompt
"""

import os
import logging
import json
import re
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
from google import genai
from google.genai import types
from PIL import Image

from ImagiNav.core.config import ReasonerConfig


class GeminiReasoner:
    """
    Gemini-based reasoning module for generating video prompts.
    
    Takes a navigation instruction and current observation,
    reasons about what the robot should see next, and generates
    a detailed prompt for video generation.
    """
    
    def __init__(self, config: ReasonerConfig, logger: Optional[logging.Logger] = None, agent_name: str = "", output_dir: Optional[str] = None):
        """
        Initialize Gemini reasoner.
        
        Args:
            config: Reasoner configuration
            logger: Optional logger
            agent_name: Name of agent ("imaginav" or "navdp")
            output_dir: Directory to save logs/images
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = output_dir
        self.agent_name = agent_name
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.mock_response = getattr(config, "mock_response", None)
        if self.mock_response:
            self.client = None
            self.chat = None
            self.logger.info("Using mock Gemini reasoner response; Google API client disabled.")
            return
        
        # Get API key from environment
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set environment variable: {config.api_key_env}"
            )

        self.client = genai.Client(api_key=api_key)
        self._init_chat()

    def _init_chat(self) -> None:
        """(Re)initialize the Gemini chat session and send the system prompt."""
        self.chat = self.client.chats.create(model=self.config.model_name)

        if self.agent_name == "imaginav":
            self.chat.send_message(self.config.system_prompt)
        elif self.agent_name == "navdp":
            self.chat.send_message(self.config.system_prompt)
        else:
            raise ValueError(f"Unknown agent name: {self.agent_name}")

        self.logger.info(
            f"Initialized Gemini model: {self.config.model_name} and sent system prompt."
        )

    def reset(self) -> None:
        """Reset the Gemini chat session for a new episode."""
        def _extract_error_code(text: str) -> Optional[str]:
            for token in text.replace("'", "").replace("\n", " ").split():
                if token.isdigit():
                    return token
            return None

        while True:
            if self.mock_response:
                self.logger.info("Mock Gemini reasoner reset skipped.")
                return
            try:
                self._init_chat()
                self.logger.info("Gemini reasoner chat reset for new episode")
                return
            except Exception as e:
                error_text = str(e)
                error_code = _extract_error_code(error_text)
                if error_code == "503":
                    self.logger.warning(
                        "Gemini overloaded (503) during reset. Waiting 15s before retrying."
                    )
                    time.sleep(15)
                    continue
                self.logger.error(f"Failed to reset Gemini reasoner chat: {e}")
                raise
    
    def generate_prompt(
        self,
        image_input,
        instruction: str,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate video generation prompt.
        
        Args:
            image_input: Path to image (str) or PIL Image/ndarray
            instruction: Navigation instruction
            additional_context: Optional additional context
        
        Returns:
            Video generation prompt for LTX-Video
        """
        video_prompt, _, _, _ = self.generate_decision(
            image_input=image_input,
            instruction=instruction,
            additional_context=additional_context
        )
        return video_prompt

    def generate_decision(
        self,
        image_input,
        instruction: str,
        additional_context: Optional[str] = None
    ) -> Tuple[str, list, str, str]:
        """
        Generate both the video prompt and navigation command.

        Returns:
            (video_prompt, navigation_command, justification, expert_type)
        """
        # return "Dolly forward towards the hall.", [1, -0.4, 30], "Fallback prompt/command", "right"
        if self.mock_response:
            video_prompt = str(self.mock_response.get("video_prompt", "Dolly forward down the hallway."))
            nav_cmd = self.mock_response.get("navigation_command", [0.6, 0.0, 0.0])
            justification = str(self.mock_response.get("justification", "Mock smoke-test navigation command."))
            expert_type = str(self.mock_response.get("type_of_expert", "left")).strip().lower()
            if expert_type not in ("left", "right"):
                expert_type = "left"
            self.logger.info(
                "Mock Gemini decision: prompt=%r command=%s expert=%s",
                video_prompt,
                nav_cmd,
                expert_type,
            )
            return video_prompt, nav_cmd, justification, expert_type

        image = self._process_image(image_input)

        # Save prompt observation
        self._save_debug_image(image, subfolder="prompt_generation_images")

        user_prompt = self.config.prompt_template.format(instruction=instruction)
        if additional_context:
            user_prompt += f"\n\nAdditional context: {additional_context}"

        self.logger.info(f"Querying Gemini for prompt+command: {instruction}")

        try:
            response = self.chat.send_message([user_prompt, image])
            text_response = response.text or ""
            self.logger.info(f"Gemini Response: {text_response}")

            match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str = re.sub(r'^```(?:json)?\s*', '', json_str.strip(), flags=re.IGNORECASE)
                json_str = re.sub(r'\s*```$', '', json_str.strip())

                try:
                    data = json.loads(json_str)
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON from Gemini response: {e}")
                    data = None

                if isinstance(data, dict):
                    video_prompt = data.get("video_prompt", "")
                    nav_cmd = data.get("navigation_command", [0.0, 0.0, 0.0])
                    justification = data.get("justification", "")
                    expert_type = str(data.get("type_of_expert", "right")).strip().lower()
                    if expert_type not in ("left", "right"):
                        self.logger.warning(f"Unknown expert_type '{expert_type}'; defaulting to 'right'")
                        expert_type = "right"

                    # Normalize command to [x, y, theta]
                    if isinstance(nav_cmd, list):
                        if len(nav_cmd) == 1:
                            x = float(nav_cmd[0])
                            y = 0.0
                            theta = 0.0
                        elif len(nav_cmd) == 2:
                            # Backward-compat: treat as [x, theta]
                            x = float(nav_cmd[0])
                            y = 0.0
                            theta = float(nav_cmd[1])
                        else:
                            x = float(nav_cmd[0])
                            y = float(nav_cmd[1])
                            theta = float(nav_cmd[2])
                    else:
                        self.logger.error("navigation_command is not a list")
                        raise ValueError("navigation_command must be a list")

                    # Convert LLM convention (positive = right) to controller convention (positive = left)
                    y = -float(y)
                    theta = -float(theta)
                    nav_cmd = [x, y, theta]

                    # Normalize near-zero values to avoid -0.0 artifacts
                    if abs(nav_cmd[0]) < 1e-6:
                        nav_cmd[0] = 0.0
                    if abs(nav_cmd[1]) < 1e-6:
                        nav_cmd[1] = 0.0
                    if abs(nav_cmd[2]) < 1e-6:
                        nav_cmd[2] = 0.0

                    if not video_prompt:
                        # Gemini decided the robot should stop or that no video is required.
                        # Treat an empty video_prompt as an explicit STOP/NO-OP rather than an error.
                        self.logger.info("video_prompt is empty in Gemini response; interpreting as STOP/NO-OP.")
                        # Throttle to avoid quota bursts
                        try:
                            wait = float(getattr(self.config, 'throttle_seconds', 0))
                            if wait > 0:
                                self.logger.info(f"Waiting {wait}s after Gemini call to avoid quota spikes.")
                                time.sleep(wait)
                        except Exception:
                            pass
                        # Return empty prompt with parsed navigation command and justification.
                        return "", nav_cmd, justification, expert_type

                    # Throttle to avoid quota bursts
                    try:
                        wait = float(getattr(self.config, 'throttle_seconds', 0))
                        if wait > 0:
                            self.logger.info(f"Waiting {wait}s after Gemini call to avoid quota spikes.")
                            time.sleep(wait)
                    except Exception:
                        pass

                    return video_prompt, nav_cmd, justification, expert_type

            # Fallback parsing if JSON extraction fails
            self.logger.error("Could not parse JSON from response; falling back")
            raise ValueError("Failed to parse JSON from Gemini response")
            # return "Move forward", [0.5, 0.0], "Fallback prompt/command"

        except Exception as e:
            self.logger.error(f"Gemini Error: {e}")
            raise RuntimeError(f"Gemini API error: {e}")
        # return "Move forward torwards the bedroom.", [0.5, 0.0], "Fallback prompt/command"

    def score_video(
        self,
        video_prompt: str,
        video_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Score a generated video against the video prompt.

        Returns:
            Dict with keys: decision (ACCEPT/REJECT), justification
        """
        if not self.config.enable_scorer:
            return {"decision": "ACCEPT", "justification": "Scorer disabled"}

        video_path = Path(video_path)
        if not video_path.exists():
            self.logger.warning(f"Video not found for scoring: {video_path}")
            return {"decision": "ACCEPT", "justification": "Video missing; skipped scoring"}

        size_mb = video_path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.scorer_max_video_mb:
            self.logger.warning(
                f"Video size {size_mb:.2f}MB exceeds limit {self.config.scorer_max_video_mb}MB; skip scoring"
            )
            return {"decision": "ACCEPT", "justification": "Video too large; skipped scoring"}

        try:
            video_bytes = video_path.read_bytes()
            prompt_text = self.config.score_prompt_template.format(
                video_prompt=video_prompt
            )
            if self.config.scorer_system_prompt:
                prompt_text = f"{self.config.scorer_system_prompt}\n\n{prompt_text}"

            response = self.client.models.generate_content(
                model=self.config.scorer_model_name,
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                        ),
                        types.Part(text=prompt_text)
                    ]
                )
            )

            text_response = response.text or ""
            self.logger.info(f"Scorer response: {text_response}")

            match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str = re.sub(r'^```(?:json)?\s*', '', json_str.strip(), flags=re.IGNORECASE)
                json_str = re.sub(r'\s*```$', '', json_str.strip())

                try:
                    data = json.loads(json_str)
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON from scorer response: {e}")
                    data = None

                if isinstance(data, dict):
                    decision = str(data.get("decision", "ACCEPT")).upper()
                    justification = data.get("justification", "")
                    if decision not in ("ACCEPT", "REJECT"):
                        decision = "ACCEPT"
                    return {"decision": decision, "justification": justification}

            self.logger.warning("Scorer response not JSON; defaulting to ACCEPT")
            return {"decision": "ACCEPT", "justification": "Scorer response not JSON"}

        except Exception as e:
            self.logger.error(f"Scorer error: {e}", exc_info=True)
            # Treat scorer failures as fatal Gemini errors and abort evaluation immediately.
            raise RuntimeError(f"Gemini Scorer error: {e}") from e

    def rank_videos(
        self,
        video_prompt: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Rank multiple candidate videos against the video prompt.

        Sends all candidate videos to the Gemini scorer in a single API call and
        asks it to select the best one based on motion-prompt alignment, trajectory
        safety, and visual quality.

        Args:
            video_prompt: The prompt used to generate all candidate videos.
            candidates: List of dicts, each with at minimum:
                - 'path' (str | Path): path to the candidate video file.
                - 'description' (str): human-readable description of the configuration.
                - 'index' (int): original candidate index (used in return value).

        Returns:
            Dict with keys:
                best_index (int): ``candidates[?]['index']`` value of the winner.
                justification (str): scorer's explanation.
                candidate_scores (list): per-candidate score dicts from the model.
        """
        if not self.config.enable_scorer:
            return {
                "best_index": candidates[0].get("index", 0) if candidates else 0,
                "justification": "Scorer disabled; defaulting to first candidate",
                "candidate_scores": [],
            }

        if not candidates:
            return {"best_index": 0, "justification": "No candidates provided", "candidate_scores": []}

        if len(candidates) == 1:
            return {
                "best_index": candidates[0].get("index", 0),
                "justification": "Only one candidate available",
                "candidate_scores": [],
            }

        # --- Filter: drop missing / oversized videos ---
        valid_pairs: List[Dict[str, Any]] = []
        for cand in candidates:
            vpath = Path(cand.get("path", ""))
            if not vpath.exists():
                self.logger.warning(f"[Scorer] Candidate video not found: {vpath}")
                continue
            size_mb = vpath.stat().st_size / (1024 * 1024)
            if size_mb > self.config.scorer_max_video_mb:
                self.logger.warning(
                    f"[Scorer] Candidate video too large "
                    f"({size_mb:.2f} MB > {self.config.scorer_max_video_mb} MB): {vpath}"
                )
                continue
            valid_pairs.append(cand)

        if not valid_pairs:
            self.logger.warning("[Scorer] No valid candidate videos; defaulting to first candidate")
            return {
                "best_index": candidates[0].get("index", 0),
                "justification": "No valid videos for ranking",
                "candidate_scores": [],
            }

        if len(valid_pairs) == 1:
            return {
                "best_index": valid_pairs[0].get("index", 0),
                "justification": "Only one valid candidate after filtering",
                "candidate_scores": [],
            }

        # --- Build ranking prompt ---
        num_valid = len(valid_pairs)
        candidate_list_text = "\n".join(
            f"Video {i + 1}: {c.get('description', f'Candidate {i + 1}')}"
            for i, c in enumerate(valid_pairs)
        )

        # Build the per-call ranking prompt from config (with an inline fallback for
        # configs that predate the rank_prompt_template field).
        _fallback_rank_template = (
            "{num_candidates} candidate navigation videos are provided (in order).\n"
            'Video prompt: "{video_prompt}"\n\nCandidates:\n{candidate_list}\n\n'
            "Select the best candidate. Output a JSON object:\n"
            '{{"best_video_number": <integer 1 to {num_candidates}>, '
            '"justification": "<explanation>", '
            '"candidate_scores": [{{"video_number": 1, "score": <0-10>, "comment": "<...>"}}, ...]}}'
        )
        rank_prompt_template = getattr(self.config, "rank_prompt_template", _fallback_rank_template)
        rank_prompt = rank_prompt_template.format(
            num_candidates=num_valid,
            video_prompt=video_prompt,
            candidate_list=candidate_list_text,
        )
        # Prepend the scorer system prompt if set
        scorer_system = getattr(self.config, "scorer_system_prompt", "")
        if scorer_system:
            rank_prompt = f"{scorer_system}\n\n{rank_prompt}"

        # --- Build API parts: one video blob per valid candidate + text at the end ---
        parts = []
        for cand in valid_pairs:
            vpath = Path(cand["path"])
            parts.append(
                types.Part(inline_data=types.Blob(data=vpath.read_bytes(), mime_type="video/mp4"))
            )
        parts.append(types.Part(text=rank_prompt))

        try:
            response = self.client.models.generate_content(
                model=self.config.scorer_model_name,
                contents=types.Content(parts=parts),
            )

            text_response = response.text or ""
            self.logger.info(f"[Scorer] Ranking response: {text_response}")

            match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str = re.sub(r'^```(?:json)?\s*', '', json_str.strip(), flags=re.IGNORECASE)
                json_str = re.sub(r'\s*```$', '', json_str.strip())

                try:
                    data = json.loads(json_str)
                except Exception as parse_err:
                    self.logger.error(f"[Scorer] Failed to parse ranking JSON: {parse_err}")
                    data = None

                if isinstance(data, dict):
                    best_video_num = int(data.get("best_video_number", 1))
                    if 1 <= best_video_num <= num_valid:
                        best_cand = valid_pairs[best_video_num - 1]
                    else:
                        self.logger.warning(
                            f"[Scorer] best_video_number {best_video_num} out of range "
                            f"[1, {num_valid}]; defaulting to video 1"
                        )
                        best_cand = valid_pairs[0]

                    return {
                        "best_index": best_cand.get("index", 0),
                        "justification": data.get("justification", ""),
                        "candidate_scores": data.get("candidate_scores", []),
                    }

            self.logger.warning(
                "[Scorer] Could not parse ranking JSON; defaulting to first valid candidate"
            )
            return {
                "best_index": valid_pairs[0].get("index", 0),
                "justification": "Could not parse ranking response",
                "candidate_scores": [],
            }

        except Exception as e:
            self.logger.error(f"[Scorer] Ranking error: {e}", exc_info=True)
            raise RuntimeError(f"Gemini Scorer error: {e}") from e

    def generate_waypoint(
        self,
        image_input,
        instruction: str,
        additional_context: Optional[str] = None,
        step_count: int = 0
    ) -> list:
        """
        Generate navigation waypoint.
        
        Args:
            image_input: Path to image (str) or PIL Image/ndarray
            instruction: Navigation instruction
            additional_context: Optional additional context
            step_count: Current step count for logging
        
        Returns:
            Waypoint [x, y]
        """
        # Load image
        image = self._process_image(image_input)

        # Save query observation
        self._save_debug_image(image, subfolder="decision_making_images", step_count=step_count)
        
        # Format prompt
        user_prompt = self.config.prompt_template.format(instruction=instruction)
        
        if additional_context:
            user_prompt += f"\n\nAdditional context: {additional_context}"
        
        self.logger.info(f"Querying Gemini with instruction: {instruction}")
        
        # Generate response
        try:
            response = self.chat.send_message([user_prompt, image])
            text_response = response.text
            self.logger.info(f"Gemini Response: {text_response}")
            
            # Parse JSON
            match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if match:
                json_str = match.group(0)
                # Strip common Markdown code fences if present
                json_str = re.sub(r'^```(?:json)?\s*', '', json_str.strip(), flags=re.IGNORECASE)
                json_str = re.sub(r'\s*```$', '', json_str.strip())

                try:
                    data = json.loads(json_str)
                except Exception as e:
                    self.logger.error(f"Failed to parse JSON from Gemini response: {e}")
                    data = None

                if isinstance(data, dict):
                    waypoint = data.get("waypoint", [0.0, 0.0, 0.0])
                    justification = data.get("justification", "No justification provided.")

                    # Ensure triplet
                    if len(waypoint) == 2:
                        self.logger.info("Waypoint has only 2 elements, appending 0.0 for theta.")
                        waypoint.append(0.0)

                    self.logger.info(f"Parsed Waypoint: {waypoint}")
                    self.logger.info(f"Justification: {justification}")
                    return waypoint

                # Fallback: try to extract waypoint with regex if JSON parse failed
                wp_match = re.search(r'"waypoint"\s*:\s*\[\s*([-0-9\.]+)\s*,\s*([-0-9\.]+)\s*(?:,\s*([-0-9\.]+)\s*)?\]', text_response)
                if wp_match:
                    x = float(wp_match.group(1))
                    y = float(wp_match.group(2))
                    z = float(wp_match.group(3)) if wp_match.group(3) is not None else 0.0
                    waypoint = [x, y, z]
                    self.logger.info(f"Parsed Waypoint via regex fallback: {waypoint}")
                    return waypoint

                self.logger.error("Could not parse JSON or extract waypoint from response")
                return [0.5, 0.0, 0.0]  # Fallback safe forward
            else:
                self.logger.error("Could not find JSON in response")
                return [0.5, 0.0, 0.0] # Fallback safe forward

        except Exception as e:
            self.logger.error(f"Gemini Error: {e}")
            raise RuntimeError(f"Gemini API error: {e}")
        # return [0, 0.0, 0.0]  # Dummy waypoint for testing

    def batch_generate(
        self,
        image_paths: list,
        instructions: list
    ) -> list:
        """
        Generate prompts for multiple image-instruction pairs.
        
        Args:
            image_paths: List of image paths
            instructions: List of instructions
        
        Returns:
            List of video prompts
        """
        if len(image_paths) != len(instructions):
            raise ValueError("Number of images and instructions must match")
        
        prompts = []
        for img_path, instr in zip(image_paths, instructions):
            prompt = self.generate_prompt(img_path, instr)
            prompts.append(prompt)
        
        return prompts

    def _process_image(self, image_input) -> Image.Image:
        """Convert input to PIL Image handling various formats."""
        if isinstance(image_input, str):
            return Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # Handle float arrays [0, 1]
            if image_input.dtype in [np.float32, np.float64] and image_input.max() <= 1.05:
                arr = (image_input * 255).astype(np.uint8)
            elif image_input.dtype in [np.float32, np.float64]:
                arr = image_input.astype(np.uint8)
            else:
                arr = image_input
            return Image.fromarray(arr)
        elif isinstance(image_input, Image.Image):
            return image_input
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

    def _save_debug_image(self, image: Image.Image, subfolder: str, step_count: Optional[int] = None):
        """Save debug image to specific subfolder."""
        # Determine Log Dir
        log_dir = None
        # 1) Explicit output_dir set on the reasoner (preferred)
        if self.output_dir:
            log_dir = Path(self.output_dir)
        # 2) Check the config for a logs_dir attribute (note: config uses 'logs_dir')
        elif hasattr(self.config, 'logs_dir') and self.config.logs_dir:
            log_dir = Path(self.config.logs_dir)
        else:
            # 3) Look for a file handler on the provided logger and use its directory
            if hasattr(self.logger, 'handlers'):
                for h in self.logger.handlers:
                    base = getattr(h, 'baseFilename', None)
                    if base:
                        log_file = Path(base)
                        log_dir = log_file.parent
                        break

        # 4) Final fallback
        if log_dir is None:
            log_dir = Path("logs")

        save_dir = log_dir / subfolder
        
        if not save_dir.exists():
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Could not create log directory {save_dir}: {e}")
                return

        timestamp = int(time.time() * 1000)
        if step_count is not None:
             filename = f"step_{step_count:04d}_obs_{timestamp}.jpg"
        else:
             filename = f"obs_{timestamp}.jpg"
             
        save_path = save_dir / filename
        try:
             # Convert to RGB if needed
             if image.mode != 'RGB':
                 image = image.convert('RGB')
             image.save(save_path)
             self.logger.info(f"Saved observation to {save_path}")
        except Exception as e:
             self.logger.warning(f"Could not save observation image to {save_path}: {e}")
