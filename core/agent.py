"""
ImagiNav Agent - Main Orchestration

The core agent that orchestrates the three-stage pipeline:
1. Reasoning (Gemini)
2. Imagination (LTX-Video)
3. Navigation (VGGT + Controller)

Completely standalone - no InternNav dependencies.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import time

from ImagiNav.core.config import ImagiNavConfig
from ImagiNav.core.types import TrajectoryOutput, PipelineState, NavigationCommand, CameraPose
from ImagiNav.models.reasoner import GeminiReasoner
from ImagiNav.models.imagination import LTXVideoGenerator
from ImagiNav.models.navigator import VGGTNavigator
from ImagiNav.utils.logger import setup_logger
from ImagiNav.utils.output_manager import OutputManager


def _build_video_candidates(
    video_prompt: str,
    expert_type: Optional[str],
    base_guidance_scale: float,
    lora_mode: str,
) -> list:
    """Build the three candidate video-generation configurations for sampling-based planning.

    Strategy
    --------
    Normalise the prompt first so that 'turn left/right' → 'pan left/right'.

    **Pure-forward prompt** (no pan/turn left/right keywords):
      - Candidate 0: left/forward expert, gs = base
      - Candidate 1: right expert,        gs = base
      - Candidate 2: reasoner-chosen expert, gs = base − 0.5

    **Turning prompt** (has pan/turn left/right):
      - Candidate 0: reasoner-chosen expert, gs = base + 0.5
      - Candidate 1: reasoner-chosen expert, gs = base
      - Candidate 2: reasoner-chosen expert, gs = base − 0.5

    In single-LoRA mode the adapter is always 'default' and only the
    guidance-scale axis is varied.

    Returns
    -------
    List of dicts with keys: adapter, guidance_scale, description.
    """
    # Import here to avoid a circular dependency at module level
    from ImagiNav.models.imagination import LTXVideoGenerator as _LTXGen

    normalised = _LTXGen.normalize_motion_primitives(video_prompt)
    is_forward = _LTXGen._is_pure_forward(normalised)

    gs0 = base_guidance_scale - 0.5
    gs_hi = base_guidance_scale + 0.5

    if lora_mode == "dual":
        # Map reasoner's expert_type to adapter name
        gemini_adapter = "right" if str(expert_type or "").strip().lower() == "right" else "left_forward"

        if is_forward:
            return [
                {
                    "adapter": "left_forward",
                    "guidance_scale": base_guidance_scale,
                    "description": f"Forward — left/forward expert (gs={base_guidance_scale:.1f})",
                },
                {
                    "adapter": "right",
                    "guidance_scale": base_guidance_scale,
                    "description": f"Forward — right expert (gs={base_guidance_scale:.1f})",
                },
                {
                    "adapter": gemini_adapter,
                    "guidance_scale": gs0,
                    "description": (
                        f"Forward — reasoner expert '{gemini_adapter}' (gs={gs0:.1f})"
                    ),
                },
            ]
        else:
            return [
                {
                    "adapter": gemini_adapter,
                    "guidance_scale": gs_hi,
                    "description": f"Turn — expert '{gemini_adapter}' (gs={gs_hi:.1f})",
                },
                {
                    "adapter": gemini_adapter,
                    "guidance_scale": base_guidance_scale,
                    "description": f"Turn — expert '{gemini_adapter}' (gs={base_guidance_scale:.1f})",
                },
                {
                    "adapter": gemini_adapter,
                    "guidance_scale": gs0,
                    "description": f"Turn — expert '{gemini_adapter}' (gs={gs0:.1f})",
                },
            ]
    else:
        # Single-LoRA mode: vary guidance scale only
        return [
            {
                "adapter": "default",
                "guidance_scale": gs_hi,
                "description": f"Default adapter (gs={gs_hi:.1f})",
            },
            {
                "adapter": "default",
                "guidance_scale": base_guidance_scale,
                "description": f"Default adapter (gs={base_guidance_scale:.1f})",
            },
            {
                "adapter": "default",
                "guidance_scale": gs0,
                "description": f"Default adapter (gs={gs0:.1f})",
            },
        ]


class ImagiNavAgent:
    """
    ImagiNav: Imagination-based Navigation Agent
    
    Pipeline:
        Image + Instruction → [Reasoner] → Video Prompt
        Image + Prompt → [Imagination] → Video
        Video → [Navigator] → Velocity Commands
    
    Usage:
        ```python
        from imaginav import ImagiNavAgent, ImagiNavConfig
        
        # Initialize
        config = ImagiNavConfig.from_yaml("config.yaml")
        agent = ImagiNavAgent(config)
        
        # Navigate
        trajectory = agent.navigate(
            image_path="observation.jpg",
            instruction="Go to the kitchen"
        )
        
        # Use outputs
        actions = trajectory.to_internnav_actions()
        ros_path = trajectory.to_ros_path()
        trajectory.to_json("output.json")
        ```
    """
    
    def __init__(self, config: ImagiNavConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize ImagiNav agent.
        
        Args:
            config: Configuration object
            logger: Optional custom logger
        """
        self.config = config
        self.logger = logger or setup_logger(
            name="ImagiNav",
            level=config.logging.log_level,
            log_dir=config.logging.logs_dir,
            log_to_file=False
        )
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Invalid configuration. Check logs for details.")
        
        # Initialize output manager
        self.output_manager = OutputManager(
            logs_dir=config.logging.logs_dir,
            use_timestamp_dirs=config.logging.use_timestamp_dirs,
            keep_n_recent=config.logging.keep_n_recent
        )
        
        # Initialize pipeline components (lazy loading)
        self._reasoner: Optional[GeminiReasoner] = None
        self._imagination: Optional[LTXVideoGenerator] = None
        self._navigator: Optional[VGGTNavigator] = None
        
        # Pipeline state
        self.state = PipelineState()
        
        self.logger.info("ImagiNav agent initialized")
        self.logger.info(f"Logs directory: {config.logging.logs_dir}")
    
    @property
    def reasoner(self) -> GeminiReasoner:
        """Lazy-load Gemini reasoner."""
        if self._reasoner is None:
            self.logger.info("Loading Gemini reasoner...")
            self._reasoner = GeminiReasoner(
                config=self.config.reasoner,
                logger=self.logger,
                agent_name="imaginav"
            )
        return self._reasoner
    
    @property
    def imagination(self) -> LTXVideoGenerator:
        """Lazy-load LTX-Video generator."""
        if self._imagination is None:
            self.logger.info("Loading LTX-Video generator...")
            self._imagination = LTXVideoGenerator(
                config=self.config.imagination,
                logger=self.logger
            )
        return self._imagination
    
    @property
    def navigator(self) -> VGGTNavigator:
        """Lazy-load VGGT navigator."""
        if self._navigator is None:
            self.logger.info("Loading VGGT navigator...")
            self._navigator = VGGTNavigator(
                config=self.config.navigation,
                logger=self.logger
            )
        return self._navigator
    
    def navigate(
        self,
        image_path: str,
        instruction: str,
        output_dir: Optional[Path] = None,
        episode_id: Optional[str] = None,
        step_id: Optional[int] = None
    ) -> TrajectoryOutput:
        """
        Execute full ImagiNav pipeline.
        
        Args:
            image_path: Path to input RGB image (robot's current view)
            instruction: Natural language navigation instruction
            output_dir: Optional custom output directory (overrides episode_id/step_id)
            episode_id: Optional episode identifier for structured logging
            step_id: Optional step identifier (e.g., the Nth video generation)
        
        Returns:
            TrajectoryOutput containing all results
        
        Raises:
            RuntimeError: If any pipeline stage fails
        """
        start_time = time.time()
        
        # Setup output directory
        if output_dir is None:
            if episode_id is not None and step_id is not None:
                # Structured logging: logs_dir/episode_id/step_id
                output_dir = Path(self.config.logging.logs_dir) / str(episode_id) / f"step_{step_id}"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Default timestamp logging
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = self.output_manager.create_session_dir(session_id)
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("="*70)
        self.logger.info(f"Starting ImagiNav pipeline")
        self.logger.info(f"Instruction: {instruction}")
        self.logger.info(f"Image: {image_path}")
        self.logger.info(f"Output: {output_dir}")
        self.logger.info("="*70)

        # Ensure per-episode debug files and logs are stored under the episode output directory
        try:
            # Tell the reasoner to save debug images under this output directory
            if self._reasoner is not None:
                self._reasoner.output_dir = str(output_dir)
            # Add a per-episode file handler to capture logs for this episode
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_tag = str(episode_id) if episode_id is not None else "unknown_episode"
            step_tag = f"step_{step_id}" if step_id is not None else "unknown_step"
            per_episode_log = Path(output_dir) / f"imaginav_{episode_tag}_{step_tag}_{ts}.log"
            fh = logging.FileHandler(per_episode_log)
            fh.setLevel(getattr(logging, self.config.logging.log_level.upper()))
            # Try to reuse existing formatter if present
            fmt = None
            for h in self.logger.handlers:
                if getattr(h, 'formatter', None):
                    fmt = h.formatter
                    break
            if fmt is None:
                fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self._episode_log_handler = fh
            self.logger.info(f"Attached per-episode log: {per_episode_log}")
        except Exception:
            self.logger.debug("Could not attach per-episode log handler", exc_info=True)

        # Reset per-episode counters when episode_id changes
        if episode_id is not None:
            last_episode_id = self.state.artifacts.get("episode_id")
            if last_episode_id != episode_id:
                self.state.artifacts["reasoner_prompts_sent"] = 0
                self.state.artifacts["episode_id"] = episode_id
                if self._reasoner is not None:
                    try:
                        self._reasoner.reset()
                        self.logger.info("Reset reasoner chat for new episode")
                    except Exception as e:
                        # Log as error and abort by raising a RuntimeError to stop evaluation
                        self.logger.error("Failed to reset reasoner chat for new episode", exc_info=True)
                        raise RuntimeError("Failed to reset reasoner chat for new episode") from e

        try:
            # Stage 1: Reasoning
            self.state.stage = "reasoning"
            self.state.current_step = "Generating video prompt with Gemini"
            self.logger.info(f"\n[1/3] {self.state.current_step}...")

            # Enforce per-episode max prompt attempts (configurable) with retry
            max_prompts = getattr(self.config.reasoner, 'max_prompts_per_episode', None)
            current_tries = int(self.state.artifacts.get('reasoner_prompts_sent', 0))

            def _call_reasoner_with_retry():
                nonlocal current_tries
                while True:
                    if max_prompts is not None and current_tries >= int(max_prompts):
                        msg = (
                            f"Max reasoner prompts reached for this episode "
                            f"({current_tries} >= {max_prompts}). Signaling STOP for this episode."
                        )
                        # Treat as a non-fatal STOP action: return an empty prompt and a zero navigation command
                        # so the outer logic handles stopping and continues to the next episode.
                        self.logger.warning(msg)
                        return "", [0.0, 0.0, 0.0], msg, None

                    current_tries += 1
                    self.state.artifacts['reasoner_prompts_sent'] = current_tries

                    try:
                        return self.reasoner.generate_decision(
                            image_input=image_path,
                            instruction=instruction
                        )
                    except Exception as e:
                        error_text = str(e)
                        def _extract_error_code(text: str) -> Optional[str]:
                            for token in text.replace("'", "").replace("\n", " ").split():
                                if token.isdigit():
                                    return token
                            return None

                        error_code = _extract_error_code(error_text)
                        if error_code == "503":
                            self.logger.warning(
                                "Gemini overloaded (503). Waiting 15s before retrying."
                            )
                            time.sleep(15)
                            continue
                        # Any Gemini/reasoner exception should abort evaluation immediately.
                        self.logger.error(
                            "Gemini reasoning error on attempt %s/%s: %s",
                            current_tries,
                            max_prompts if max_prompts is not None else "∞",
                            e,
                            exc_info=True,
                        )
                        self.logger.error("Aborting evaluation due to Gemini/reasoner failure.")
                        raise RuntimeError(f"Gemini API error: {e}") from e

            video_prompt, navigation_command, command_justification, expert_type = _call_reasoner_with_retry()
            
            self.logger.info(f"Generated prompt: {video_prompt[:100]}...")
            self.logger.info(f"Reasoner command: {navigation_command}")

            # Defer saving the prompt until after we check whether we should stop or rotate in place
            try:
                if isinstance(navigation_command, (list, tuple)):
                    if len(navigation_command) == 1:
                        nav_cmd = [float(navigation_command[0]), 0.0, 0.0]
                    elif len(navigation_command) == 2:
                        nav_cmd = [float(navigation_command[0]), 0.0, float(navigation_command[1])]
                    else:
                        nav_cmd = [
                            float(navigation_command[0]),
                            float(navigation_command[1]),
                            float(navigation_command[2])
                        ]
                else:
                    nav_cmd = navigation_command
            except Exception:
                nav_cmd = navigation_command

            # Full stop condition: [0, 0, 0]
            if isinstance(nav_cmd, (list, tuple)) and len(nav_cmd) >= 3 and abs(nav_cmd[0]) < 1e-6 and abs(nav_cmd[1]) < 1e-6 and abs(nav_cmd[2]) < 1e-6:
                # If the reasoner returned a non-empty prompt while also outputting [0,0], log the mismatch and ignore the prompt
                if video_prompt:
                    self.logger.warning("Reasoner returned a non-empty video prompt but signaled STOP via navigation_command [0,0]; ignoring the prompt.")
                self.logger.info("Reasoner signaled STOP/NO-OP (navigation_command near zero). Skipping imagination and navigation.")
                self.state.stage = "complete"
                trajectory_output = TrajectoryOutput(
                    reasoning_prompt="",
                    video_path=Path(""),
                    camera_trajectory=[],
                    navigation_commands=[[0.0, 0.0, 0.0]],
                    instruction=instruction,
                    input_image_path=Path(image_path),
                    total_duration=0.0,
                    success=True,
                    intermediate_outputs={
                        "output_dir": str(output_dir),
                        "execution_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat(),
                        "reasoner_command": nav_cmd,
                        "reasoner_justification": command_justification,
                        "reasoner_expert_type": expert_type,
                        "stopped": True
                    }
                )
                if self.config.logging.save_trajectories:
                    trajectory_file = output_dir / "trajectory.json"
                    trajectory_output.to_json(trajectory_file)
                    self.logger.info(f"Saved trajectory to: {trajectory_file}")
                self.logger.info("\n" + trajectory_output.get_summary())
                elapsed = time.time() - start_time
                self.logger.info(f"\n✅ Pipeline complete in {elapsed:.2f}s (Stopped by reasoner)")
                return trajectory_output

            # In-place rotation condition: x=y=0 but non-zero theta
            if isinstance(nav_cmd, (list, tuple)) and len(nav_cmd) >= 3 and abs(nav_cmd[0]) < 1e-6 and abs(nav_cmd[1]) < 1e-6 and abs(nav_cmd[2]) >= 1e-3:
                self.logger.info("Reasoner requested in-place rotation. Performing direct navigation without generating a video.")
                self.state.current_step = "Direct navigation (rotate-in-place) from reasoner command"

                camera_trajectory = []
                navigation_commands = self.navigator.direct_navigate(nav_cmd)
                self.logger.info(f"Generated {len(navigation_commands)} direct commands for rotation")

                # Compute total duration
                if self.config.navigation.controller.inverse_dynamics_fps:
                    dt = 1.0 / float(self.config.navigation.controller.inverse_dynamics_fps)
                else:
                    dt = self.config.navigation.controller.pure_pursuit.dt
                total_duration = len(navigation_commands) * dt

                self.state.stage = "complete"
                trajectory_output = TrajectoryOutput(
                    reasoning_prompt=video_prompt,
                    video_path=Path(""),
                    camera_trajectory=camera_trajectory,
                    navigation_commands=navigation_commands,
                    instruction=instruction,
                    input_image_path=Path(image_path),
                    total_duration=total_duration,
                    success=True,
                    intermediate_outputs={
                        "output_dir": str(output_dir),
                        "execution_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat(),
                        "reasoner_command": nav_cmd,
                        "reasoner_justification": command_justification,
                        "reasoner_expert_type": expert_type,
                        "rotation_only": True
                    }
                )

                if self.config.logging.save_trajectories:
                    trajectory_file = output_dir / "trajectory.json"
                    trajectory_output.to_json(trajectory_file)
                    self.logger.info(f"Saved trajectory to: {trajectory_file}")

                self.logger.info("\n" + trajectory_output.get_summary())
                elapsed = time.time() - start_time
                self.logger.info(f"\n✅ Pipeline complete in {elapsed:.2f}s (Rotation executed)")
                return trajectory_output

            # Stage 2: Imagination
            self.state.stage = "imagination"

            # Save prompt only when we're actually going to use it to generate a video
            if self.config.logging.save_intermediate:
                prompt_file = output_dir / "01_reasoning_prompt.txt"
                prompt_file.write_text(video_prompt)
            self.state.current_step = "Generating video with LTX-Video"
            self.logger.info(f"\n[2/3] {self.state.current_step}...")

            scorer_result = None
            video_path = None

            if self.config.reasoner.enable_scorer:
                # ── Sampling-based planning ──────────────────────────────────────────
                # Build 3 candidate configurations and generate each video, then rank.
                base_gs = self.config.imagination.guidance_scale
                candidates = _build_video_candidates(
                    video_prompt=video_prompt,
                    expert_type=expert_type,
                    base_guidance_scale=base_gs,
                    lora_mode=self.config.imagination.lora_mode,
                )

                self.logger.info(
                    f"[Scorer] Sampling-based planning: generating {len(candidates)} candidate videos"
                )
                for i, cand in enumerate(candidates):
                    self.logger.info(
                        f"  Candidate {i}: adapter={cand['adapter']}, "
                        f"guidance_scale={cand['guidance_scale']:.2f} — {cand['description']}"
                    )

                # Generate each candidate
                generated_candidates = []
                for i, cand in enumerate(candidates):
                    try:
                        self.logger.info(
                            f"  [Scorer] Generating candidate {i + 1}/{len(candidates)}: "
                            f"{cand['description']}"
                        )
                        cand_video = self.imagination.generate_video_with_params(
                            image_path=image_path,
                            prompt=video_prompt,
                            output_dir=output_dir,
                            negative_prompt="blurry, low quality, static, stationary",
                            seed=42,
                            adapter_name=cand["adapter"],
                            guidance_scale=cand["guidance_scale"],
                            video_name=f"candidate_{i:02d}",
                        )
                        generated_candidates.append(
                            {**cand, "path": cand_video, "index": i}
                        )
                        self.logger.info(f"    Saved: {cand_video}")
                    except Exception as cand_err:
                        self.logger.error(
                            f"  [Scorer] Candidate {i} generation failed: {cand_err}",
                            exc_info=True,
                        )
                        generated_candidates.append(
                            {**cand, "path": None, "index": i, "error": str(cand_err)}
                        )

                # Rank candidates
                valid_for_ranking = [
                    c for c in generated_candidates
                    if c.get("path") and Path(c["path"]).exists()
                ]
                if not valid_for_ranking:
                    raise RuntimeError("All candidate video generations failed")

                if len(valid_for_ranking) == 1:
                    best_candidate = valid_for_ranking[0]
                    self.logger.info(
                        f"[Scorer] Only one valid candidate; skipping ranking. "
                        f"Using: {best_candidate['description']}"
                    )
                    scorer_result = {
                        "best_index": best_candidate["index"],
                        "justification": "Only one valid candidate generated",
                        "candidate_scores": [],
                    }
                else:
                    self.logger.info(
                        f"[Scorer] Ranking {len(valid_for_ranking)} "
                        f"out of {len(generated_candidates)} candidates..."
                    )
                    
                    # Retry wrapper for rank_videos with 503 error handling
                    def _call_rank_videos_with_retry():
                        while True:
                            try:
                                return self.reasoner.rank_videos(
                                    video_prompt=video_prompt,
                                    candidates=valid_for_ranking,
                                )
                            except Exception as e:
                                error_text = str(e)
                                def _extract_error_code(text: str) -> Optional[str]:
                                    for token in text.replace("'", "").replace("\n", " ").split():
                                        if token.isdigit():
                                            return token
                                    return None

                                error_code = _extract_error_code(error_text)
                                if error_code == "503":
                                    self.logger.warning(
                                        "Gemini overloaded (503) during ranking. Waiting 15s before retrying."
                                    )
                                    time.sleep(15)
                                    continue
                                # Any other Gemini/reasoner exception should be re-raised
                                raise
                    
                    rank_result = _call_rank_videos_with_retry()
                    best_idx = rank_result.get(
                        "best_index", valid_for_ranking[0]["index"]
                    )
                    best_candidate = next(
                        (c for c in generated_candidates if c["index"] == best_idx),
                        valid_for_ranking[0],
                    )
                    scorer_result = {
                        **rank_result,
                        "candidates": [
                            {
                                "index": c.get("index"),
                                "adapter": c.get("adapter"),
                                "guidance_scale": c.get("guidance_scale"),
                                "description": c.get("description"),
                                "path": str(c.get("path", "")),
                            }
                            for c in generated_candidates
                        ],
                        "selected_candidate": {
                            "index": best_candidate.get("index"),
                            "adapter": best_candidate.get("adapter"),
                            "guidance_scale": best_candidate.get("guidance_scale"),
                            "description": best_candidate.get("description"),
                        },
                    }
                    self.logger.info(
                        f"[Scorer] Best candidate: index={best_idx}, "
                        f"adapter={best_candidate.get('adapter')}, "
                        f"guidance_scale={best_candidate.get('guidance_scale'):.2f}"
                    )
                    self.logger.info(
                        f"[Scorer] Justification: {rank_result.get('justification', '')}"
                    )
                    if rank_result.get("candidate_scores"):
                        for cs in rank_result["candidate_scores"]:
                            self.logger.info(
                                f"  [Scorer] Video {cs.get('video_number')}: "
                                f"score={cs.get('score')} — {cs.get('comment', '')}"
                            )

                # Copy best video to the canonical name for downstream use
                best_video_path = best_candidate["path"]
                canonical_path = (
                    output_dir / f"02_generated_video.{self.config.imagination.output_format}"
                )
                if Path(best_video_path) != canonical_path:
                    shutil.copy2(str(best_video_path), str(canonical_path))
                    self.logger.info(
                        f"[Scorer] Copied best candidate to canonical path: {canonical_path}"
                    )
                video_path = canonical_path

            else:
                # ── Single video generation (original behaviour) ─────────────────────
                video_path = self.imagination.generate_video(
                    image_path=image_path,
                    prompt=video_prompt,
                    output_dir=output_dir,
                    negative_prompt="blurry, low quality, static, stationary",
                    seed=42,
                    expert_type=expert_type,
                )

            self.logger.info(f"Generated video: {video_path}")

            # Stage 3: Navigation
            self.state.stage = "navigation"
            self.state.current_step = "Extracting trajectory with VGGT + Controller"
            self.logger.info(f"\n[3/3] {self.state.current_step}...")

            camera_trajectory, navigation_commands = self.navigator.extract_trajectory(
                video_path=video_path,
                output_dir=output_dir,
            )

            self.logger.info(f"Extracted {len(camera_trajectory)} camera poses")
            self.logger.info(f"Generated {len(navigation_commands)} navigation commands")
            
            # Compute total duration
            if self.config.navigation.controller.inverse_dynamics_fps:
                dt = 1.0 / float(self.config.navigation.controller.inverse_dynamics_fps)
            else:
                dt = self.config.navigation.controller.pure_pursuit.dt
            total_duration = len(navigation_commands) * dt
            
            # Create output object
            self.state.stage = "complete"
            trajectory_output = TrajectoryOutput(
                reasoning_prompt=video_prompt,
                video_path=video_path,
                camera_trajectory=camera_trajectory,
                navigation_commands=navigation_commands,
                instruction=instruction,
                input_image_path=Path(image_path),
                total_duration=total_duration,
                success=True,
                intermediate_outputs={
                    "output_dir": str(output_dir),
                    "execution_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            )

            trajectory_output.intermediate_outputs["reasoner_command"] = navigation_command
            trajectory_output.intermediate_outputs["reasoner_justification"] = command_justification
            trajectory_output.intermediate_outputs["reasoner_expert_type"] = expert_type
            if scorer_result is not None:
                trajectory_output.intermediate_outputs["scorer"] = scorer_result
            
            # Save trajectory
            if self.config.logging.save_trajectories:
                trajectory_file = output_dir / "trajectory.json"
                trajectory_output.to_json(trajectory_file)
                self.logger.info(f"Saved trajectory to: {trajectory_file}")
            
            # Print summary
            self.logger.info("\n" + trajectory_output.get_summary())
            
            elapsed = time.time() - start_time
            self.logger.info(f"\n✅ Pipeline complete in {elapsed:.2f}s")
            
            return trajectory_output
        
        except Exception as e:
            self.logger.error(f"Pipeline failed at stage '{self.state.stage}': {e}", exc_info=True)
            error_text = str(e)
            if (
                "Gemini" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):
                raise
            
            # Return failed trajectory
            trajectory_output = TrajectoryOutput(
                reasoning_prompt="",
                video_path=Path(""),
                camera_trajectory=[],
                navigation_commands=[],
                instruction=instruction,
                input_image_path=Path(image_path),
                total_duration=0.0,
                success=False,
                error_message=str(e),
                intermediate_outputs={
                    "failed_stage": self.state.stage,
                    "execution_time": time.time() - start_time,
                }
            )
            
            return trajectory_output
        finally:
            # Clean up per-episode log handler if attached and reset reasoner output_dir
            try:
                if hasattr(self, '_episode_log_handler') and self._episode_log_handler in self.logger.handlers:
                    self.logger.removeHandler(self._episode_log_handler)
                    try:
                        self._episode_log_handler.close()
                    except Exception:
                        pass
                    delattr(self, '_episode_log_handler')
            except Exception:
                pass
            try:
                if hasattr(self, '_reasoner') and self._reasoner is not None:
                    try:
                        self._reasoner.output_dir = None
                    except Exception:
                        pass
            except Exception:
                pass
    
    def reset(self):
        """Reset agent state and clear cache."""
        self.state = PipelineState()
        self.logger.info("Agent state reset")
    
    def cleanup(self):
        """Cleanup resources and unload models."""
        if self._imagination is not None:
            self._imagination.cleanup()
        if self._navigator is not None:
            self._navigator.cleanup()
        
        self.logger.info("Agent cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
