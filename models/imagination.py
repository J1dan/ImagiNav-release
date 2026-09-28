"""
LTX-Video Imagination Module

Converts (Image + Prompt) → Future Video
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict
from copy import deepcopy
import torch
from PIL import Image
from diffusers.utils import export_to_video
import os
from safetensors.torch import load_file

from ImagiNav.core.config import ImaginationConfig

# Import ltxv_trainer components
from ltxv_trainer.ltxv_pipeline import LTXConditionPipeline
from ltxv_trainer.model_loader import (
    load_scheduler,
    load_tokenizer,
    load_text_encoder,
    load_vae,
    load_transformer
)
from ltxv_trainer.utils import open_image_as_srgb


class LTXVideoGenerator:
    """
    LTX-Video based imagination module.
    
    Generates first-person navigation videos from:
    - Current observation (image)
    - Video generation prompt (from Gemini)
    """
    
    def __init__(self, config: ImaginationConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize LTX-Video generator.
        
        Args:
            config: Imagination configuration
            logger: Optional logger
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.dtype)
        
        # Load pipeline components
        try:
            model_source = config.base_model if hasattr(config, 'base_model') else "LTXV_2B_0.9.6_DEV"
            
            self.logger.info(f"Loading components for {model_source}...")
            
            # 1. Load individual components (matching test_video_generation.py)
            scheduler = load_scheduler()
            tokenizer = load_tokenizer()
            
            # Text Encoder is 8-bit: AUTOMATICALLY loads to GPU. Do not move manually.
            self.logger.info("   Loading text encoder (8-bit)...")
            text_encoder = load_text_encoder(load_in_8bit=True)
            
            # VAE and Transformer need to be moved to CUDA manually
            self.logger.info("   Loading VAE (bfloat16)...")
            vae = load_vae(model_source, dtype=self.dtype).to(self.device)
            
            self.logger.info("   Loading transformer (bfloat16)...")
            transformer = load_transformer(model_source, dtype=self.dtype)

            full_checkpoint = getattr(self.config, "full_checkpoint", None)
            if full_checkpoint:
                checkpoint_path = self._resolve_checkpoint_path(full_checkpoint)
                if not checkpoint_path or not checkpoint_path.exists():
                    raise RuntimeError(f"Full transformer checkpoint not found: {full_checkpoint}")
                self.logger.info(f"📥 Loading full transformer checkpoint from {checkpoint_path}...")
                transformer.load_state_dict(load_file(checkpoint_path))
            
            # Apply quantization if needed (optional)
            # Note: Skipping quantization for now to match simpler version
            # if you want quantization, uncomment:
            # from ltxv_trainer.quantization import quantize_model
            # transformer = quantize_model(transformer, precision="fp8-quanto")
            
            # Move transformer to device
            transformer = transformer.to(self.device)
            
            # 2. Create Pipeline from components
            self.logger.info("🔧 Creating pipeline from components...")
            self.pipe = LTXConditionPipeline(
                scheduler=deepcopy(scheduler),
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
            )
            
            # DO NOT call self.pipe.to(device) - components are already on correct devices
            
            # 3. Load LoRA weights if provided
            self._lora_adapters: Dict[str, Path] = {}
            self._active_adapter: Optional[str] = None

            if self.config.lora_mode == "none":
                self.logger.info("LoRA mode is 'none'; using base model without adapters")
            elif self.config.lora_mode == "dual":
                self._load_lora_adapter(self.config.left_forward_model_path, "left_forward")
                self._load_lora_adapter(self.config.right_model_path, "right")
                # Post-load validation: ensure both adapters were loaded successfully
                missing = [name for name in ("left_forward", "right") if name not in self._lora_adapters]
                if missing:
                    msg = f"Required LoRA adapters missing after load: {missing}"
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                self._set_active_adapter("left_forward")
            else:
                self._load_lora_adapter(self.config.model_path, "default")
                # Validate default adapter loaded
                if "default" not in self._lora_adapters:
                    msg = "Configured default LoRA adapter failed to load"
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                self._set_active_adapter("default")
            
            self.logger.info("✅ LTX-Video pipeline loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to load LTX-Video model: {e}")
            raise RuntimeError(f"Model loading error: {e}")

    def _resolve_checkpoint_path(self, path_str: Optional[str]) -> Optional[Path]:
        if not path_str:
            return None
        checkpoint_path = Path(path_str)
        if not checkpoint_path.is_absolute():
            if checkpoint_path.exists():
                checkpoint_path = checkpoint_path.resolve()
            elif (Path(__file__).parent.parent / path_str).exists():
                checkpoint_path = (Path(__file__).parent.parent / path_str).resolve()
            elif (Path(__file__).parent.parent.parent / path_str).exists():
                checkpoint_path = (Path(__file__).parent.parent.parent / path_str).resolve()
        return checkpoint_path

    def _load_lora_adapter(self, path_str: Optional[str], adapter_name: str) -> None:
        # If no path provided, treat as intentional omission; caller may accept this in single-mode.
        if not path_str:
            self.logger.warning(f"LoRA path not set for adapter '{adapter_name}'")
            return
        checkpoint_path = self._resolve_checkpoint_path(path_str)
        if checkpoint_path and checkpoint_path.exists():
            self.logger.info(f"🎨 Loading LoRA '{adapter_name}' from {checkpoint_path}...")
            self.pipe.load_lora_weights(str(checkpoint_path), adapter_name=adapter_name)
            self._lora_adapters[adapter_name] = checkpoint_path
            self.logger.info(f"LoRA '{adapter_name}' loaded successfully")
        else:
            # Fail-fast: if a path was explicitly configured but not found, raise an error.
            msg = f"LoRA checkpoint not found for '{adapter_name}': {path_str}"
            self.logger.error(msg)
            raise RuntimeError(msg)

    def _set_active_adapter(self, adapter_name: str) -> None:
        if self.config.lora_mode == "none":
            return
        if adapter_name not in self._lora_adapters:
            self.logger.warning(f"Requested adapter '{adapter_name}' not loaded")
            return
        if hasattr(self.pipe, "set_adapters"):
            try:
                self.pipe.set_adapters([adapter_name])
            except TypeError:
                self.pipe.set_adapters(adapter_name)
        elif hasattr(self.pipe, "set_adapter"):
            self.pipe.set_adapter(adapter_name)
        else:
            self.logger.warning("Pipeline does not support adapter switching; using default weights")
            return
        self._active_adapter = adapter_name

    def _select_adapter_from_prompt(self, prompt: str) -> str:
        if self.config.lora_mode != "dual":
            return "default"
        text = (prompt or "").lower()
        right_keywords = [
            "turn right",
            "turning right",
            "pan right",
            "rotate right",
            "yaw right",
            "clockwise",
            "rightward",
            "to the right",
        ]
        left_keywords = [
            "turn left",
            "turning left",
            "pan left",
            "rotate left",
            "yaw left",
            "counterclockwise",
            "leftward",
            "to the left",
        ]

        has_right = any(re.search(rf"\b{re.escape(k)}\b", text) for k in right_keywords)
        has_left = any(re.search(rf"\b{re.escape(k)}\b", text) for k in left_keywords)

        if has_right and not has_left:
            return "right"
        return "left_forward"

    def _select_adapter(self, prompt: str, expert_type: Optional[str]) -> str:
        if self.config.lora_mode == "none":
            return ""
        if self.config.lora_mode != "dual":
            return "default"
        if expert_type:
            expert = str(expert_type).strip().lower()
            if expert == "right":
                return "right"
            if expert == "left":
                return "left_forward"
        return self._select_adapter_from_prompt(prompt)
    
    @staticmethod
    def _is_pure_forward(prompt: str) -> bool:
        """Return True if the (possibly already normalised) prompt contains no
        pan/turn left-right keywords, i.e. it is a pure forward-motion prompt.

        Call *after* normalize_motion_primitives so that 'turn left/right'
        variants have already been collapsed to 'pan left/right'.
        """
        text = (prompt or "").lower()
        turn_keywords = [
            "pan left",
            "pan right",
            "turn left",
            "turn right",
            "turning left",
            "turning right",
        ]
        return not any(kw in text for kw in turn_keywords)

    @staticmethod
    def normalize_motion_primitives(prompt: str) -> str:
        """Normalize common motion verb phrases to the video-model verbs.

        Examples:
            - "move forward" / "moves forward" -> "dolly forward"
            - "turn left" / "turning right" / "rotate left" -> "pan left/right"
        """
        if not prompt:
            return prompt
        text = prompt

        # Normalize forward motions -> dolly forward
        forward_patterns = [r"\b(move|moves|moving|go|goes|going|walk|walks|walking)\s+forward\b"]
        for pat in forward_patterns:
            text = re.sub(pat, "dolly forward", text, flags=re.IGNORECASE)

        # Normalize turning/rotation motions -> pan left/right
        turn_patterns = [r"\b(turn|turns|turning|rotate|rotates|rotating|yaw|yaws|yawing)\s+(left|right)\b"]
        for pat in turn_patterns:
            text = re.sub(pat, lambda m: f"pan {m.group(2).lower()}", text, flags=re.IGNORECASE)

        # Also handle patterns like 'move forward and then turn left' (already covered)

        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def generate_video(
        self,
        image_path: str,
        prompt: str,
        output_dir: Path,
        negative_prompt: str = "blurry, low quality, static, stationary",
        seed: Optional[int] = None,
        expert_type: Optional[str] = None
    ) -> Path:
        """
        Generate navigation video.
        
        Args:
            image_path: Path to input image (current observation)
            prompt: Video generation prompt
            output_dir: Directory to save output video
            negative_prompt: Negative prompt for generation
            seed: Random seed for reproducibility
        
        Returns:
            Path to generated video file
        """
        self.logger.info("Generating video...")

        # Normalize motion primitives to match video-model training
        prompt = self.normalize_motion_primitives(prompt)

        # Ensure prompt begins with 'POV, ' (idempotent)
        if prompt is None:
            prompt = ""
        # strip leading whitespace but preserve original punctuation
        if not prompt.lower().lstrip().startswith("pov,"):
            prompt = "POV, " + prompt.lstrip()
        # Capitalize first letter of the prompt body (after 'POV, ')
        if prompt.startswith("POV, ") and len(prompt) > 5:
            body = prompt[5:]
            prompt = "POV, " + (body[0].upper() + body[1:] if len(body) > 1 else body.upper())

        self.logger.info(f"Prompt (sent to model): {prompt}")

        # Select LoRA adapter based on expert_type (preferred) or prompt
        adapter_name = self._select_adapter(prompt, expert_type)
        print("self._active_adapter:", self._active_adapter)
        print("adapter_name:", adapter_name)
        if adapter_name != self._active_adapter:
            print("self._active_adapter:", self._active_adapter)
            self.logger.info(f"Switching LoRA adapter to '{adapter_name}'")
            self._set_active_adapter(adapter_name)
        
        # Load input image
        image = open_image_as_srgb(image_path)
        
        # Resize to model's expected dimensions
        image = image.resize((self.config.width, self.config.height))
        
        # Set random seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate video (with autocast like test_video_generation.py)
        try:
            with torch.autocast(str(self.device), dtype=self.dtype):
                output = self.pipe(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                    height=self.config.height,
                    width=self.config.width,
                    num_frames=self.config.num_frames,
                    generator=generator,
                )
            
            frames = output.frames[0]  # Get first (and only) video
            
            # Save video
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = output_dir / f"02_generated_video.{self.config.output_format}"
            export_to_video(frames, str(video_path), fps=self.config.frame_rate)
            
            self.logger.info(f"Video saved to: {video_path}")
            
            # Optionally save individual frames
            if self.config.save_frames:
                frames_dir = output_dir / "video_frames"
                frames_dir.mkdir(exist_ok=True)
                for i, frame in enumerate(frames):
                    frame.save(frames_dir / f"frame_{i:04d}.png")
                self.logger.info(f"Frames saved to: {frames_dir}")
            
            return video_path
        
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}")
            raise RuntimeError(f"Generation error: {e}")
    
    def generate_video_with_params(
        self,
        image_path: str,
        prompt: str,
        output_dir: Path,
        negative_prompt: str = "blurry, low quality, static, stationary",
        seed: Optional[int] = None,
        adapter_name: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        video_name: str = "02_generated_video",
    ) -> Path:
        """Generate a navigation video with explicit adapter and guidance-scale overrides.

        Used by sampling-based planning to produce multiple candidate videos with
        different configurations before ranking them with the scorer.

        Args:
            image_path: Path to the input observation image.
            prompt: Raw video-generation prompt (will be normalised internally).
            output_dir: Directory where the video file will be saved.
            negative_prompt: Negative prompt for the diffusion model.
            seed: Random seed for reproducibility.
            adapter_name: LoRA adapter to activate; falls back to currently
                active adapter when None or unavailable.
            guidance_scale: Guidance-scale override; uses config value when None.
            video_name: Output filename stem (no extension).

        Returns:
            Path to the generated video file.
        """
        # --- Prompt normalisation (same logic as generate_video) ---
        prompt = self.normalize_motion_primitives(prompt or "")
        if not prompt.lower().lstrip().startswith("pov,"):
            prompt = "POV, " + prompt.lstrip()
        if prompt.startswith("POV, ") and len(prompt) > 5:
            body = prompt[5:]
            prompt = "POV, " + (body[0].upper() + body[1:] if len(body) > 1 else body.upper())

        # --- Adapter resolution ---
        if adapter_name is None:
            adapter_name = self._active_adapter
        if adapter_name not in self._lora_adapters:
            self.logger.warning(
                f"Adapter '{adapter_name}' not loaded; falling back to '{self._active_adapter}'"
            )
            adapter_name = self._active_adapter
        if adapter_name != self._active_adapter:
            self.logger.info(f"Switching LoRA adapter to '{adapter_name}'")
            self._set_active_adapter(adapter_name)

        # --- Guidance scale ---
        eff_gs = float(guidance_scale) if guidance_scale is not None else self.config.guidance_scale

        self.logger.info(
            f"generate_video_with_params: adapter={adapter_name}, "
            f"guidance_scale={eff_gs:.2f}, video_name={video_name}"
        )
        self.logger.info(f"Prompt (sent to model): {prompt}")

        # --- Image loading ---
        image = open_image_as_srgb(image_path)
        image = image.resize((self.config.width, self.config.height))

        # --- Generator ---
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        try:
            with torch.autocast(str(self.device), dtype=self.dtype):
                output = self.pipe(
                    prompt=prompt,
                    image=image,
                    negative_prompt=negative_prompt,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=eff_gs,
                    height=self.config.height,
                    width=self.config.width,
                    num_frames=self.config.num_frames,
                    generator=generator,
                )

            frames = output.frames[0]
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            video_path = output_dir / f"{video_name}.{self.config.output_format}"
            export_to_video(frames, str(video_path), fps=self.config.frame_rate)
            self.logger.info(f"Video saved to: {video_path}")

            if self.config.save_frames:
                frames_dir = output_dir / f"{video_name}_frames"
                frames_dir.mkdir(exist_ok=True)
                for idx, frame in enumerate(frames):
                    frame.save(frames_dir / f"frame_{idx:04d}.png")
                self.logger.info(f"Frames saved to: {frames_dir}")

            return video_path

        except Exception as e:
            self.logger.error(f"Video generation failed ({video_name}): {e}")
            raise RuntimeError(f"Generation error: {e}")

    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'pipe'):
            del self.pipe
            torch.cuda.empty_cache()
            self.logger.info("LTX-Video pipeline cleaned up")
