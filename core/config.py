"""
Configuration management for ImagiNav.

Supports both Python dataclass and YAML file loading.
"""

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import json


@dataclass
class ReasonerConfig:
    """Configuration for Gemini reasoning module."""
    model_name: str = "gemini-2.0-flash-exp"
    scorer_model_name: str = "models/gemini-3-flash-preview"
    api_key_env: str = "GEMINI_API_KEY"  # Environment variable name
    temperature: float = 0.7
    max_output_tokens: int = 1024
    enable_scorer: bool = False
    scorer_max_video_mb: int = 20
    # Maximum number of prompts (total) to send to Gemini in a single episode.
    # If reached, the agent will treat further attempts as a STOP action [0, 0].
    # Set to None for unlimited.
    max_prompts_per_episode: Optional[int] = None
    # Seconds to wait after a successful Gemini call to avoid quota spikes
    throttle_seconds: float = 10.0
    mock_response: Optional[Dict[str, Any]] = None
    
    # Prompt template
    system_prompt: str = (
        "You are a navigation reasoning assistant. Given an image and a navigation instruction, "
        "generate a detailed prompt for a video generation model that visualizes the robot's "
        "first-person view as it executes the instruction. Focus on camera motion, scene changes, "
        "and visual landmarks."
    )
    
    prompt_template: str = (
        "Instruction: {instruction}\n\n"
        "Generate a video prompt that shows what the robot's camera would see as it follows this instruction. "
        "Describe the camera motion (forward, turning, etc.) and visual changes in detail."
    )

    scorer_system_prompt: str = (
        "You are a navigation video quality evaluator for a first-person robot navigation system. "
        "You will receive multiple candidate navigation videos, all generated from the same video prompt "
        "but with different model configurations (expert adapter and/or guidance scale). "
        "Your task is to select the single best video for guiding the robot's next step.\n\n"
        "Evaluate each video on:\n"
        "1. Camera-motion alignment — does the video faithfully reflect the motion described in the prompt?\n"
        "2. Trajectory safety — no obvious collisions, the robot does not pass through walls or solid objects.\n"
        "3. Smoothness and visual coherence — prefer fluid, realistic motion over jittery or distorted frames."
    )

    rank_prompt_template: str = (
        "{num_candidates} candidate navigation videos are provided (in order).\n"
        'Video prompt: "{video_prompt}"\n\n'
        "Candidates:\n{candidate_list}\n\n"
        "Select the best candidate. Output a JSON object with this exact structure:\n"
        '{{\n'
        '  \"best_video_number\": <integer 1 to {num_candidates}>,\n'
        '  \"justification\": \"<brief explanation referencing prompt alignment, safety, and smoothness>\",\n'
        '  \"candidate_scores\": [\n'
        '    {{\"video_number\": 1, \"score\": <0-10>, \"comment\": \"<one-line assessment>\"}},\n'
        '    ...\n'
        '  ]\n'
        '}}'
    )

    # Kept for backward compatibility (not used in sampling-based ranking mode)
    score_prompt_template: str = (
        "Video Prompt: {video_prompt}\n"
        "Generate the JSON response."
    )

    pure_llm_system_prompt: str = (
        "You are a navigation reasoning assistant. Given an image and a navigation instruction, generate a "
        "navigation decision leading to the most plausible subgoal of [x, y, theta] where:\n"
        "- x represents the forward distance (meters). Positive x is forward.\n"
        "- y represents the lateral distance (meters). Positive y is to the robot's right.\n"
        "- theta represents the in-place rotation (radians). Positive theta is clockwise.\n"
        "Use [0, 0, theta] for in-place rotation.\n"
    )

    pure_llm_prompt_template: str = (
        "Observation provided.\n"
        "Instruction: {instruction}\n\n"
        "Analyze the scene and instruction for the process of the task based on the visible landmarks and the instruction.\n"
        "Propose the next navigation command based on your analysis.\n"
        "Generate the JSON response."
    )


@dataclass
class ImaginationConfig:
    """Configuration for LTX-Video imagination module."""
    model_path: str = "checkpoints/ltx-video"  # Path to LoRA checkpoint (single mode)
    lora_mode: str = "single"  # "single" or "dual"
    left_forward_model_path: Optional[str] = None  # LoRA for left/forward motions (dual mode)
    right_model_path: Optional[str] = None  # LoRA for right-turn motions (dual mode)
    base_model: str = "Lightricks/LTX-Video"
    full_checkpoint: Optional[str] = None  # Full transformer checkpoint for full-finetuned models
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    height: int = 256
    width: int = 480
    num_frames: int = 121
    frame_rate: int = 24
    
    # Device settings
    device: str = "cuda"
    dtype: str = "bfloat16"  # or "float16"
    
    # Output settings
    output_format: str = "mp4"
    save_frames: bool = False  # Save individual frames

@dataclass
class ProportionalControlConfig:
    k_v: float = 0.8
    k_theta: float = 1.0

@dataclass
class PurePursuitConfig:
    # Moved simulation params here
    dt: float = 0.1
    goal_threshold: float = 0.2
    lookahead_distance: float = 0.3
    smooth_trajectory: bool = True

# --- Main Controller Configuration ---

@dataclass
class ControllerConfig:
    # Common Constraints
    forward_speed: float = 0.25
    rotation_speed: float = 1.5
    lateral_speed: float = 0.0
    
    # Common Safety/System
    max_steps: int = 30
    inverse_dynamics_fps: int = 10
    
    # Selector
    type: str = "pure_pursuit" 

    # Nested Specific Configs
    proportional_control: ProportionalControlConfig = field(default_factory=ProportionalControlConfig)
    pure_pursuit: PurePursuitConfig = field(default_factory=PurePursuitConfig)

    # Wrapper/Legacy params 
    use_arc_movements: bool = True
    rotate_first: Optional[bool] = None
    min_rotation_threshold: float = 0.1
    min_translation_threshold: float = 0.05

@dataclass
class NavigationConfig:
    vggt_checkpoint: str = "facebook/VGGT-1B"
    video_fps: float = 24.0
    controller: ControllerConfig = field(default_factory=ControllerConfig)

@dataclass
class LoggingConfig:
    """Configuration for logging and output management."""
    logs_dir: Path = Path("imaginav/logs")
    save_videos: bool = True
    save_trajectories: bool = True
    save_intermediate: bool = True  # Save intermediate outputs (prompts, frames, etc.)
    
    # Organization
    use_timestamp_dirs: bool = True  # Create timestamped subdirectories
    keep_n_recent: Optional[int] = 100  # Keep only N most recent runs (None = keep all)
    
    # Verbosity
    verbose: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR


@dataclass
class ImagiNavConfig:
    """
    Main configuration for ImagiNav agent.
    
    Can be instantiated directly or loaded from YAML file.
    """
    # Module configs
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    checkpoint_dir: Path = Path("imaginav/checkpoints")
    cache_dir: Path = Path("imaginav/cache")
    
    # Runtime settings
    enable_caching: bool = True  # Cache intermediate results
    parallel_execution: bool = False  # Future: parallel pipeline stages
    
    @classmethod
    def from_yaml(cls, filepath: str) -> "ImagiNavConfig":
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ImagiNavConfig":
        """Create config from dictionary."""
        # Parse nested configs
        reasoner = ReasonerConfig(**config_dict.get("reasoner", {}))
        imagination = ImaginationConfig(**config_dict.get("imagination", {}))
        
        # --- NAVIGATION & CONTROLLER LOGIC START ---
        nav_dict = config_dict.get("navigation", {}).copy()
        
        # 1. Extract explicit controller dict if it exists
        ctrl_dict = nav_dict.pop("controller", {})
        if not isinstance(ctrl_dict, dict):
            ctrl_dict = {}

        # 2. Backward Compatibility: 
        # Look for controller params sitting directly in 'navigation' and move them to ctrl_dict
        legacy_keys = [
            'forward_speed', 'rotation_speed', 'lateral_speed', 'type',
            'dt', 'max_steps', 'goal_threshold', 'lookahead_distance', 'smooth_trajectory'
        ]
        for k in legacy_keys:
            if k in nav_dict:
                ctrl_dict[k] = nav_dict.pop(k)

        # 3. Handle Nested Configs (Pure Pursuit & Proportional)
        # Ensure sub-dictionaries exist and capture any "loose" keys from ctrl_dict
        
        # 3a. Proportional Control
        prop_data = ctrl_dict.get('proportional_control')
        if not isinstance(prop_data, dict):
            prop_data = {}
        ctrl_dict['proportional_control'] = ProportionalControlConfig(**prop_data)

        # 3b. Pure Pursuit (needs migration of loose keys)
        pp_data = ctrl_dict.get('pure_pursuit')
        if not isinstance(pp_data, dict):
            pp_data = {}
            
        # Migrate specific loose keys into pure_pursuit
        pp_loose_keys = ['dt', 'goal_threshold', 'lookahead_distance', 'smooth_trajectory']
        for k in pp_loose_keys:
            if k in ctrl_dict:
                pp_data[k] = ctrl_dict.pop(k)
        
        ctrl_dict['pure_pursuit'] = PurePursuitConfig(**pp_data)

        # 4. Create Controller & Navigation Objects
        controller_config = ControllerConfig(**ctrl_dict)
        navigation = NavigationConfig(controller=controller_config, **nav_dict)
        # --- NAVIGATION & CONTROLLER LOGIC END ---
        
        logging = LoggingConfig(**config_dict.get("logging", {}))
        
        # Parse global settings
        global_settings = {
            k: v for k, v in config_dict.items() 
            if k not in ["reasoner", "imagination", "navigation", "logging"]
        }
        
        return cls(
            reasoner=reasoner,
            imagination=imagination,
            navigation=navigation,
            logging=logging,
            **global_settings
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        config_dict = convert_paths(config_dict)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration."""
        errors = []
        warnings = []

        def resolve_lora_path(path_str: Optional[str]) -> Optional[Path]:
            if not path_str:
                return None
            ltx_path = Path(path_str)
            if not ltx_path.is_absolute():
                if ltx_path.exists():
                    ltx_path = ltx_path.resolve()
                elif (Path(__file__).parent.parent / path_str).exists():
                    ltx_path = (Path(__file__).parent.parent / path_str).resolve()
                elif (Path(__file__).parent.parent.parent / path_str).exists():
                    ltx_path = (Path(__file__).parent.parent.parent / path_str).resolve()
            return ltx_path
        
        # Check LoRA paths based on mode
        if self.imagination.lora_mode not in {"single", "dual"}:
            errors.append(f"Invalid imagination.lora_mode: {self.imagination.lora_mode} (must be 'single' or 'dual')")
        elif self.imagination.lora_mode == "single":
            ltx_path = resolve_lora_path(self.imagination.model_path)
            if ltx_path and not ltx_path.exists():
                errors.append(f"LoRA checkpoint REQUIRED but not found: {self.imagination.model_path}")
        else:
            left_path = resolve_lora_path(self.imagination.left_forward_model_path)
            right_path = resolve_lora_path(self.imagination.right_model_path)
            if not self.imagination.left_forward_model_path:
                errors.append("left_forward_model_path is required when lora_mode='dual'")
            elif left_path and not left_path.exists():
                errors.append(
                    f"Left/forward LoRA checkpoint not found: {self.imagination.left_forward_model_path}"
                )
            if not self.imagination.right_model_path:
                errors.append("right_model_path is required when lora_mode='dual'")
            elif right_path and not right_path.exists():
                errors.append(
                    f"Right-turn LoRA checkpoint not found: {self.imagination.right_model_path}"
                )
        
        # VGGT checkpoint validation
        vggt_ckpt = self.navigation.vggt_checkpoint
        if vggt_ckpt and not vggt_ckpt.startswith(("facebook/", "google/", "microsoft/")):
            vggt_path = Path(vggt_ckpt)
            if not vggt_path.exists():
                warnings.append(f"VGGT checkpoint not found: {vggt_ckpt} (will attempt HuggingFace download)")
        
        # Check numeric ranges
        if not 0 <= self.reasoner.temperature <= 2.0:
            errors.append(f"Invalid temperature: {self.reasoner.temperature} (must be 0-2.0)")

        # Validate reasoner prompt limits
        if getattr(self.reasoner, 'max_prompts_per_episode', None) is not None:
            v = self.reasoner.max_prompts_per_episode
            if not isinstance(v, int) or v < 0:
                errors.append(f"Invalid reasoner.max_prompts_per_episode: {v} (must be non-negative integer or null)")
        
        if not 0 <= self.imagination.guidance_scale <= 20:
            errors.append(f"Invalid guidance scale: {self.imagination.guidance_scale} (must be 0-20)")
        
        if self.imagination.num_frames < 1:
            errors.append(f"Invalid num_frames: {self.imagination.num_frames} (must be >= 1)")
        
        # Print warnings
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        # Print errors
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  ❌ {error}")
            return False
        
        return True
