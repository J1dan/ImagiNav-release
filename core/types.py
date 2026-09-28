"""
Data types for ImagiNav.

All types are designed to be JSON-serializable for easy interfacing
with ROS, web APIs, or simulation environments.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json


@dataclass
class NavigationCommand:
    """
    Single navigation command with velocity and timing.
    
    Attributes:
        linear_velocity: Forward/backward velocity (m/s)
        angular_velocity: Rotational velocity (rad/s)
        lateral_velocity: Lateral velocity (m/s) - optional, often 0
        duration: Command duration (seconds)
        timestamp: Relative timestamp in trajectory
        command_type: Type of command ('arc', 'rotate', 'forward', 'stop')
    """
    linear_velocity: float
    angular_velocity: float
    lateral_velocity: float = 0.0
    duration: float = 0.25  # Default 250ms per command
    timestamp: float = 0.0
    command_type: str = "arc"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_ros_twist(self) -> Dict[str, Any]:
        """Convert to ROS Twist message format."""
        return {
            "linear": {
                "x": self.linear_velocity,
                "y": self.lateral_velocity,
                "z": 0.0
            },
            "angular": {
                "x": 0.0,
                "y": 0.0,
                "z": self.angular_velocity
            }
        }
    
    def to_internnav_action(self) -> Tuple[float, float, float]:
        """Convert to InternNav action format [forward, lateral, yaw]."""
        return (self.linear_velocity, self.lateral_velocity, self.angular_velocity)


@dataclass
class CameraPose:
    """Camera pose from VGGT trajectory estimation."""
    position: Tuple[float, float, float]  # (x, y, z)
    rotation: Tuple[float, float, float]  # Euler angles (roll, pitch, yaw) in radians
    timestamp: float
    frame_index: int


@dataclass
class TrajectoryOutput:
    """
    Complete output from ImagiNav pipeline.
    
    This is the main output structure that can be consumed by:
    - ROS nodes (via to_ros_path())
    - InternNav simulator (via to_internnav_actions())
    - Logging/visualization tools (via to_json())
    """
    # Pipeline outputs
    reasoning_prompt: str  # LTX video prompt from Gemini
    video_path: Path  # Generated video file
    camera_trajectory: List[CameraPose]  # Camera poses from VGGT
    navigation_commands: List[Any]  # Velocity commands (list or objects)
    
    # Metadata
    instruction: str  # Original navigation instruction
    input_image_path: Path  # Initial observation
    total_duration: float  # Total trajectory duration (seconds)
    success: bool = True
    error_message: Optional[str] = None
    
    # Intermediate artifacts
    intermediate_outputs: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "reasoning_prompt": self.reasoning_prompt,
            "video_path": str(self.video_path),
            "instruction": self.instruction,
            "input_image_path": str(self.input_image_path),
            "total_duration": self.total_duration,
            "success": self.success,
            "error_message": self.error_message,
            "camera_trajectory": [
                {
                    "position": pose.position,
                    "rotation": pose.rotation,
                    "timestamp": pose.timestamp,
                    "frame_index": pose.frame_index
                }
                for pose in self.camera_trajectory
            ],
            "navigation_commands": [
                cmd.to_dict() if hasattr(cmd, "to_dict") else cmd 
                for cmd in self.navigation_commands
            ],
            "intermediate_outputs": self.intermediate_outputs,
        }
    
    def to_json(self, filepath: Optional[Path] = None) -> str:
        """
        Serialize to JSON.
        
        Args:
            filepath: If provided, save to file
            
        Returns:
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if filepath:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def to_internnav_actions(self) -> List[Tuple[float, float, float]]:
        """
        Convert to InternNav action format.
        
        Returns:
            List of (forward, lateral, yaw) tuples
        """
        actions = []
        for cmd in self.navigation_commands:
            if hasattr(cmd, 'to_internnav_action'):
                actions.append(cmd.to_internnav_action())
            elif isinstance(cmd, (list, tuple)) and len(cmd) >= 3:
                actions.append((float(cmd[0]), float(cmd[1]), float(cmd[2])))
        return actions
    
    def to_ros_path(self) -> Dict[str, Any]:
        """
        Convert to ROS nav_msgs/Path format.
        
        Returns:
            Dictionary compatible with ROS Path message
        """
        return {
            "header": {
                "frame_id": "base_link",
                "stamp": {"sec": 0, "nanosec": 0}
            },
            "poses": [
                {
                    "header": {
                        "frame_id": "base_link",
                        "stamp": {"sec": int(pose.timestamp), "nanosec": int((pose.timestamp % 1) * 1e9)}
                    },
                    "pose": {
                        "position": {
                            "x": pose.position[0],
                            "y": pose.position[1],
                            "z": pose.position[2]
                        },
                        "orientation": {
                            # Convert Euler to quaternion would go here
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "w": 1.0
                        }
                    }
                }
                for pose in self.camera_trajectory
            ]
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        return f"""
ImagiNav Trajectory Summary
{'='*60}
Instruction: {self.instruction}
Status: {'✅ Success' if self.success else f'❌ Failed: {self.error_message}'}
Duration: {self.total_duration:.2f}s
Commands: {len(self.navigation_commands)}
Camera Poses: {len(self.camera_trajectory)}

Reasoning Prompt:
  {self.reasoning_prompt}

Output Files:
  Video: {self.video_path}
  Image: {self.input_image_path}
{'='*60}
"""


@dataclass
class PipelineState:
    """Internal state tracking for the ImagiNav pipeline."""
    stage: str = "idle"  # idle, reasoning, imagination, navigation, complete
    progress: float = 0.0  # 0.0 to 1.0
    current_step: str = ""
    artifacts: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = {}
