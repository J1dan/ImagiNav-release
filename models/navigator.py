"""
VGGT Navigation Module

Converts Video → Camera Trajectory → Velocity Commands

Based on: ~/chenjie/test/vggt/infer_move.py
And controller logic from: goal_to_actions.py
"""

import logging
import math
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from PIL import Image
import cv2

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from ImagiNav.core.config import NavigationConfig
from ImagiNav.core.types import CameraPose, NavigationCommand
from ImagiNav.models.controller import PurePursuitController, ProportionalController


class VGGTNavigator:
    """
    VGGT-based navigation module.
    
    Pipeline:
    1. VGGT: Video → Camera trajectory (positions, orientations)
    2. Controller: Camera trajectory → Velocity commands (v, ω)
    """
    
    def __init__(self, config: NavigationConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize VGGT navigator.
        
        Args:
            config: Navigation configuration
            logger: Optional logger
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Device and dtype setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")  # Force CPU for now
        if self.device.type == "cuda":
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        
        # Load VGGT model
        self.logger.info(f"Loading VGGT model from: {config.vggt_checkpoint}")
        self.logger.info(f"Using device={self.device}, dtype={self.dtype}")
        self._load_vggt_model()
        
        # Initialize Controller based on Config
        self.c_conf = config.controller
        
        if hasattr(self.c_conf, 'type') and self.c_conf.type == "proportional":
            self.logger.info("Initializing ProportionalController")
            self.controller = ProportionalController(
                kv=self.c_conf.proportional_control.k_v,
                ktheta=self.c_conf.proportional_control.k_theta,
                max_linear_speed=self.c_conf.forward_speed,
                max_angular_speed=self.c_conf.rotation_speed,
                max_steps=self.c_conf.max_steps
            )
        else:
            self.logger.info("Initializing PurePursuitController (Default)")
            self.controller = PurePursuitController(
                lookahead_distance=self.c_conf.lookahead_distance,
                max_linear_speed=self.c_conf.forward_speed,
                max_angular_speed=self.c_conf.rotation_speed,
                smooth_trajectory=self.c_conf.smooth_trajectory,
                dt=self.c_conf.pure_pursuit.dt,
                max_steps=self.c_conf.max_steps
            )
    
    def _load_vggt_model(self):
        """
        Load VGGT model.
        
        Based on vggt/infer_move.py implementation.
        """
        try:
            # Load pretrained VGGT model
            model_name = self.config.vggt_checkpoint
            if model_name is None or model_name == "":
                model_name = "facebook/VGGT-1B"  # Default model
            
            self.logger.info(f"Loading VGGT model: {model_name}")
            self.vggt_model = VGGT.from_pretrained(model_name).to(self.device)
            self.vggt_model.eval()
            self.logger.info("VGGT model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load VGGT model: {e}")
            raise
    
    def extract_trajectory(
        self,
        video_path: Path,
        output_dir: Path
    ) -> Tuple[List[CameraPose], List[List[float]]]:
        """
        Extract navigation trajectory from video.
        
        Args:
            video_path: Path to generated video
            output_dir: Output directory for intermediate files
        
        Returns:
            Tuple of (camera_poses, navigation_commands)
            navigation_commands is a list of [v, lat, w] lists.
        """
        self.logger.info("Extracting camera trajectory from video...")
        
        # Step 1: VGGT - Extract camera poses
        camera_poses = self._extract_camera_poses(video_path)
        
        self.logger.info(f"Extracted {len(camera_poses)} camera poses")
        
        # Save camera trajectory
        if output_dir:
            self._save_camera_trajectory(camera_poses, output_dir / "03_camera_trajectory.npy")
        
        # Step 2: Controller - Convert to velocity commands
        navigation_commands = self._trajectory_to_commands(camera_poses)
        
        self.logger.info(f"Generated {len(navigation_commands)} navigation commands")
        
        return camera_poses, navigation_commands
    
    def _extract_camera_poses(self, video_path: Path) -> List[CameraPose]:
        """
        Use VGGT to extract camera poses from video.
        
        Based on vggt/infer_move.py implementation.
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of camera poses
        """
        if self.vggt_model is None:
            raise RuntimeError("VGGT model not loaded")
        
        # Step 1: Extract frames from video
        self.logger.info(f"Extracting frames from video: {video_path}")
        frames_dir = video_path.parent / f"{video_path.stem}_frames"
        frames_dir.mkdir(exist_ok=True)
        
        frame_paths = self._extract_video_frames(video_path, frames_dir)
        self.logger.info(f"Extracted {len(frame_paths)} frames")
        
        if len(frame_paths) < 2:
            raise ValueError("Need at least 2 frames for trajectory extraction")
        
        # Step 2: Run VGGT on consecutive frame pairs
        camera_poses = []
        
        # Initialize first pose at origin
        camera_poses.append(CameraPose(
            position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
            timestamp=0.0,
            frame_index=0
        ))
        
        current_position = np.array([0.0, 0.0, 0.0])
        current_yaw = 0.0
        
        for i in range(len(frame_paths) - 1):
            current_frame = frame_paths[i]
            next_frame = frame_paths[i + 1]
            
            # Load and preprocess images
            images_tensor = load_and_preprocess_images(
                [str(current_frame), str(next_frame)]
            ).to(self.device)
            
            if images_tensor.dim() == 3:
                images_tensor = images_tensor.unsqueeze(0)
            
            # Run VGGT inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    images_scene = images_tensor.unsqueeze(0)
                    aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(images_scene)
                    pose_enc_list = self.vggt_model.camera_head(aggregated_tokens_list)
                    pose_enc = pose_enc_list[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        pose_enc, images_scene.shape[-2:]
                    )
            
            # Extract camera poses from extrinsic matrices
            extrinsic0 = extrinsic[0, 0]  # Current frame pose
            extrinsic1 = extrinsic[0, 1]  # Next frame pose
            
            # Calculate relative motion using infer_move.py logic
            yaw_deg, forward_m = self._compute_relative_motion(extrinsic0, extrinsic1)
            
            # Update global pose
            current_yaw += math.radians(yaw_deg)
            
            # Update position based on new yaw
            # Forward movement is along the NEW yaw direction
            dx = forward_m * math.cos(current_yaw)
            dy = forward_m * math.sin(current_yaw)
            current_position += np.array([dx, dy, 0.0])
            
            # Add new pose
            camera_poses.append(CameraPose(
                position=tuple(current_position.tolist()),
                rotation=(0.0, 0.0, current_yaw),  # Only tracking yaw
                timestamp=(i + 1) / self.config.controller.inverse_dynamics_fps,
                frame_index=i + 1
            ))
        
        self.logger.info(f"Extracted {len(camera_poses)} camera poses")
        return camera_poses

    def _compute_relative_motion(self, extrinsic1, extrinsic2) -> Tuple[float, float]:
        """
        Compute relative motion between two extrinsics.
        Matches compute_action_from_poses in infer_move.py
        
        Returns:
            (yaw_deg, forward_m)
        """
        # Ensure inputs are on CPU and numpy
        if isinstance(extrinsic1, torch.Tensor):
            extrinsic1 = extrinsic1.detach().cpu().numpy()
        if isinstance(extrinsic2, torch.Tensor):
            extrinsic2 = extrinsic2.detach().cpu().numpy()

        # Convert 3x4 extrinsics to 4x4 if needed
        if extrinsic1.shape == (3, 4):
            T1 = np.eye(4)
            T1[:3] = extrinsic1
        else:
            T1 = extrinsic1
        
        if extrinsic2.shape == (3, 4):
            T2 = np.eye(4)
            T2[:3] = extrinsic2
        else:
            T2 = extrinsic2

        # relative transform mapping coords in cam1 to cam2
        T1_inv = np.linalg.inv(T1)
        T_rel = T2 @ T1_inv

        R = T_rel[:3,:3]
        t = T_rel[:3,3]

        # yaw extraction: atan2(R[0,2], R[2,2])
        yaw_rad = math.atan2(R[0,2], R[2,2])
        yaw_deg = math.degrees(yaw_rad)

        # forward distance: -t[2]
        forward_m = float(-t[2])
        
        return yaw_deg, forward_m
    
    def _extract_video_frames(
        self,
        video_path: Path,
        output_dir: Path,
        target_fps: float = None
    ) -> List[Path]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            target_fps: Target FPS for extraction (None = use video config)
        
        Returns:
            List of frame paths
        """
        if target_fps is None:
            target_fps = self.config.controller.inverse_dynamics_fps
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(original_fps / target_fps)))
        
        frame_count = 0
        saved_count = 0
        frame_paths = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at target FPS intervals
            if frame_count % frame_interval == 0:
                frame_filename = output_dir / f"{saved_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                frame_paths.append(frame_filename)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        self.logger.info(
            f"Extracted {saved_count} frames (original FPS: {original_fps:.2f}, "
            f"target: {target_fps})"
        )
        
        return frame_paths
    
    def _extrinsic_to_camera_pose(
        self,
        extrinsic: torch.Tensor,
        frame_index: int
    ) -> CameraPose:
        """
        Convert extrinsic matrix to CameraPose.
        
        Args:
            extrinsic: 4x4 or 3x4 extrinsic matrix (camera from world)
            frame_index: Frame index
        
        Returns:
            CameraPose object
        """
        if isinstance(extrinsic, torch.Tensor):
            extrinsic = extrinsic.detach().cpu().numpy()
        
        # Convert 3x4 to 4x4 if needed
        if extrinsic.shape == (3, 4):
            T = np.eye(4)
            T[:3] = extrinsic
        else:
            T = extrinsic
        
        # Extract rotation matrix and translation vector
        R = T[:3, :3]
        t = T[:3, 3]
        
        # Convert to position (world coordinates)
        # For camera extrinsic (camera from world), the camera position in world is -R^T @ t
        position = -R.T @ t
        
        # Extract Euler angles from rotation matrix (ZYX convention)
        # This gives roll, pitch, yaw
        sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        
        rotation = (roll, pitch, yaw)
        
        # Timestamp from frame index
        timestamp = frame_index / self.config.controller.inverse_dynamics_fps
        
        return CameraPose(
            position=tuple(position.tolist()),
            rotation=rotation,
            timestamp=timestamp,
            frame_index=frame_index
        )
    
    def _trajectory_to_commands(
        self,
        camera_poses: List[CameraPose]
    ) -> List[List[float]]:
        """
        Convert camera trajectory to velocity commands using PurePursuitController.
        
        Args:
            camera_poses: List of camera poses
        
        Returns:
            List of navigation commands [v, lat, w]
        """
        if len(camera_poses) < 2:
            return []
        
        # Convert poses to waypoints (N, 3)
        # We assume the trajectory starts at (0,0,0) or close to it.
        waypoints = np.array([[p.position[0], p.position[1], p.rotation[2]] for p in camera_poses])
        
        commands = []
        
        # Simulation state
        # current_x = 0.0
        # current_y = 0.0
        # current_yaw = 0.0
        # current_time = 0.0
        
        

        commands = self.controller.direct_calculate_trajectory_commands(waypoints=waypoints)

        # commands = self.controller.calculate_trajectory_commands(waypoints=waypoints)
            
        self.logger.info(f"Generated {len(commands)} commands using controller")
        
        return commands
    
    def _save_camera_trajectory(self, poses: List[CameraPose], filepath: Path):
        """Save camera trajectory to file."""
        data = {
            'positions': np.array([p.position for p in poses]),
            'rotations': np.array([p.rotation for p in poses]),
            'timestamps': np.array([p.timestamp for p in poses])
        }
        np.save(filepath, data)
        self.logger.info(f"Saved camera trajectory to: {filepath}")

    def direct_navigate(self, navigation_command: List[float]) -> List[List[float]]:
        """
        Convert a reasoner command [x, y, theta] into control commands.

        If x and y are near 0 and theta is non-zero, perform turn-in-place.
        Otherwise, rotate toward waypoint [x, y] and move forward. Theta is ignored for non-zero [x, y].
        """
        if not isinstance(navigation_command, (list, tuple)) or len(navigation_command) < 2:
            self.logger.warning("Invalid navigation_command; returning stop")
            return [[0.0, 0.0, 0.0]]

        # Normalize inputs to [x, y, theta]
        if len(navigation_command) == 2:
            x = float(navigation_command[0])
            y = 0.0
            theta_deg = float(navigation_command[1])
        else:
            x = float(navigation_command[0])
            y = float(navigation_command[1])
            theta_deg = float(navigation_command[2])

        # For waypoint navigation, use heading from [x, y]
        if abs(x) >= 1e-6 or abs(y) >= 1e-6:
            theta_deg = math.degrees(math.atan2(y, x))

        # Use controller timing
        if self.c_conf.inverse_dynamics_fps and self.c_conf.inverse_dynamics_fps > 0:
            dt = 1.0 / float(self.c_conf.inverse_dynamics_fps)
        else:
            dt = getattr(self.c_conf.pure_pursuit, 'dt', 0.25)

        max_lin = getattr(self.controller, 'max_linear_speed', self.c_conf.forward_speed)
        max_ang = getattr(self.controller, 'max_angular_speed', self.c_conf.rotation_speed)

        def _build_turn_commands(angle_deg: float) -> List[List[float]]:
            """
            Build rotation commands for an in-place turn.

            - `angle_deg` is the total desired rotation in degrees (positive = CCW).
            - Commands are returned as [v, lat, w] where `w` is in radians/sec.
            """
            if abs(angle_deg) < 1e-3:
                return []
            abs_deg = abs(angle_deg)
            per_step_deg = 3.0  # degrees per command step
            num_steps = max(1, int(math.ceil(abs_deg / per_step_deg)))
            step_deg = abs_deg / num_steps
            direction = 1.0 if angle_deg >= 0 else -1.0
            # Convert step angle (deg) -> radians, then divide by dt to get rad/s
            step_ang_rad = math.radians(step_deg)
            step_ang_vel = direction * (step_ang_rad / dt) * 0.75  # 0.75 factor to be conservative
            # `max_ang` from controller/config is expected to be in rad/s (see default_config.yaml)
            max_ang_rad = max_ang
            if abs(step_ang_vel) > max_ang_rad:
                step_ang_vel = direction * max_ang_rad
            return [[0.0, 0.0, float(step_ang_vel)] for _ in range(num_steps)]

        def _build_forward_commands(distance: float) -> List[List[float]]:
            if distance <= 0.0:
                return []
            step_dist = max_lin * dt
            num_steps = max(1, int(math.ceil(distance / max(step_dist, 1e-6))))
            lin_vel = min(max_lin, distance / (num_steps * dt))
            return [[float(lin_vel), 0.0, 0.0] for _ in range(num_steps)]

        # In-place rotation: x=y=0, theta!=0
        if abs(x) < 1e-6 and abs(y) < 1e-6 and abs(theta_deg) > 1.0:
            commands = _build_turn_commands(theta_deg)
            self.logger.info(
                f"Direct turn: {theta_deg}° -> {len(commands)} steps"
            )
            return commands

        # Waypoint navigation: rotate toward waypoint, move forward (ignore theta)
        commands: List[List[float]] = []
        distance = math.hypot(x, y)
        if distance > 1e-6:
            heading_deg = math.degrees(math.atan2(y, x))
            commands += _build_turn_commands(heading_deg)
            commands += _build_forward_commands(distance)

        if distance > 1e-6 and abs(theta_deg) > 1.0:
            self.logger.info("Ignoring theta for waypoint navigation; using heading from [x, y].")

        if commands:
            self.logger.info(
                f"Direct waypoint: [x={x:.3f}, y={y:.3f}, theta={theta_deg:.2f}°] -> {len(commands)} steps"
            )
            return commands

        self.logger.info("Direct navigate: zero command -> stop")
        return [[0.0, 0.0, 0.0]]
    
    def cleanup(self):
        """Free resources."""
        if hasattr(self, 'vggt_model') and self.vggt_model is not None:
            del self.vggt_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("VGGT model cleaned up")
