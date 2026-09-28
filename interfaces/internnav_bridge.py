"""
InternNav Bridge for ImagiNav

Wraps ImagiNav agent as an InternNav-compatible agent.
This allows evaluation in the InternNav simulation environment.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

from ImagiNav.core.agent import ImagiNavAgent
from ImagiNav.core.config import ImagiNavConfig
from ImagiNav.core.types import TrajectoryOutput


class InternNavBridge:
    """
    Bridge between ImagiNav and InternNav simulation.
    
    This class wraps ImagiNav to be compatible with InternNav's agent interface.
    It handles:
    - Observation format conversion
    - Action execution
    - Episode management
    
    Usage with InternNav:
        ```python
        # In your InternNav agent file
        from ImagiNav.interfaces import InternNavBridge
        
        class ImagiNavInternNavAgent(Agent):
            def __init__(self, config):
                super().__init__(config)
                self.bridge = InternNavBridge(imaginav_config_path="config.yaml")
            
            def step(self, obs):
                return self.bridge.step(obs)
            
            def reset(self):
                self.bridge.reset()
        ```
    """
    
    def __init__(
        self,
        imaginav_config: ImagiNavConfig = None,
        imaginav_config_path: str = None,
        logger: logging.Logger = None
    ):
        """
        Initialize InternNav bridge.
        
        Args:
            imaginav_config: ImagiNav configuration object
            imaginav_config_path: Path to ImagiNav config file (alternative)
            logger: Optional logger
        """
        # Load config
        if imaginav_config is None and imaginav_config_path is None:
            imaginav_config = ImagiNavConfig()
        elif imaginav_config_path is not None:
            imaginav_config = ImagiNavConfig.from_yaml(imaginav_config_path)
        
        self.config = imaginav_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize ImagiNav agent
        self.agent = ImagiNavAgent(config=imaginav_config, logger=self.logger)
        
        # Episode state
        self.current_trajectory: TrajectoryOutput = None
        self.current_action_index = 0
        self.episode_count = 0
        
        # Temporary image storage
        self.temp_image_dir = Path("imaginav/temp_images")
        self.temp_image_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("InternNav bridge initialized")
    
    def step(self, obs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process observation and return action.
        
        InternNav obs format:
            [{
                'rgb': np.ndarray,  # (H, W, 3)
                'depth': np.ndarray,  # (H, W)
                'instruction': str,
                ...
            }]
        
        InternNav action format:
            [{
                'robot_name': {
                    'action_type': [forward, lateral, yaw]
                }
            }]
        
        Args:
            obs: List of observations (one per environment)
        
        Returns:
            List of actions in InternNav format
        """
        # Handle first observation - generate full trajectory
        if self.current_trajectory is None:
            self.logger.info("First step - generating trajectory...")
            
            # Extract RGB and instruction from first environment
            obs_data = obs[0]
            rgb = obs_data.get('rgb')
            instruction = obs_data.get('instruction', 'Navigate to the goal')
            
            # Save RGB to temporary file
            temp_image_path = self.temp_image_dir / f"episode_{self.episode_count}_obs.png"
            Image.fromarray((rgb * 255).astype(np.uint8)).save(temp_image_path)
            
            # Generate trajectory using ImagiNav
            self.current_trajectory = self.agent.navigate(
                image_path=str(temp_image_path),
                instruction=instruction,
                session_id=f"internnav_episode_{self.episode_count}"
            )
            
            self.current_action_index = 0
            
            if not self.current_trajectory.success:
                self.logger.error(f"Trajectory generation failed: {self.current_trajectory.error_message}")
                # Return stop action
                return [{'h1': {'stop': []}}] * len(obs)
        
        # Get current action from trajectory
        actions = []
        for _ in obs:
            if self.current_action_index < len(self.current_trajectory.navigation_commands):
                cmd = self.current_trajectory.navigation_commands[self.current_action_index]
                
                # Convert to InternNav format
                action = self._convert_to_internnav_action(cmd)
                actions.append(action)
                
                self.current_action_index += 1
            else:
                # Trajectory complete - send stop
                actions.append({'h1': {'stop': []}})
        
        return actions
    
    def _convert_to_internnav_action(self, cmd) -> Dict[str, Any]:
        """
        Convert NavigationCommand to InternNav action format.
        
        Args:
            cmd: NavigationCommand
        
        Returns:
            InternNav action dict
        """
        # InternNav expects different action types
        # Based on the action type, route to appropriate controller
        
        if abs(cmd.linear_velocity) < 0.01 and abs(cmd.angular_velocity) < 0.01:
            # Stop action
            return {'h1': {'stand_still': []}}
        
        elif abs(cmd.angular_velocity) > 0.01 and abs(cmd.linear_velocity) > 0.01:
            # Arc movement - use speed controller
            return {
                'h1': {
                    'vln_move_by_speed': [
                        cmd.linear_velocity,
                        cmd.lateral_velocity,
                        cmd.angular_velocity
                    ]
                }
            }
        
        elif abs(cmd.angular_velocity) > 0.01:
            # Pure rotation - use discrete controller with rotation
            # Convert angular velocity to discrete action
            # This is a simplification - might need tuning
            rotation_sign = 1 if cmd.angular_velocity > 0 else -1
            return {
                'h1': {
                    'move_by_discrete': [rotation_sign * 2]  # 2 = left, 4 = right
                }
            }
        
        else:
            # Pure forward - use discrete controller
            return {
                'h1': {
                    'move_by_discrete': [1]  # 1 = forward
                }
            }
    
    def reset(self, env_indices: List[int] = None, episode_ids: List[str] = None):
        """
        Reset episode state.
        
        Args:
            env_indices: Environment indices being reset
            episode_ids: Episode IDs being reset
        """
        self.current_trajectory = None
        self.current_action_index = 0
        self.episode_count += 1
        
        self.logger.info(f"Bridge reset - Episode {self.episode_count}")
    
    def get_last_trajectory(self) -> TrajectoryOutput:
        """Get the last generated trajectory."""
        return self.current_trajectory
    
    def cleanup(self):
        """Cleanup resources."""
        self.agent.cleanup()
        
        # Clean temp images
        import shutil
        if self.temp_image_dir.exists():
            shutil.rmtree(self.temp_image_dir)
