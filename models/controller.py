import numpy as np
from typing import List, Tuple, Union, Optional
import math

class Controller:
    def __init__(self):
        pass

    def get_action(self, waypoints: np.ndarray) -> Tuple[float, float, float]:
        raise NotImplementedError

    def calculate_trajectory_commands(
        self, 
        waypoints: np.ndarray, 
        dt: Optional[float] = None,
    ) -> List[List[float]]:
        """
        Simulate tracking the waypoints and generate a sequence of commands.
        Returns list of [linear_velocity, lateral_velocity, angular_velocity]
        """
        raise NotImplementedError

    def direct_calculate_trajectory_commands(
        self,
        waypoints: np.ndarray,
    ) -> List[List[float]]:
        """
        Directly compute commands from neighboring waypoints.
        Returns list of [linear_velocity, lateral_velocity, angular_velocity].
        """
        raise NotImplementedError

class PurePursuitController(Controller):
    def __init__(self, lookahead_distance: float = 0.25, max_linear_speed: float = 0.25, max_angular_speed: float = 1.5, smooth_trajectory: bool = True, dt: float = 0.3, max_steps: int = 30, goal_threshold: float = 0.2):
        super().__init__()
        self.lookahead_distance = lookahead_distance
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.smooth_trajectory = smooth_trajectory
        self.dt = dt
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold

    def calculate_trajectory_commands(
        self, 
        waypoints: np.ndarray, 
    ) -> List[List[float]]:
        """
        Simulate tracking the waypoints and generate a sequence of commands.
        """
        if len(waypoints) == 0:
            return []
            
        commands = []
        
        # Simulation state (Robot starts at 0,0,0)
        current_x = 0.0
        current_y = 0.0
        current_yaw = 0.0
        current_time = 0.0

        for _ in range(self.max_steps):
            # 1. Transform waypoints to local frame of the virtual robot
            dx = waypoints[:, 0] - current_x
            dy = waypoints[:, 1] - current_y
            
            # Rotate by -current_yaw
            c = math.cos(-current_yaw)
            s = math.sin(-current_yaw)
            
            local_x = c * dx - s * dy
            local_y = s * dx + c * dy
            
            local_waypoints = np.stack([local_x, local_y], axis=1)
            
            # 2. Get action from controller
            v, _, w = self.get_action(local_waypoints)
            
            # 3. Check termination condition
            # Distance to final point
            dist_to_goal = np.linalg.norm(local_waypoints[-1])
            if dist_to_goal < self.goal_threshold:
                break
                
            # Check if we passed the last waypoint (projected progress > 100% on last segment)
            # Only check this if we are relatively close to the goal, to avoid false positives 
            # if the last segment points backwards (e.g. noise) and we are far away.
            if len(local_waypoints) >= 2 and dist_to_goal < (self.goal_threshold + self.lookahead_distance):
                # Last segment vector
                p_end = local_waypoints[-1]
                p_prev = local_waypoints[-2]
                seg_v = p_end - p_prev
                seg_len_sq = np.dot(seg_v, seg_v)
                
                if seg_len_sq > 1e-6:
                    # Project vector from p_prev to robot(0,0) onto seg_v
                    # robot_vec = (0,0) - p_prev = -p_prev
                    robot_vec = -p_prev
                    t_proj = np.dot(robot_vec, seg_v) / seg_len_sq
                    
                    if t_proj > 1.0:
                        break
            
            # 4. Record command
            cmd = [
                float(v),
                0.0,
                float(w)
            ]
            commands.append(cmd)
            
            # 5. Update state (Euler integration)
            current_time += self.dt
            current_yaw += w * self.dt
            current_x += v * math.cos(current_yaw) * self.dt
            current_y += v * math.sin(current_yaw) * self.dt
            
        # Smooth trajectory if enabled
        if self.smooth_trajectory:
            commands = self._smooth_commands(commands)
            
        return commands

    def _smooth_commands(
        self,
        commands: List[List[float]]
    ) -> List[List[float]]:
        """Apply smoothing to command sequence."""
        # Simple smoothing: average with neighbors
        if len(commands) < 3:
            return commands
        
        smoothed = [commands[0]]  # Keep first command
        
        for i in range(1, len(commands) - 1):
            prev_cmd = commands[i - 1]
            curr_cmd = commands[i]
            next_cmd = commands[i + 1]
            
            smoothed_cmd = [
                (prev_cmd[0] + curr_cmd[0] + next_cmd[0]) / 3,
                (prev_cmd[1] + curr_cmd[1] + next_cmd[1]) / 3,
                (prev_cmd[2] + curr_cmd[2] + next_cmd[2]) / 3
            ]
            smoothed.append(smoothed_cmd)
        
        smoothed.append(commands[-1])  # Keep last command
        return smoothed

    def get_action(self, waypoints: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate action based on waypoints in robot frame.
        Robot is at (0,0) facing +X.
        
        Args:
            waypoints: (N, 2) or (N, 3) array of points [x, y] or [x, y, yaw]
        
        Returns:
            action: (forward, lateral, yaw) - velocities or displacements
            For differential drive/pure pursuit, lateral is usually 0.
        """
        if len(waypoints) == 0:
            return 0.0, 0.0, 0.0

        # Find lookahead point with interpolation
        lookahead_sq = self.lookahead_distance ** 2
        dists_sq = np.sum(waypoints[:, :2]**2, axis=1)
        
        target = waypoints[-1] # Default to last point
        
        if dists_sq[0] > lookahead_sq:
            # If start is already outside, target the start
            target = waypoints[0]
        else:
            # Look for the first segment that exits the lookahead circle
            for i in range(len(waypoints) - 1):
                d1_sq = dists_sq[i]
                d2_sq = dists_sq[i+1]
                
                if d1_sq < lookahead_sq and d2_sq >= lookahead_sq:
                    # Intersection is on this segment
                    p1 = waypoints[i, :2]
                    p2 = waypoints[i+1, :2]
                    V = p2 - p1
                    
                    # Quadratic for intersection: |p1 + t*V|^2 = L^2
                    a = np.dot(V, V)
                    b = 2 * np.dot(p1, V)
                    c = d1_sq - lookahead_sq
                    
                    if abs(a) < 1e-6:
                        continue
                        
                    disc = b**2 - 4*a*c
                    if disc < 0:
                        continue
                        
                    sqrt_disc = math.sqrt(disc)
                    # We want t in [0, 1]. Since d1 < L and d2 >= L, there is one valid positive root
                    t1 = (-b - sqrt_disc) / (2*a)
                    t2 = (-b + sqrt_disc) / (2*a)
                    
                    t = None
                    if 0 <= t2 <= 1:
                        t = t2
                    elif 0 <= t1 <= 1:
                        t = t1
                        
                    if t is not None:
                        # Interpolate
                        if waypoints.shape[1] > 2:
                            target = np.zeros(waypoints.shape[1])
                            target[:2] = p1 + t * V
                            # Linearly interpolate other dimensions if present (like z or yaw)
                            target[2:] = waypoints[i, 2:] + t * (waypoints[i+1, 2:] - waypoints[i, 2:])
                        else:
                            target = p1 + t * V
                        break

        x = target[0]
        y = target[1]
        
        
        # Calculate curvature
        # L = distance to target
        # y = lateral offset
        # R = L^2 / (2y)
        # curvature = 1/R = 2y / L^2
        
        L2 = x**2 + y**2
        if L2 < 1e-6:
            return 0.0, 0.0, 0.0
            
        curvature = 2 * y / L2
        
        # Calculate velocities
        # v = max_speed
        # w = v * curvature
        
        v = self.max_linear_speed
        w = v * curvature
        
        # Clamp angular speed
        w = np.clip(w, -self.max_angular_speed, self.max_angular_speed)
        
        # If we are very close to the goal (last point), we might want to stop or slow down
        # But for now, let's just output v, 0, w
        
        # InternNav usually expects [forward, lateral, yaw]
        # If it's velocity control:
        return v, 0.0, w * 0.5

class PointGoalController(Controller):
    """
    Simple controller that just goes towards the first waypoint.
    """
    def __init__(self, max_linear_speed: float = 0.25, max_angular_speed: float = 3.0):
        super().__init__()
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed

    def get_action(self, waypoints: np.ndarray) -> Tuple[float, float, float]:
        if len(waypoints) == 0:
            return 0.0, 0.0, 0.0
            
        target = waypoints[0] # Just take the first one? Or the last one?
        # If waypoints is a path, we should probably use Pure Pursuit.
        # If waypoints is just a goal, we use this.
        
        x = target[0]
        y = target[1]
        
        angle = np.arctan2(y, x)
        dist = np.sqrt(x**2 + y**2)
        
        if dist < 0.1:
            return 0.0, 0.0, 0.0
            
        # Turn then move, or move and turn
        # Simple proportional control
        w = angle * 2.0
        v = dist * 0.5
        
        v = np.clip(v, 0, self.max_linear_speed)
        w = np.clip(w, -self.max_angular_speed, self.max_angular_speed)
        
        return v, 0.0, w


class ProportionalController(Controller):
    """
    A minimal controller that directly maps the difference between neighboring 
    waypoints to velocity commands.
    
    Logic:
    v_cmd = kv * distance(current, target)
    w_cmd = ktheta * angle_diff(current_yaw, target_yaw)
    """
    def __init__(
        self, 
        kv: float = 0.5, 
        ktheta: float = 2.0, 
        max_linear_speed: float = 0.5, 
        max_angular_speed: float = 1.5,
        max_steps: int = 30
    ):
        super().__init__()
        self.kv = kv          
        self.ktheta = ktheta  
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_steps = max_steps

    def get_action(self, waypoints: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates action directly. 
        Expects waypoints[0] to be the LOCAL difference [dx, dy, dyaw].
        """
        if len(waypoints) == 0:
            return 0.0, 0.0, 0.0

        target = waypoints[0]
        dx = target[0]
        dy = target[1]
        dyaw = target[2] # Use the explicit yaw difference provided

        # 1. Calculate Errors
        dist_error = np.sqrt(dx**2 + dy**2)
        # Note: We use the dyaw directly as requested, instead of arctan2(dy, dx)
        
        # 2. Calculate Commands (P-Control)
        v_cmd = self.kv * dist_error
        w_cmd = self.ktheta * dyaw

        # 3. Handle In-Place Rotation 
        # If the orientation difference is large, stop moving linearly to turn faster/safer.
        if abs(dyaw) > np.pi / 4:
            v_cmd = 0.0

        # 4. Clip limits
        v_cmd = np.clip(v_cmd, 0, self.max_linear_speed)
        w_cmd = np.clip(w_cmd, -self.max_angular_speed, self.max_angular_speed)

        return v_cmd, 0.0, w_cmd

    def calculate_trajectory_commands(
        self, 
        waypoints: np.ndarray, 
        dt: Optional[float] = None,
        max_steps: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Compute commands directly from consecutive waypoints without simulating full dynamics.

        For each segment (p_i -> p_{i+1}) starting from robot origin (0,0) -> waypoints[0],
        compute:
          - dx, dy = p_next - p_curr
          - distance = sqrt(dx^2 + dy^2)
          - target_yaw = atan2(dy, dx)
          - yaw_error = normalized_angle(target_yaw - current_yaw)
          - v_cmd = kv * distance (clipped to max_linear_speed)
          - w_cmd = ktheta * yaw_error (clipped to max_angular_speed)

        If |yaw_error| > pi/4, v_cmd is set to 0 (in-place rotation preference).
        The controller updates current_yaw to target_yaw for the next segment (no full integration).

        This returns at most `max_steps` commands (or `self.max_steps` by default).
        """
        if max_steps is None:
            max_steps = self.max_steps

        if len(waypoints) < 1:
            return []

        # Build list of segment endpoints starting from origin
        pts = [np.array([0.0, 0.0])]
        for i in range(len(waypoints)):
            pts.append(np.array(waypoints[i][:2]))

        commands: List[List[float]] = []
        current_yaw = 0.0

        for i in range(len(pts) - 1):
            if len(commands) >= max_steps:
                break

            p0 = pts[i]
            p1 = pts[i + 1]
            dx = float(p1[0] - p0[0])
            dy = float(p1[1] - p0[1])
            dist = math.hypot(dx, dy)
            if dist < 0.01:
                continue

            target_yaw = math.atan2(dy, dx)
            yaw_error = target_yaw - current_yaw
            yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

            # Compute commands
            v_cmd = float(np.clip(self.kv * dist, 0.0, self.max_linear_speed))
            w_cmd = float(np.clip(self.ktheta * yaw_error, -self.max_angular_speed, self.max_angular_speed))

            commands.append([v_cmd, 0.0, w_cmd])

            # Update current_yaw to the target heading (no full kinematic sim)
            current_yaw = target_yaw

        return commands

    def direct_calculate_trajectory_commands(
        self,
        waypoints: np.ndarray,
    ) -> List[List[float]]:
        """
        Directly compute commands from neighboring waypoints using k gains.
        """
        if len(waypoints) < 2:
            return []

        commands: List[List[float]] = []

        for i in range(len(waypoints) - 1):
            curr_wp = waypoints[i]
            next_wp = waypoints[i + 1]

            dx = float(next_wp[0] - curr_wp[0])
            dy = float(next_wp[1] - curr_wp[1])
            dyaw = float(next_wp[2] - curr_wp[2])
            dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi

            dist = math.hypot(dx, dy)

            v_cmd = self.kv * dist
            w_cmd = self.ktheta * dyaw

            if abs(dyaw) > math.pi / 4:
                v_cmd = 0.0

            v_cmd = float(np.clip(v_cmd, 0.0, self.max_linear_speed))
            w_cmd = float(np.clip(w_cmd, -self.max_angular_speed, self.max_angular_speed))

            commands.append([v_cmd, 0.0, w_cmd])

        return commands[:self.max_steps]