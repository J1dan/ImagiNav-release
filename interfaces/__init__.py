"""
Interfaces for connecting ImagiNav to different platforms.

Bridges:
- InternNav (simulation)
- ROS (real robot)
- Standalone execution
"""

from ImagiNav.interfaces.internnav_bridge import InternNavBridge
from ImagiNav.interfaces.ros_bridge import ROSBridge

__all__ = ["InternNavBridge", "ROSBridge"]
