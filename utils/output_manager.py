"""
Output management utilities for ImagiNav.

Handles logging directory organization and cleanup.
"""

import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime


class OutputManager:
    """
    Manages output directory structure and cleanup.
    
    Directory structure:
        logs/
            20231201_120000/  # Session directory
                01_reasoning_prompt.txt
                02_generated_video.mp4
                03_camera_trajectory.npy
                trajectory.json
                imaginav_20231201_120000.log
    """
    
    def __init__(
        self,
        logs_dir: Path = Path("imaginav/logs"),
        use_timestamp_dirs: bool = True,
        keep_n_recent: Optional[int] = None
    ):
        """
        Initialize output manager.
        
        Args:
            logs_dir: Root logs directory
            use_timestamp_dirs: Create timestamped subdirectories
            keep_n_recent: Keep only N most recent sessions (None = keep all)
        """
        self.logs_dir = Path(logs_dir)
        self.use_timestamp_dirs = use_timestamp_dirs
        self.keep_n_recent = keep_n_recent
        
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session_dir(self, session_id: Optional[str] = None) -> Path:
        """
        Create a new session directory.
        
        Args:
            session_id: Optional session ID (default: timestamp)
        
        Returns:
            Path to session directory
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.use_timestamp_dirs:
            session_dir = self.logs_dir / session_id
        else:
            session_dir = self.logs_dir
        
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Cleanup old sessions if needed
        if self.keep_n_recent is not None:
            self._cleanup_old_sessions()
        
        return session_dir
    
    def _cleanup_old_sessions(self):
        """Remove old session directories, keeping only N most recent."""
        if not self.use_timestamp_dirs:
            return
        
        # Get all session directories
        session_dirs = [
            d for d in self.logs_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        # Sort by modification time (newest first)
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old sessions
        for old_dir in session_dirs[self.keep_n_recent:]:
            shutil.rmtree(old_dir)
    
    def get_latest_session(self) -> Optional[Path]:
        """Get the most recent session directory."""
        session_dirs = [
            d for d in self.logs_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        if not session_dirs:
            return None
        
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return session_dirs[0]
    
    def list_sessions(self) -> list:
        """List all session directories."""
        session_dirs = [
            d for d in self.logs_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        session_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return session_dirs
