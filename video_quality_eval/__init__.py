"""Video Quality Evaluation package for ImagiNav.

Contains evaluator to score generated videos using:
- VGGT (motion fidelity)
- Gemini (instruction following, safety, physical coherence)
- LTX-Video generator (to produce videos)
"""

__all__ = ["evaluator", "metrics"]
