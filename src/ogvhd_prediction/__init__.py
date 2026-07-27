"""Modular ocular GVHD prediction pipeline."""

from .profiles import ProfileName, get_profile
from .runner import run_analysis

# Keep the public surface small; importing the package must not load cohort data.
__all__ = ["ProfileName", "get_profile", "run_analysis"]
