"""Modular ocular GVHD prediction pipeline."""

from .profiles import ProfileName, get_profile
from .runner import run_analysis

# Export only lightweight selectors and runners so package import never loads cohort data.
__all__ = ["ProfileName", "get_profile", "run_analysis"]
