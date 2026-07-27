"""Public experiment runner."""

from __future__ import annotations

from .profiles import ProfileName, get_profile


def run_analysis(profile: ProfileName):
    """Run one profile through the explicit stage pipeline."""
    # Import orchestration lazily so importing the public package remains side-effect-free.
    from .pipeline import run_pipeline

    return run_pipeline(get_profile(profile))
