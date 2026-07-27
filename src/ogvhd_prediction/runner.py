"""Public experiment runner."""

from __future__ import annotations

import importlib
import os
import sys

from .profiles import ProfileName, get_profile


def run_analysis(profile: ProfileName):
    """Run one profile in the current Python process."""
    selected = get_profile(profile)
    loaded_config = sys.modules.get("ogvhd_prediction.config")
    if loaded_config is not None:
        active = loaded_config.PROFILE.key
        if active != selected.key:
            raise RuntimeError(
                "A different profile is already loaded in this process; "
                "start a fresh Python process for the second profile."
            )
    os.environ["OGVHD_PROFILE"] = selected.key
    return importlib.import_module("ogvhd_prediction.metadata")
