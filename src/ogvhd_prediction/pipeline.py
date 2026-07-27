"""Explicit orchestration for the ocular GVHD modeling stages."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import sys
from typing import Any

from .profiles import ProfileDefaults, ProfileName, get_profile


MODULE_ORDER = (
    "config",
    "data",
    "targets",
    "metrics",
    "preprocessing",
    "splitting",
    "tuning",
    "explainability",
    "temporal",
    "simplified",
    "audit",
    "evaluation",
    "finalization",
    "reporting",
    "subgroups",
    "diagnostics",
    "panels",
    "metadata",
)

EXECUTION_STAGES = (
    "data",
    "targets",
    "evaluation",
    "finalization",
    "reporting",
    "subgroups",
    "diagnostics",
    "panels",
    "metadata",
)


@dataclass(frozen=True)
class RunResult:
    """Small handle to one completed analysis run."""

    profile: ProfileDefaults
    output_directory: str
    run_config: dict[str, Any]


def _clear_runtime_modules() -> None:
    """Discard run-specific module state before starting a fresh analysis."""
    for name in reversed(MODULE_ORDER):
        sys.modules.pop(f"ogvhd_prediction.{name}", None)


def run_pipeline(profile: ProfileName | ProfileDefaults) -> RunResult:
    """Run all modeling stages in their prespecified order."""
    selected = profile if isinstance(profile, ProfileDefaults) else get_profile(profile)
    previous_profile = os.environ.get("OGVHD_PROFILE")
    os.environ["OGVHD_PROFILE"] = selected.key
    _clear_runtime_modules()

    try:
        modules = {}
        for name in MODULE_ORDER:
            module = importlib.import_module(f"ogvhd_prediction.{name}")
            modules[name] = module
            if name == "config":
                module.initialize_runtime()
            elif name in EXECUTION_STAGES:
                module.run_stage()

        return RunResult(
            profile=selected,
            output_directory=str(modules["config"].OUTPUT_DIR),
            run_config=modules["metadata"].run_config,
        )
    except Exception:
        _clear_runtime_modules()
        raise
    finally:
        if previous_profile is None:
            os.environ.pop("OGVHD_PROFILE", None)
        else:
            os.environ["OGVHD_PROFILE"] = previous_profile
