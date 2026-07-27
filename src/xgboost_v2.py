"""Compatibility entry point for the primary analysis."""

from ogvhd_prediction import run_analysis


# Retain the original command path while delegating execution to the modular pipeline.
if __name__ == "__main__":
    run_analysis("primary")
