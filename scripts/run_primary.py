"""Compatibility entry point for the primary analysis."""

from ogvhd_prediction import run_analysis


# Keep the primary CLI thin so scripts and notebooks execute the same pipeline.
if __name__ == "__main__":
    run_analysis("primary")
