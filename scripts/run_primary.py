"""Compatibility entry point for the primary analysis."""

from ogvhd_prediction import run_analysis


# Keep this entry point thin; the shared pipeline owns the analysis.
if __name__ == "__main__":
    run_analysis("primary")
