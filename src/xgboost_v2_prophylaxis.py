"""Compatibility entry point for the prophylaxis-inclusive analysis."""

from ogvhd_prediction import run_analysis


# Delegate to the shared engine instead of maintaining a second prophylaxis workflow.
if __name__ == "__main__":
    run_analysis("prophylaxis")
