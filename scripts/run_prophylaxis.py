"""Compatibility entry point for the prophylaxis-inclusive analysis."""

from ogvhd_prediction import run_analysis


# This entry point changes only the profile; all modeling logic remains shared.
if __name__ == "__main__":
    run_analysis("prophylaxis")
