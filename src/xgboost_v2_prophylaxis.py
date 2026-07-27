"""Compatibility entry point for the prophylaxis-inclusive analysis."""

from ogvhd_prediction import run_analysis


# Preserve the legacy prophylaxis command; route it through the shared pipeline.
if __name__ == "__main__":
    run_analysis("prophylaxis")
