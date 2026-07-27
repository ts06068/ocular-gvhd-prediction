"""Compatibility entry point for the prophylaxis-inclusive analysis."""

from ogvhd_prediction import run_analysis


# Select the prophylaxis profile here; keep its modeling logic in the shared pipeline.
if __name__ == "__main__":
    run_analysis("prophylaxis")
