# Ocular GVHD Prediction

## Study Objective

Develop and internally validate pretransplant, D+30, D+60, and D+90 XGBoost models for any-stage and severe ocular graft-versus-host disease after allogeneic hematopoietic stem-cell transplantation, with acute/chronic component outcomes and a secondary prophylaxis-inclusive analysis. This repository contains modeling code and fully synthetic sample data only; clinical data and manuscript-generation files are excluded.

## Folder structure

- `src/ogvhd_prediction/`: shared, role-based modeling modules.
- `scripts/`: primary and prophylaxis-inclusive experiment entry points.
- `notebooks/`: thin interactive wrappers around the same experiment runner.
- `data/sample.xlsx`: 24-row synthetic workbook for schema inspection only.
- `tests/`: release-integrity checks.

## Environment

Use Python 3.12.3, then run `python -m pip install -r requirements.txt` and `python -m pip install -e .`.

## How to run

Keep `dataset_v2.xlsx` and `frozen_test_split_manifest.csv` outside this repository, set `XGB_V2_DATA_PATH` and `XGB_V2_FROZEN_TEST_SPLIT_MANIFEST` to their local paths, then run `python scripts/run_primary.py` or `python scripts/run_prophylaxis.py`.
