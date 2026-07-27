# Ocular GVHD Prediction

## Study Objective

Develop and internally validate pretransplant, D+30, D+60, and D+90 XGBoost models for any-stage and severe ocular graft-versus-host disease after allogeneic hematopoietic stem-cell transplantation, with acute/chronic component outcomes and a secondary prophylaxis-inclusive analysis. This repository contains modeling code and fully synthetic sample data only; clinical data and manuscript-generation files are excluded.

## Folder structure

- `notebooks/`: concise notebook versions of the primary and prophylaxis-inclusive analyses.
- `src/`: executable Python exports that match the notebook code.
- `data/sample.xlsx`: 24-row synthetic workbook for schema inspection only.
- `tests/`: release-integrity checks.

## Environment

Use Python 3.12.3 and install the pinned dependencies with `python -m pip install -r requirements.txt`.

## How to run

Keep `dataset_v2.xlsx` and `frozen_test_split_manifest.csv` outside this repository, set `XGB_V2_DATA_PATH` and `XGB_V2_FROZEN_TEST_SPLIT_MANIFEST` to their local paths, then run `python src/xgboost_v2.py` or `python src/xgboost_v2_prophylaxis.py`.
