# Ocular GVHD Prediction

## Study Objective

Develop and internally validate pretransplant, D+30, D+60, and D+90 XGBoost models for any-stage and severe ocular graft-versus-host disease after allogeneic hematopoietic stem-cell transplantation, with acute/chronic component outcomes and a secondary prophylaxis-inclusive analysis. This repository contains modeling code and fully synthetic sample data only; clinical data and manuscript-generation files are excluded.

## Folder structure

- `src/ogvhd_prediction/`: shared, role-based modeling modules.
- `scripts/`: primary and prophylaxis-inclusive experiment entry points.
- `notebooks/`: thin interactive wrappers around the same experiment runner.
- `data/sample.xlsx`: 24-row synthetic workbook for schema inspection only.
- `tests/`: release-integrity checks.

## Code structure

`scripts/run_primary.py` and `scripts/run_prophylaxis.py` are thin entry points to the same pipeline.

- `src/ogvhd_prediction/runner.py`: selects the requested analysis profile without loading cohort data.
- `src/ogvhd_prediction/pipeline.py`: imports and executes the modeling stages in a fixed order.
- `src/ogvhd_prediction/profiles.py` and `config.py`: preserve profile settings and resolve runtime paths.
- `src/ogvhd_prediction/data.py`, `targets.py`, and `splitting.py`: build landmark outcomes and patient-level partitions.
- `src/ogvhd_prediction/preprocessing.py`, `tuning.py`, and `explainability.py`: preprocess, tune XGBoost, and rank features.
- `src/ogvhd_prediction/evaluation.py`, `finalization.py`, `reporting.py`, `diagnostics.py`, and `panels.py`: evaluate holdouts and save outputs.

## Environment

Use Python 3.12.3, then run `python -m pip install -r requirements.txt` and `python -m pip install -e .`.

## How to run

Keep `dataset_v2.xlsx` and `frozen_test_split_manifest.csv` outside this repository, set `XGB_V2_DATA_PATH` and `XGB_V2_FROZEN_TEST_SPLIT_MANIFEST` to their local paths, then run `python scripts/run_primary.py` or `python scripts/run_prophylaxis.py`.
