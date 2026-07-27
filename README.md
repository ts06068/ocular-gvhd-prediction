# Ocular GVHD Prediction

## Study Objective

Build and internally validate XGBoost models for any-stage and severe ocular graft-versus-host disease before transplant and at D+30, D+60, and D+90 after allogeneic hematopoietic stem-cell transplantation. Evaluate acute and chronic component outcomes and a prophylaxis-inclusive secondary analysis. This repository contains modeling code and synthetic sample data. Clinical data and manuscript code do not belong here.

## Folder structure

- `src/ogvhd_prediction/`: modeling modules, split by responsibility.
- `scripts/`: command-line entry points; no model logic.
- `notebooks/`: interactive wrappers; no model logic.
- `data/sample.xlsx`: 24 synthetic rows for schema checks only. Never train on it.
- `tests/`: structural, import-safety, and release checks.

## Code structure

Both entry points call one pipeline. Profiles change configuration, not control flow.

- `scripts/run_primary.py` and `scripts/run_prophylaxis.py`: choose a profile and start the run.
- `src/ogvhd_prediction/runner.py` and `src/ogvhd_prediction/pipeline.py`: resolve the profile and execute the fixed stage order.
- `src/ogvhd_prediction/profiles.py` and `src/ogvhd_prediction/config.py`: freeze profile settings and resolve runtime paths.
- `src/ogvhd_prediction/data.py`, `src/ogvhd_prediction/targets.py`, and `src/ogvhd_prediction/splitting.py`: build landmark outcomes and patient-level splits.
- `src/ogvhd_prediction/preprocessing.py`, `src/ogvhd_prediction/tuning.py`, and `src/ogvhd_prediction/explainability.py`: fit preprocessing inside folds, tune XGBoost, and rank features.
- `src/ogvhd_prediction/evaluation.py`, `src/ogvhd_prediction/finalization.py`, `src/ogvhd_prediction/reporting.py`, `src/ogvhd_prediction/diagnostics.py`, and `src/ogvhd_prediction/panels.py`: evaluate the frozen holdout and write final artifacts.

## Environment

Use Python 3.12.3. Install pinned dependencies with `python -m pip install -r requirements.txt`. Install the package with `python -m pip install -e .`.

## How to run

```powershell
$env:XGB_V2_DATA_PATH = "D:\data\dataset_v2.xlsx"
$env:XGB_V2_FROZEN_TEST_SPLIT_MANIFEST = "D:\data\frozen_test_split_manifest.csv"
python scripts/run_primary.py
python scripts/run_prophylaxis.py  # Optional secondary analysis
```
