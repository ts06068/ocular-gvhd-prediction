import ast
import builtins
import hashlib
import importlib
import io
import json
import os
import sys
import tempfile
import tokenize
import types
import unittest
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "ogvhd_prediction"
LEGACY_MANIFEST = ROOT / "tests" / "legacy_ast_manifest.json"


@contextmanager
def isolated_package_imports():
    """Expose the source tree while restoring import state after the check."""
    prefix = "ogvhd_prediction"
    previous_path = list(sys.path)
    previous_bytecode_setting = sys.dont_write_bytecode
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == prefix or name.startswith(f"{prefix}.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(SRC))
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_bytecode_setting
        for name in list(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


class ReleaseTests(unittest.TestCase):
    def test_readme_is_concise(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        headings = [line for line in readme.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings,
            [
                "## Study Objective",
                "## Folder structure",
                "## Code structure",
                "## Environment",
                "## How to run",
            ],
        )
        folder_structure = readme.split("## Folder structure\n\n", maxsplit=1)[1].split(
            "\n\n## Code structure", maxsplit=1
        )[0]
        self.assertTrue(folder_structure.startswith("```text\n.\n"))
        self.assertTrue(folder_structure.rstrip().endswith("```"))
        for path in (
            ".github/workflows/ci.yml",
            "data/",
            "notebooks/",
            "scripts/",
            "src/",
            "tests/",
            "pyproject.toml",
            "requirements.txt",
        ):
            self.assertIn(path, folder_structure)

        for path in sorted(PACKAGE.glob("*.py")):
            self.assertIn(path.name, folder_structure)

        how_to_run = readme.split("## How to run\n\n", maxsplit=1)[1]
        self.assertTrue(how_to_run.startswith("```powershell\n"))
        self.assertTrue(how_to_run.rstrip().endswith("```"))
        self.assertIn("$env:XGB_V2_DATA_PATH", how_to_run)
        self.assertIn("$env:XGB_V2_FROZEN_TEST_SPLIT_MANIFEST", how_to_run)
        self.assertIn("python scripts/run_primary.py", how_to_run)
        self.assertIn("python scripts/run_prophylaxis.py", how_to_run)
        self.assertLessEqual(len(readme.splitlines()), 90)

    def test_notebooks_are_thin_wrappers(self):
        paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
        self.assertEqual(
            [path.name for path in paths],
            ["xgboost_v2.ipynb", "xgboost_v2_prophylaxis.ipynb"],
        )

        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = notebook["cells"]
            code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
            self.assertEqual(len(code_cells), 2, path.name)
            self.assertEqual(len(cells), 4, path.name)

            total_nonblank_code_lines = 0
            for index in range(0, len(cells), 2):
                markdown = cells[index]
                code = cells[index + 1]
                self.assertEqual(markdown["cell_type"], "markdown", path.name)
                self.assertEqual(code["cell_type"], "code", path.name)

                markdown_lines = "".join(markdown["source"]).splitlines()
                self.assertEqual(len(markdown_lines), 2, path.name)
                self.assertTrue(markdown_lines[0].strip().startswith("#"), path.name)
                self.assertTrue(markdown_lines[1].strip(), path.name)
                self.assertFalse(markdown_lines[1].strip().startswith("#"), path.name)

                code_source = "".join(code["source"])
                tree = ast.parse(code_source, filename=str(path))
                forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                self.assertFalse(
                    any(isinstance(node, forbidden) for node in ast.walk(tree)),
                    f"Definitions belong in the package, not {path.name}",
                )
                total_nonblank_code_lines += sum(
                    bool(line.strip()) for line in code_source.splitlines()
                )
                self.assertEqual(code.get("outputs", []), [], path.name)
                self.assertIsNone(code.get("execution_count"), path.name)

            self.assertLessEqual(total_nonblank_code_lines, 10, path.name)

    def test_every_python_file_has_a_reader_comment(self):
        python_files = sorted(
            path
            for folder in (SRC, ROOT / "scripts", ROOT / "tests")
            for path in folder.rglob("*.py")
        )
        self.assertTrue(python_files)

        missing_comments = []
        for path in python_files:
            source = path.read_text(encoding="utf-8")
            comments = [
                token.string.removeprefix("#").strip()
                for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT
            ]
            reader_comments = [
                comment
                for comment in comments
                if len(comment.split()) >= 6
                and not comment.casefold().startswith(("noqa", "ruff", "type:"))
            ]
            if not reader_comments:
                missing_comments.append(str(path.relative_to(ROOT)))

        self.assertFalse(
            missing_comments,
            f"Python files missing a reader-oriented comment: {missing_comments}",
        )

    # CI checks source integrity; it must not load clinical data or run models.
    def test_ci_workflow_lints_tests_and_builds_without_data_access(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        required_fragments = (
            "push:",
            "pull_request:",
            "workflow_dispatch:",
            "contents: read",
            "ubuntu-24.04",
            "actions/checkout@v7",
            "persist-credentials: false",
            "actions/setup-python@v7",
            'python-version: "3.12.3"',
            "python -m pip install -r requirements.txt",
            "python -m pip install --no-deps -e .",
            "ruff check --no-cache src scripts tests",
            "ruff format --no-cache --check src scripts tests",
            "python -B -m unittest discover -s tests -v",
            "python -m build",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, workflow)

        for analysis_entry_point in ("run_primary.py", "run_prophylaxis.py"):
            self.assertNotIn(analysis_entry_point, workflow)

        job_level_configuration = workflow.split("    steps:", maxsplit=1)[0]
        self.assertNotIn("${{ runner.", job_level_configuration)

    def test_every_python_source_compiles_without_creating_pycache(self):
        paths = sorted(ROOT.rglob("*.py"))
        self.assertTrue(paths)
        for path in paths:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec", dont_inherit=True)

    def test_legacy_algorithm_ast_is_preserved(self):
        expected_manifest = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(len(expected_manifest), 184)
        expected = Counter(
            (key.rsplit(":", 1)[0], digest) for key, digest in expected_manifest.items()
        )

        actual = Counter()
        for path in sorted(PACKAGE.glob("*.py")):
            if path.name in {"profiles.py", "runner.py"}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    continue
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                digest = hashlib.sha256(
                    ast.dump(node, include_attributes=False).encode("utf-8")
                ).hexdigest()
                actual[(f"{kind}:{node.name}", digest)] += 1

        missing = expected - actual
        self.assertFalse(
            missing,
            f"Legacy definitions missing or changed: {missing}",
        )

    def test_profile_defaults_preserve_exact_analysis_deltas(self):
        expected = {
            "key": ("primary", "prophylaxis"),
            "model_profile_name": ("v2", "v2_prophylaxis"),
            "default_output_dir": (
                "outputs_xgboost_v2",
                "outputs_xgboost_v2_prophylaxis",
            ),
            "development_split_seed_overrides": (
                (
                    ("oGVHD_any", 60),
                    ("oGVHD_severe", 300),
                    ("cGVHD_any", 500),
                    ("cGVHD_severe", 1243),
                ),
                (
                    ("oGVHD_any", 2503),
                    ("oGVHD_severe", 2024),
                    ("cGVHD_any", 2503),
                    ("cGVHD_severe", 1243),
                ),
            ),
            "optuna_trial_overrides": (
                (
                    ("oGVHD_any|Preoperative", 10),
                    ("oGVHD_any|Post transplant D+30", 10),
                    ("oGVHD_any|Post transplant D+60", 10),
                    ("oGVHD_any|Post transplant D+90", 10),
                    ("oGVHD_severe|Preoperative", 100),
                    ("cGVHD_any|Preoperative", 30),
                    ("cGVHD_any|Post transplant D+30", 100),
                    ("cGVHD_any|Post transplant D+60", 100),
                    ("cGVHD_any|Post transplant D+90", 100),
                    ("cGVHD_severe|Preoperative", 100),
                ),
                (
                    ("oGVHD_any|Preoperative", 10),
                    ("oGVHD_any|Post transplant D+30", 10),
                    ("oGVHD_any|Post transplant D+60", 10),
                    ("oGVHD_any|Post transplant D+90", 10),
                    ("oGVHD_severe|Preoperative", 100),
                    ("cGVHD_any|Preoperative", 10),
                    ("cGVHD_severe|Preoperative", 100),
                ),
            ),
            "optuna_objective_metric": ("auroc", "auroc_auprc_lift"),
            "validation_candidate_trial_overrides": (
                (
                    ("oGVHD_any|Preoperative", 30),
                    ("oGVHD_any|Post transplant D+30", 30),
                    ("oGVHD_any|Post transplant D+60", 30),
                    ("oGVHD_any|Post transplant D+90", 30),
                    ("oGVHD_severe|Preoperative", 30),
                    ("cGVHD_any|Preoperative", 30),
                    ("cGVHD_any|Post transplant D+30", 30),
                    ("cGVHD_any|Post transplant D+60", 30),
                    ("cGVHD_any|Post transplant D+90", 30),
                    ("cGVHD_severe|Preoperative", 30),
                ),
                (),
            ),
            "ensemble_top_k_candidates": ((1, 2, 3, 4, 5, 8, 10), (1, 2, 3, 4, 5)),
            "validation_selection_prior_min_class_count": (30, 10),
            "performance_rescue_analyses": (
                (
                    "oGVHD_severe|Preoperative",
                    "oGVHD_severe|Post transplant D+30",
                    "oGVHD_severe|Post transplant D+60",
                    "oGVHD_severe|Post transplant D+90",
                    "cGVHD_any|Post transplant D+30",
                    "cGVHD_any|Post transplant D+60",
                    "cGVHD_any|Post transplant D+90",
                    "cGVHD_severe|Preoperative",
                    "cGVHD_severe|Post transplant D+90",
                ),
                (
                    "oGVHD_severe|Preoperative",
                    "oGVHD_severe|Post transplant D+30",
                    "oGVHD_severe|Post transplant D+60",
                ),
            ),
            "validation_model_selection_metric": ("auroc", "auroc_auprc_lift"),
            "dual_discrimination_selection": (False, True),
            "simplified_optuna_trial_overrides": (
                (("oGVHD_any|Post transplant D+60", 50),),
                (),
            ),
            "simplified_validation_candidate_trial_overrides": (
                (("oGVHD_any|Post transplant D+60", 10),),
                (),
            ),
            "include_prophylaxis": (False, True),
        }

        with isolated_package_imports():
            profiles = importlib.import_module("ogvhd_prediction.profiles")
            primary = profiles.PRIMARY
            prophylaxis = profiles.PROPHYLAXIS
            actual = {
                name: (getattr(primary, name), getattr(prophylaxis, name))
                for name in primary.__dataclass_fields__
                if getattr(primary, name) != getattr(prophylaxis, name)
            }

        self.assertEqual(actual, expected)

    def test_package_public_imports_are_data_and_side_effect_free(self):
        with tempfile.TemporaryDirectory() as temp_dir, isolated_package_imports():
            previous_cwd = Path.cwd()
            os.chdir(temp_dir)
            try:
                before = list(Path(temp_dir).iterdir())
                package = importlib.import_module("ogvhd_prediction")
                profiles = importlib.import_module("ogvhd_prediction.profiles")
                runner = importlib.import_module("ogvhd_prediction.runner")
                after = list(Path(temp_dir).iterdir())
                loaded = {
                    name
                    for name in sys.modules
                    if name == "ogvhd_prediction"
                    or name.startswith("ogvhd_prediction.")
                }
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(before, after)
            self.assertEqual(
                loaded,
                {
                    "ogvhd_prediction",
                    "ogvhd_prediction.profiles",
                    "ogvhd_prediction.runner",
                },
            )
            self.assertTrue(callable(package.run_analysis))
            self.assertTrue(callable(profiles.get_profile))
            self.assertTrue(callable(runner.run_analysis))

    # Imports must not read clinical data, write files, or train models.
    def test_internal_module_imports_do_not_read_write_or_train(self):
        import cloudpickle
        import joblib
        import matplotlib.figure
        import matplotlib.pyplot
        import optuna
        import sklearn.model_selection
        import sklearn.pipeline
        import xgboost

        def forbidden(operation):
            return mock.Mock(
                side_effect=AssertionError(
                    f"Package import attempted forbidden operation: {operation}"
                )
            )

        real_open = builtins.open

        def read_only_open(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                raise AssertionError(
                    f"Package import attempted to open an output file: {file}"
                )
            return real_open(file, mode, *args, **kwargs)

        module_names = sorted(
            path.stem for path in PACKAGE.glob("*.py") if path.name != "__init__.py"
        )
        self.assertTrue(module_names)

        with tempfile.TemporaryDirectory() as temp_dir, isolated_package_imports():
            temp_path = Path(temp_dir)
            environment = {
                "OGVHD_PROFILE": "primary",
                "XGB_V2_DATA_PATH": str(temp_path / "must-not-be-read.xlsx"),
                "XGB_V2_FROZEN_TEST_SPLIT_MANIFEST": str(
                    temp_path / "must-not-be-read.csv"
                ),
                "XGB_V2_OUTPUT_DIR": str(temp_path / "must-not-be-created"),
            }
            previous_cwd = Path.cwd()
            os.chdir(temp_path)
            try:
                before = list(temp_path.iterdir())
                with (
                    mock.patch.dict(os.environ, environment, clear=False),
                    mock.patch.object(Path, "read_bytes", forbidden("Path.read_bytes")),
                    mock.patch.object(
                        Path, "write_bytes", forbidden("Path.write_bytes")
                    ),
                    mock.patch.object(Path, "write_text", forbidden("Path.write_text")),
                    mock.patch.object(Path, "mkdir", forbidden("Path.mkdir")),
                    mock.patch.object(pd, "read_excel", forbidden("pandas.read_excel")),
                    mock.patch.object(pd, "read_csv", forbidden("pandas.read_csv")),
                    mock.patch.object(
                        pd.DataFrame, "to_csv", forbidden("DataFrame.to_csv")
                    ),
                    mock.patch.object(
                        matplotlib.figure.Figure,
                        "savefig",
                        forbidden("Figure.savefig"),
                    ),
                    mock.patch.object(
                        sklearn.pipeline.Pipeline,
                        "fit",
                        forbidden("Pipeline.fit"),
                    ),
                    mock.patch.object(
                        xgboost.XGBClassifier,
                        "fit",
                        forbidden("XGBClassifier.fit"),
                    ),
                    mock.patch.object(
                        optuna.study.Study,
                        "optimize",
                        forbidden("Study.optimize"),
                    ),
                    mock.patch.object(
                        joblib.Parallel,
                        "__call__",
                        forbidden("Parallel.__call__"),
                    ),
                    mock.patch.object(
                        sklearn.model_selection,
                        "cross_validate",
                        forbidden("cross_validate"),
                    ),
                    mock.patch.object(
                        cloudpickle, "dump", forbidden("cloudpickle.dump")
                    ),
                    mock.patch.object(builtins, "open", side_effect=read_only_open),
                ):
                    importlib.import_module("ogvhd_prediction")
                    for module_name in module_names:
                        importlib.import_module(f"ogvhd_prediction.{module_name}")
                after = list(temp_path.iterdir())
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(before, after)

    def test_runner_calls_pipeline_for_each_same_profile_invocation(self):
        first_result = object()
        second_result = object()
        run_pipeline = mock.Mock(side_effect=[first_result, second_result])
        fake_pipeline = types.ModuleType("ogvhd_prediction.pipeline")
        fake_pipeline.run_pipeline = run_pipeline
        fake_metadata = types.ModuleType("ogvhd_prediction.metadata")

        with isolated_package_imports():
            sys.modules[fake_pipeline.__name__] = fake_pipeline
            sys.modules[fake_metadata.__name__] = fake_metadata
            runner = importlib.import_module("ogvhd_prediction.runner")
            with mock.patch.dict(os.environ, {"OGVHD_PROFILE": "primary"}, clear=False):
                actual_first = runner.run_analysis("primary")
                actual_second = runner.run_analysis("primary")

        self.assertIs(actual_first, first_result)
        self.assertIs(actual_second, second_result)
        self.assertEqual(run_pipeline.call_count, 2)
        for call in run_pipeline.call_args_list:
            self.assertEqual(len(call.args), 1)
            self.assertFalse(call.kwargs)
            self.assertEqual(getattr(call.args[0], "key", call.args[0]), "primary")

    def test_pipeline_runs_named_stages_in_order(self):
        with isolated_package_imports():
            pipeline = importlib.import_module("ogvhd_prediction.pipeline")
            profiles = importlib.import_module("ogvhd_prediction.profiles")
            events = []
            run_config = {"model_profile": "v2"}
            modules = {}

            for name in pipeline.MODULE_ORDER:
                module = types.SimpleNamespace()
                if name == "config":
                    module.OUTPUT_DIR = Path("output")
                    module.initialize_runtime = lambda stage=name: events.append(
                        f"run:{stage}"
                    )
                elif name in pipeline.EXECUTION_STAGES:
                    module.run_stage = lambda stage=name: events.append(f"run:{stage}")
                if name == "metadata":
                    module.run_config = run_config
                modules[name] = module

            def import_stage(full_name):
                name = full_name.rsplit(".", 1)[-1]
                events.append(f"import:{name}")
                return modules[name]

            with (
                mock.patch.object(
                    pipeline.importlib,
                    "import_module",
                    side_effect=import_stage,
                ),
                mock.patch.object(
                    pipeline,
                    "_clear_runtime_modules",
                    side_effect=lambda: events.append("clear"),
                ),
            ):
                result = pipeline.run_pipeline(profiles.PRIMARY)

        expected = ["clear"]
        for name in pipeline.MODULE_ORDER:
            expected.append(f"import:{name}")
            if name == "config" or name in pipeline.EXECUTION_STAGES:
                expected.append(f"run:{name}")
        self.assertEqual(events, expected)
        self.assertIs(result.profile, profiles.PRIMARY)
        self.assertEqual(result.output_directory, "output")
        self.assertIs(result.run_config, run_config)

    def test_package_modules_and_entry_points_are_small(self):
        modules = sorted(PACKAGE.glob("*.py"))
        self.assertTrue(modules)
        for path in modules:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(line_count, 1_500, path.name)

        entry_points = sorted(SRC.glob("*.py")) + sorted(
            (ROOT / "scripts").glob("*.py")
        )
        self.assertEqual(len(entry_points), 4)
        for path in entry_points:
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(line_count, 12, str(path.relative_to(ROOT)))

    def test_sample_is_synthetic_and_matches_the_model_schema(self):
        path = ROOT / "data" / "sample.xlsx"
        self.assertTrue(path.is_file())
        with pd.ExcelFile(path) as workbook:
            self.assertEqual(workbook.sheet_names, ["Sheet1"])
            sample = pd.read_excel(workbook, sheet_name="Sheet1")

        self.assertEqual(sample.shape, (24, 93))
        self.assertEqual(len(sample.columns), len(set(sample.columns)))

        required = {
            "Patient ID",
            "Research Patient ID",
            "Transplant Date",
            "Transplant Date raw",
            "Outcome transplant date",
            "Outcome match status",
            "Outcome label rule",
            "Age",
            "Male Sex",
            "Sex",
            "Diagnosis",
            "Graft source",
            "Myeloablating regimen MAC vs RIC",
            "Prophylaxis regimen",
            "Related",
            "ABO matching",
            "CD34",
            "Donor T cell number",
            "Death_3M",
        }
        for outcome in ("aGVHD", "cGVHD", "oGVHD"):
            required.update(
                {
                    f"{outcome} Stage",
                    f"Previous {outcome} Stage",
                    f"{outcome} onset date",
                    f"{outcome} max date",
                    f"{outcome} max ocular score",
                    f"{outcome} label source",
                }
            )
        for locus in ("A", "B", "C", "DR"):
            required.add(f"HLA-{locus} antigen matched")
            required.add(f"HLA-{locus} allele matched")
        for laboratory in (
            "PLT_count",
            "PDW",
            "PDW_pct",
            "AST_GOT",
            "Total_cholesterol",
            "CRP",
            "ESR",
            "Ferritin",
            "HDL_cholesterol",
            "Triglyceride",
            "LDL_cholesterol",
        ):
            required.update(
                {
                    f"{laboratory}_baseline",
                    f"{laboratory}+30",
                    f"{laboratory}+60",
                    f"{laboratory}+90",
                }
            )
        self.assertTrue(
            required.issubset(sample.columns), sorted(required - set(sample.columns))
        )

        self.assertTrue(sample["Patient ID"].astype(str).str.startswith("SYN-").all())
        self.assertTrue(
            sample["Research Patient ID"].astype(str).str.startswith("SYN-R-").all()
        )
        self.assertTrue(sample["DoB"].astype(str).str.startswith("SYNTHETIC-").all())
        self.assertTrue(
            (pd.to_datetime(sample["Transplant Date"]).dt.year >= 2035).all()
        )
        for outcome in ("aGVHD", "cGVHD", "oGVHD"):
            self.assertTrue(
                sample[f"{outcome} label source"]
                .astype(str)
                .str.casefold()
                .eq("synthetic")
                .all()
            )


if __name__ == "__main__":
    unittest.main()
