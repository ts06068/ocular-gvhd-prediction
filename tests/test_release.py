import ast
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_readme_is_concise(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        headings = [line for line in readme.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings,
            [
                "## Study Objective",
                "## Folder structure",
                "## Environment",
                "## How to run",
            ],
        )
        self.assertLessEqual(len(readme.splitlines()), 24)

    def test_notebook_structure_and_outputs(self):
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(notebook["cells"]), 22)
            for index in range(0, len(notebook["cells"]), 2):
                markdown = notebook["cells"][index]
                code = notebook["cells"][index + 1]
                self.assertEqual(markdown["cell_type"], "markdown")
                self.assertEqual(code["cell_type"], "code")
                nonempty = [
                    line.strip()
                    for line in "".join(markdown["source"]).splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(nonempty), 2)
                self.assertTrue(nonempty[0].startswith("#"))
                self.assertEqual(code.get("outputs", []), [])
                self.assertIsNone(code.get("execution_count"))

    def test_python_sources_compile(self):
        for path in sorted((ROOT / "src").glob("*.py")):
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            self.assertIn(
                'RUN_BOOTSTRAP_CI = os.getenv("XGB_V2_RUN_BOOTSTRAP_CI", "0") == "1"',
                source,
            )
            self.assertIn(
                'SPLIT_SEED_SEARCH_ENABLED = os.getenv(\n'
                '    "XGB_V2_SPLIT_SEED_SEARCH_ENABLED",\n'
                '    "0",\n'
                ') == "1"',
                source,
            )

    def test_notebook_and_python_sources_match(self):
        for notebook_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            notebook_code = "\n\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )
            script_path = ROOT / "src" / f"{notebook_path.stem}.py"
            notebook_ast = ast.dump(ast.parse(notebook_code), include_attributes=False)
            script_ast = ast.dump(
                ast.parse(script_path.read_text(encoding="utf-8")),
                include_attributes=False,
            )
            self.assertEqual(notebook_ast, script_ast)

    def test_sample_is_synthetic_schema_only(self):
        path = ROOT / "data" / "sample.xlsx"
        sample = pd.read_excel(path, sheet_name="Sheet1")
        self.assertEqual(sample.shape, (24, 93))
        self.assertTrue(sample["Patient ID"].astype(str).str.startswith("SYN-").all())
        self.assertTrue(sample["Research Patient ID"].astype(str).str.startswith("SYN-R-").all())
        self.assertTrue((pd.to_datetime(sample["Transplant Date"]).dt.year >= 2035).all())


if __name__ == "__main__":
    unittest.main()
