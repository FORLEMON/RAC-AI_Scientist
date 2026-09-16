import json
import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.hosts.data_to_paper import _write_data_descriptions


class DataToPaperDescriptionTests(unittest.TestCase):
    def test_materializes_general_and_per_file_descriptions(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            (workspace / "task.json").write_text(json.dumps({
                "data": [{
                    "path": "./data/records.csv",
                    "description": "Annual public records.",
                }],
            }), encoding="utf-8")

            _write_data_descriptions(
                workspace,
                "Analyze the supplied records.",
                ["data/records.csv"],
            )

            self.assertEqual(
                (workspace / "general_description.txt").read_text(encoding="utf-8"),
                "Analyze the supplied records.",
            )
            self.assertEqual(
                (workspace / "records.csv.description.txt").read_text(encoding="utf-8"),
                "Annual public records.",
            )

    def test_rejects_duplicate_data_basenames(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            (workspace / "task.json").write_text(json.dumps({
                "data": [
                    {"path": "data/a/records.csv", "description": "A"},
                    {"path": "data/b/records.csv", "description": "B"},
                ],
            }), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unique data basenames"):
                _write_data_descriptions(
                    workspace,
                    "Analyze the supplied records.",
                    ["data/a/records.csv", "data/b/records.csv"],
                )


if __name__ == "__main__":
    unittest.main()
