import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rac_ai_scientist.hosts.auto_research_claw import _benchmark_execution_topic


class AutoInputPathTests(unittest.TestCase):
    def test_manifest_input_remains_readable_from_native_project_cwd(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw) / "workspace with spaces"
            data = workspace / "data" / "measurements.dat"
            data.parent.mkdir(parents=True)
            data.write_text("1 2\n3 4\n", encoding="utf-8")
            (workspace / "task_info.json").write_text(json.dumps({"data": [{
                "path": "./data/measurements.dat", "description": "observations",
            }]}), encoding="utf-8")
            native_cwd = workspace / "auto_research_claw_native/stage-10/agent_sandbox/_project_1"
            native_cwd.mkdir(parents=True)
            topic = _benchmark_execution_topic(workspace, "Analyze the supplied observations")
            listed = next(line[2:].split(": ", 1)[0] for line in topic.splitlines() if line.startswith("- "))
            result = subprocess.run([sys.executable, "-c",
                "from pathlib import Path; import sys; print(Path(sys.argv[1]).read_text())", listed],
                cwd=native_cwd, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 2", result.stdout)
            self.assertEqual(Path(listed), data.resolve())
            self.assertEqual(data.read_text(encoding="utf-8"), "1 2\n3 4\n")

    def test_named_input_uses_the_same_absolute_path_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            (workspace / "task_info.json").write_text(json.dumps({"data": [{"name": "other.csv"}]}))
            topic = _benchmark_execution_topic(workspace, "Analyze")
            self.assertIn((workspace / "data/other.csv").resolve().as_posix(), topic)


if __name__ == "__main__":
    unittest.main()
