import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rac_ai_scientist.cli import _run_one


RUNTIME = "d38fddf6197d888a951de7c09165dbda0ecdd9a92e569225ae292f6d95002d06"
OFFICIAL = "a952224ea0e3216f9c6cff1c30444afef04a9d8bb0bbb206c60df2125c8ec0c6"
SNAPSHOT = "48de1aef80d0fe39fe1148441e992d3b22a20f49678801d62ca005990129c528"


class AIRuntimePinTests(unittest.TestCase):
    def test_packaged_runtime_is_distinct_from_official_and_archive_sources(self):
        lock = json.loads((Path(__file__).parents[1] / "upstream.lock.json").read_text())
        spec = lock["upstreams"]["ai_researcher"]
        self.assertEqual(spec["tree_sha256"], OFFICIAL)
        self.assertEqual(spec["snapshot_tree_sha256"], SNAPSHOT)
        self.assertEqual(spec["revision"], "e9c3294cee5dc85ce70cb1e30b7a2f86f69d0502")
        self.assertEqual(spec["url"], "https://github.com/HKUDS/AI-Researcher.git")
        cases = (("packaged", RUNTIME, True), ("packaged", OFFICIAL, False),
                 ("packaged", "wrong-runtime", False), ("official", OFFICIAL, True),
                 ("official", RUNTIME, False), ("snapshot", SNAPSHOT, True),
                 ("snapshot", RUNTIME, False))
        for kind, actual, accepted in cases:
            with self.subTest(kind=kind, actual=actual), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                (root / "upstream.lock.json").write_text(json.dumps(lock))
                manifest = root / "configs/hosts/ai_researcher.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text("{}")
                task = root / "task"
                task.mkdir()
                (task / "task_info.json").write_text(json.dumps({"task": "fake objective", "data": []}))
                upstream = (Path("/opt/host").resolve() if kind == "packaged" else
                            root / (spec["local_snapshot"] if kind == "snapshot" else "official"))
                args = SimpleNamespace(project_root=str(root), task_dir=str(task), run_root="runs",
                    model="FAKE", max_cost_usd=1, max_input_tokens=10, max_output_tokens=10,
                    max_agent_calls=1, max_wall_seconds=10, max_hops=1, host="ai_researcher",
                    upstream=str(upstream), episode_id="fake", condition="N0", seed=0, review_score_threshold=8)
                bridge = SimpleNamespace(configure_condition=lambda value: None, initialize_native=lambda **kwargs: None)
                with patch.dict(os.environ, {"AGENT_API_KEY": "FAKE-ONLY", "RAC_HOST_ROOT": "/opt/host"}), \
                     patch("rac_ai_scientist.cli._selected_upstream", return_value=upstream), \
                     patch("rac_ai_scientist.cli.make_bridge", return_value=bridge), \
                     patch("rac_ai_scientist.cli.materialize_rcb_workspace", side_effect=lambda task, workspace: workspace.mkdir()), \
                     patch("rac_ai_scientist.cli.tree_hash", return_value=(actual, 1, 1)), \
                     patch("rac_ai_scientist.cli._execute_episode", return_value=(SimpleNamespace(status="completed", hops=1, reason="fake"), None)) as execute, \
                     contextlib.redirect_stdout(io.StringIO()):
                    if accepted:
                        self.assertEqual(_run_one(args), 0)
                        execute.assert_called_once()
                    else:
                        with self.assertRaisesRegex(ValueError, "checkout does not match"):
                            _run_one(args)
                        execute.assert_not_called()
                metadata = json.loads((root / "runs/fake/episode.json").read_text())
                expected = {"packaged": RUNTIME, "official": OFFICIAL, "snapshot": SNAPSHOT}[kind]
                self.assertEqual(metadata["upstream"]["expected_tree_sha256"], expected)


if __name__ == "__main__":
    unittest.main()
