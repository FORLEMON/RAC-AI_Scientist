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


# Previously recorded successful runtime and unchanged official source pins.
RUNTIME = "4e29d772759b560a6dd37c790d816a780eeedf155560b93f40702a846a4f5a26"
OFFICIAL = "e02d651fa3579b0fb878130deef5be6dff10cd40977e5df0fc6099e9b9326d6b"


class AutoRuntimePinTests(unittest.TestCase):
    def test_packaged_runtime_and_official_checkout_use_their_respective_pins(self):
        lock = json.loads((Path(__file__).parents[1] / "upstream.lock.json").read_text())
        spec = lock["upstreams"]["auto_research_claw"]
        self.assertEqual(spec["tree_sha256"], OFFICIAL)
        for packaged, actual in ((True, RUNTIME), (False, OFFICIAL), (False, RUNTIME)):
            with self.subTest(packaged=packaged, actual=actual), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                (root / "upstream.lock.json").write_text(json.dumps(lock))
                manifest = root / "configs/hosts/auto_research_claw.json"
                manifest.parent.mkdir(parents=True)
                manifest.write_text("{}")
                task = root / "task"
                task.mkdir()
                (task / "task_info.json").write_text(json.dumps({"task": "fake objective", "data": []}))
                upstream = Path("/opt/host").resolve() if packaged else root / "official"
                args = SimpleNamespace(project_root=str(root), task_dir=str(task), run_root="runs",
                    model="FAKE", max_cost_usd=1, max_input_tokens=10, max_output_tokens=10,
                    max_agent_calls=1, max_wall_seconds=10, max_hops=1, host="auto_research_claw",
                    upstream=str(upstream), episode_id="fake", condition="N0", seed=0, review_score_threshold=8)
                bridge = SimpleNamespace(configure_condition=lambda value: None, initialize_native=lambda **kwargs: None)
                with patch.dict(os.environ, {"AGENT_API_KEY": "FAKE-ONLY", "RAC_HOST_ROOT": "/opt/host"}), \
                     patch("rac_ai_scientist.cli._selected_upstream", return_value=upstream), \
                     patch("rac_ai_scientist.cli.make_bridge", return_value=bridge), \
                     patch("rac_ai_scientist.cli.materialize_rcb_workspace", side_effect=lambda task, workspace: workspace.mkdir()), \
                     patch("rac_ai_scientist.cli.tree_hash", return_value=(actual, 1, 1)), \
                     patch("rac_ai_scientist.cli._execute_episode", return_value=(SimpleNamespace(status="completed", hops=1, reason="fake"), None)) as execute, \
                     contextlib.redirect_stdout(io.StringIO()):
                    if not packaged and actual == RUNTIME:
                        with self.assertRaisesRegex(ValueError, "checkout does not match"):
                            _run_one(args)
                        execute.assert_not_called()
                    else:
                        self.assertEqual(_run_one(args), 0)
                metadata = json.loads((root / "runs/fake/episode.json").read_text())
                self.assertEqual(metadata["upstream"]["expected_tree_sha256"], RUNTIME if packaged else OFFICIAL)
        self.assertEqual(spec["revision"], "c4125fe47d57da2c65ed2108efbacc2cd170b3aa")
        self.assertEqual(spec["url"], "https://github.com/JY0xLU/AutoResearchClaw.git")


if __name__ == "__main__":
    unittest.main()
