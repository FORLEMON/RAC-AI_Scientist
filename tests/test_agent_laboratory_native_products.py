import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.cli import _run_one, build_parser
from rac_ai_scientist.hosts.agent_laboratory import AgentLaboratoryBridge
from rac_ai_scientist.schemas import Budget


class RateLimitError(Exception):
    pass


class FakeWorkflow:
    def __init__(self, error=None):
        self.error = error
        self.phd = types.SimpleNamespace(
            lit_review="", plan="", dataset_code="", results_code="",
            exp_results="", interpretation="", report="",
        )

    def perform_research(self):
        self.phd.dataset_code = "data = [1, 2, 3]\n"
        self.phd.results_code = "print(sum(data))\n"
        self.phd.exp_results = "sum = 6"
        self.phd.interpretation = "The observed sum is 6."
        if self.error is not None:
            raise self.error
        self.phd.report = "# Report\n\nThe observed sum is 6."


def bridge_for(workspace, error=None):
    manifest = Path(__file__).resolve().parents[1] / "configs/hosts/agent_laboratory.json"
    bridge = AgentLaboratoryBridge(
        workspace, manifest, Budget(25, 1000, 1000, 10, 100, 10), "fake-model", "fake-only"
    )
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".rac").mkdir(exist_ok=True)
    (workspace / "report").mkdir(exist_ok=True)
    bridge.workspace = workspace
    bridge.workflow = FakeWorkflow(error)
    return bridge


class AgentLaboratoryNativeProductsTests(unittest.TestCase):
    def test_native_failure_persists_products_and_preserves_exception(self):
        errors = (
            RuntimeError("arXiv API search failed after native retries"),
            RateLimitError("aicoo_http_429"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as raw:
                workspace = Path(raw)
                bridge = bridge_for(workspace, error)
                previous = Path.cwd()
                with self.assertRaises(type(error)) as caught:
                    bridge.run_native()
                self.assertIs(caught.exception, error)
                self.assertEqual(Path.cwd(), previous)
                self.assertFalse(bridge.terminal)
                self.assertFalse((workspace / "report/report.md").exists())
                self.assertFalse((workspace / ".rac/agent_laboratory.pkl").exists())
                for relative, content in (
                    ("code/agent_laboratory/load_data.py", bridge.workflow.phd.dataset_code),
                    ("code/agent_laboratory/run_experiments.py", bridge.workflow.phd.results_code),
                    ("outputs/agent_laboratory/experiment_results.txt", bridge.workflow.phd.exp_results),
                    ("state/agent_laboratory/interpretation.txt", bridge.workflow.phd.interpretation),
                ):
                    self.assertTrue((workspace / relative).is_file(), relative)
                    self.assertEqual((workspace / relative).read_text(encoding="utf-8"), content)

    def test_successful_native_run_still_completes(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = bridge_for(Path(raw))
            previous = Path.cwd()
            result = bridge.run_native()
            self.assertEqual(result.status, "completed")
            self.assertTrue(bridge.terminal)
            self.assertEqual(Path.cwd(), previous)
            self.assertTrue((bridge.workspace / ".rac/agent_laboratory.pkl").is_file())
            self.assertEqual((bridge.workspace / "report/report.md").read_text(encoding="utf-8"),
                             bridge.workflow.phd.report)

    def test_cli_keeps_failed_status_with_preserved_products_and_partial_report(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task = root / "task"
            task.mkdir()
            (task / "task_info.json").write_text(json.dumps({"task": "Test research"}), encoding="utf-8")
            manifest = root / "configs/hosts/agent_laboratory.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"host_id": "agent_laboratory", "capabilities": []}), encoding="utf-8")
            error = RateLimitError("aicoo_http_429")
            bridge = bridge_for(root / "fixture", error)

            def initialize_native(**kwargs):
                bridge.workspace = kwargs["workspace"]
                (bridge.workspace / "report/report.md").write_text("Existing partial report", encoding="utf-8")

            args = build_parser().parse_args([
                "run-one", "--project-root", str(root), "--host", "agent_laboratory",
                "--condition", "N0", "--upstream", str(root), "--task-dir", str(task),
                "--episode-id", "failed-episode", "--model", "fake-model",
                "--max-cost-usd", "25", "--max-input-tokens", "1000", "--max-output-tokens", "1000",
                "--max-agent-calls", "10", "--max-wall-seconds", "100", "--max-hops", "10",
            ])
            with patch.dict(os.environ, {"AGENT_API_KEY": "fake-only"}), \
                    patch("rac_ai_scientist.cli.make_bridge", return_value=bridge), \
                    patch.object(bridge, "initialize_native", side_effect=initialize_native):
                with self.assertRaises(RateLimitError) as caught:
                    _run_one(args)
            self.assertIs(caught.exception, error)
            episode = root / "runs/failed-episode"
            metadata = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["reason"], "aicoo_http_429")
            self.assertEqual(metadata["error_type"], "RateLimitError")
            self.assertEqual((bridge.workspace / "report/report.md").read_text(encoding="utf-8"),
                             "Existing partial report")
            self.assertTrue((bridge.workspace / "code/agent_laboratory/run_experiments.py").is_file())


if __name__ == "__main__":
    unittest.main()
