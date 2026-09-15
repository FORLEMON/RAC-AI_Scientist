import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.cli import _execute_episode, _uses_host_native_n0
from rac_ai_scientist.hosts.ark import (
    NATIVE_DEV_ITERATIONS,
    NATIVE_REVIEW_ITERATIONS,
    ArkBridge,
)
from rac_ai_scientist.hosts.agent_laboratory import AgentLaboratoryBridge
from rac_ai_scientist.hosts.ai_researcher import AIResearcherBridge
from rac_ai_scientist.hosts.auto_research_claw import AutoResearchClawBridge
from rac_ai_scientist.hosts.data_to_paper import DataToPaperBridge
from rac_ai_scientist.hosts.evo_scientist import EvoScientistBridge
from rac_ai_scientist.ledger import JsonlLedger
from rac_ai_scientist.schemas import Budget, NativeRunResult, Usage


class FakeOrchestrator:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.iteration = 0
        self._stop_requested = False
        self._agent_stats = []
        self.run_calls = 0
        self.code_dir = Path(kwargs["code_dir"])
        type(self).created.append(self)

    def run(self):
        self.run_calls += 1
        self.iteration = NATIVE_REVIEW_ITERATIONS
        report = self.code_dir / "report"
        report.mkdir(parents=True, exist_ok=True)
        (report / "main.tex").write_text("native ARK report", encoding="utf-8")
        self._agent_stats.append(
            {
                "cost_usd": 1.25,
                "input_tokens": 120,
                "output_tokens": 30,
            }
        )

    def load_paper_state(self):
        return {"status": "accepted", "current_score": 8.5}


class NativeOnlyBridge:
    host_id = "ark"

    def __init__(self):
        self.calls = 0

    def run_native(self):
        self.calls += 1
        return NativeRunResult(
            status="completed",
            reason="native complete",
            native_iterations=3,
            artifacts_before=[],
            artifacts_after=[],
            usage=Usage(agent_calls=7),
            native_status="accepted",
        )


class ArkNativeN0Tests(unittest.TestCase):
    def test_all_n0_conditions_select_host_native_execution(self):
        for host in (
            "ark", "agent_laboratory", "data_to_paper", "ai_researcher",
            "evo_scientist", "auto_research_claw",
        ):
            self.assertTrue(_uses_host_native_n0(host, "N0"))
            self.assertFalse(_uses_host_native_n0(host, "R1"))

    def test_every_host_implements_its_own_native_entrypoint(self):
        for bridge_type in (
            ArkBridge,
            AgentLaboratoryBridge,
            DataToPaperBridge,
            AIResearcherBridge,
            EvoScientistBridge,
            AutoResearchClawBridge,
        ):
            self.assertIn("initialize_native", bridge_type.__dict__)
            self.assertIn("run_native", bridge_type.__dict__)

    def test_n0_bypasses_episode_runner_and_records_only_native_boundary(self):
        bridge = NativeOnlyBridge()
        with tempfile.TemporaryDirectory() as raw:
            ledger = JsonlLedger(Path(raw) / "coordination.jsonl")
            with patch("rac_ai_scientist.cli.EpisodeRunner") as runner:
                outcome, result = _execute_episode(
                    bridge,
                    "N0",
                    ledger,
                    hard_hop_limit=40,
                    review_score_threshold=8.0,
                )

            runner.assert_not_called()
            self.assertEqual(bridge.calls, 1)
            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.hops, 0)
            self.assertIsNotNone(result)
            events = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["type"] for event in events], ["native_run_start", "native_run_complete"])
            self.assertNotIn("decision", {event["type"] for event in events})

    def test_native_initialization_preserves_ark_research_and_dev_phases(self):
        FakeOrchestrator.created.clear()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            upstream = root / "ARK"
            (upstream / "ark" / "orchestrator").mkdir(parents=True)
            templates = upstream / "ark" / "templates" / "agents"
            templates.mkdir(parents=True)
            (templates / "researcher.prompt").write_text("{PROJECT_NAME}", encoding="utf-8")

            manifest = root / "ark.json"
            manifest.write_text(
                json.dumps({"host_id": "ark", "capabilities": []}),
                encoding="utf-8",
            )
            workspace = root / "workspace"

            ark_package = types.ModuleType("ark")
            ark_package.__path__ = []
            orchestrator_module = types.ModuleType("ark.orchestrator")
            orchestrator_module.Orchestrator = FakeOrchestrator
            with patch.dict(
                sys.modules,
                {"ark": ark_package, "ark.orchestrator": orchestrator_module},
            ):
                bridge = ArkBridge(
                    upstream,
                    manifest,
                    Budget(25, 100_000_000, 1_800_000, 50, 43_200, 40),
                    "azure/gpt-5.5",
                    "test-key",
                )
                objective = "Study this task:\nuse the supplied data."
                bridge.initialize_native(
                    episode_id="native-episode",
                    workspace=workspace,
                    objective=objective,
                    seed=0,
                )

                orchestrator = FakeOrchestrator.created[-1]
                self.assertEqual(orchestrator.kwargs["max_iterations"], NATIVE_REVIEW_ITERATIONS)
                self.assertFalse((workspace / "report" / "main.tex").exists())
                self.assertFalse((workspace / "auto_research" / "state" / "idea.md").exists())
                self.assertFalse((workspace / "auto_research" / "state" / "project_context.md").exists())

                self.assertFalse((workspace / ".rac").exists())
                config = (workspace / ".ark" / "native_project" / "config.yaml").read_text(encoding="utf-8")
                self.assertIn(f"max_dev_iterations: {NATIVE_DEV_ITERATIONS}", config)
                self.assertIn(f"max_iterations: {NATIVE_REVIEW_ITERATIONS}", config)
                self.assertIn(json.dumps(objective, ensure_ascii=False), config)

                result = bridge.run_native()

            self.assertEqual(orchestrator.run_calls, 1)
            self.assertEqual(result.native_iterations, NATIVE_REVIEW_ITERATIONS)
            self.assertEqual(result.native_status, "accepted")
            self.assertEqual(result.usage.agent_calls, 1)
            self.assertEqual(
                (workspace / "report" / "report.md").read_text(encoding="utf-8"),
                "native ARK report",
            )


if __name__ == "__main__":
    unittest.main()
