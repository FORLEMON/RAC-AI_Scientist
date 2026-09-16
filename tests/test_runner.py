import dataclasses
import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.bridge import HostBridge
from rac_ai_scientist.ledger import JsonlLedger
from rac_ai_scientist.runner import EpisodeRunner
from rac_ai_scientist.schemas import (
    ArtifactRecord,
    Budget,
    CapabilityCard,
    Checkpoint,
    InvocationResult,
    Issue,
    Usage,
    WorkContract,
)


class FakeBridge(HostBridge):
    host_id = "fake"

    def __init__(self):
        self.done = False
        self.invocations = 0

    def initialize(self, *, episode_id: str, workspace: Path, objective: str, seed: int) -> None:
        pass

    def checkpoint(self):
        artifacts = []
        if self.done:
            artifacts.append(ArtifactRecord("report", "report/report.md", "terminal_report", "new", 500))
        return Checkpoint(
            "ep",
            self.invocations,
            "finish report",
            "write",
            artifacts,
            [] if self.done else [Issue("I-0", "writing", "missing report")],
            Budget(10, 1000, 1000, 5, 100, 5),
            [CapabilityCard("write", "write report", ("writing", "finalize"), (), ("report/**",), ("terminal_report",))],
            terminal=self.done,
        )

    def native_next(self, checkpoint):
        return None if self.done else "write"

    def invoke(self, capability_id: str, contract: WorkContract | None):
        self.invocations += 1
        self.done = True
        after = ArtifactRecord("report", "report/report.md", "terminal_report", "new", 500)
        return InvocationResult(capability_id, "wrote report", [], [after], Usage(agent_calls=1))


class RunnerTests(unittest.TestCase):
    def test_r5_episode_reaches_terminal_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            ledger = JsonlLedger(Path(raw) / "trace.jsonl")
            outcome = EpisodeRunner(FakeBridge(), "R5", ledger).run(hard_hop_limit=3)
            self.assertEqual(outcome.status, "stop")
            self.assertTrue(ledger.path.is_file())
            self.assertGreaterEqual(len(ledger.path.read_text(encoding="utf-8").splitlines()), 4)

    def test_failed_native_invocation_is_not_completion(self):
        class FailedBridge(FakeBridge):
            def invoke(self, capability_id, contract):
                return InvocationResult(capability_id, "", [], [], error="provider failed")

        with tempfile.TemporaryDirectory() as raw:
            outcome = EpisodeRunner(FailedBridge(), "N0", JsonlLedger(Path(raw) / "trace.jsonl")).run(hard_hop_limit=2)
        self.assertEqual(outcome.status, "failed")

    def test_budget_stop_is_not_successful_stop(self):
        class ExhaustedBridge(FakeBridge):
            def checkpoint(self):
                return dataclasses.replace(super().checkpoint(), remaining_budget=Budget(0, 1000, 1000, 5, 100, 5))

        with tempfile.TemporaryDirectory() as raw:
            outcome = EpisodeRunner(ExhaustedBridge(), "N0", JsonlLedger(Path(raw) / "trace.jsonl")).run(hard_hop_limit=2)
        self.assertEqual(outcome.status, "budget_exhausted")

    def test_provider_http_402_is_a_terminal_budget_result(self):
        class BudgetRejectedBridge(FakeBridge):
            def invoke(self, capability_id, contract):
                return InvocationResult(
                    capability_id, "", [], [],
                    error="ProviderStreamError: Error code: 402 - request limit reached",
                )

        with tempfile.TemporaryDirectory() as raw:
            outcome = EpisodeRunner(
                BudgetRejectedBridge(), "R5", JsonlLedger(Path(raw) / "trace.jsonl")
            ).run(hard_hop_limit=10)
        self.assertEqual(outcome.status, "budget_exhausted")
        self.assertEqual(outcome.hops, 1)
        self.assertIn("402", outcome.reason)
