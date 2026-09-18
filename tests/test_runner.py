import dataclasses
import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.artifacts import snapshot_workspace
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

    def test_r1_rolls_back_timeout_marks_failed_and_continues_fixed_workflow(self):
        class TimeoutBridge(HostBridge):
            host_id = "timeout"

            def __init__(self, workspace):
                self.workspace = workspace
                self.stage = "run"
                self.invoked = []
                self.failures = []

            def initialize(self, **kwargs):
                pass

            def checkpoint(self):
                cards = [
                    CapabilityCard("run", "experiment", ("experiment",), (), ("outputs/**",), ("result",)),
                    CapabilityCard("write", "write", ("writing",), (), ("report/**",), ("terminal_report",)),
                ]
                return Checkpoint(
                    "ep",
                    len(self.invoked),
                    "finish",
                    self.stage,
                    snapshot_workspace(self.workspace),
                    [Issue(f"native:{self.stage}", "native_requirement", f"{self.stage} remains")],
                    Budget(10, 10000, 10000, 10, 100, 10),
                    cards,
                )

            def native_next(self, checkpoint):
                return self.stage

            def transaction_workspace(self):
                return self.workspace

            def invoke(self, capability_id, contract):
                self.invoked.append(capability_id)
                before = snapshot_workspace(self.workspace)
                if capability_id == "run":
                    partial = self.workspace / "outputs" / "partial.json"
                    partial.parent.mkdir(parents=True, exist_ok=True)
                    partial.write_text("partial", encoding="utf-8")
                    return InvocationResult(
                        capability_id,
                        "",
                        before,
                        snapshot_workspace(self.workspace),
                        timed_out=True,
                        metrics={"native_timeout_continuable": 1.0},
                    )
                return InvocationResult(capability_id, "report complete", before, before, proposed_done=True)

            def fail_invocation(self, result, evaluation):
                self.failures.append(result.capability_id)
                marker = self.workspace / "state" / "failures.jsonl"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text('{"capability_id":"run","status":"failed"}\n', encoding="utf-8")
                self.stage = "write"

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bridge = TimeoutBridge(root)
            outcome = EpisodeRunner(bridge, "R1", JsonlLedger(root / "trace.jsonl")).run(hard_hop_limit=3)

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(bridge.invoked, ["run", "write"])
            self.assertEqual(bridge.failures, ["run"])
            self.assertFalse((root / "outputs" / "partial.json").exists())
            self.assertTrue((root / "state" / "failures.jsonl").is_file())

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

    def test_refuted_invocation_is_rolled_back_before_retry(self):
        class TransactionalBridge(HostBridge):
            host_id = "transactional"

            def __init__(self, workspace):
                self.workspace = workspace
                self.attempts = 0
                self.done = False

            def initialize(self, **kwargs):
                pass

            def checkpoint(self):
                return Checkpoint(
                    "ep",
                    self.attempts,
                    "write",
                    "write",
                    snapshot_workspace(self.workspace),
                    [] if self.done else [Issue("native:write", "native_requirement", "write remains", required_tags=("writing",))],
                    Budget(10, 10000, 10000, 10, 100, 10),
                    [CapabilityCard("write", "write", ("writing", "finalize"), (), ("report/**",), ("terminal_report",))],
                    terminal=self.done,
                )

            def native_next(self, checkpoint):
                return "write"

            def transaction_workspace(self):
                return self.workspace

            def invoke(self, capability_id, contract):
                before = snapshot_workspace(self.workspace)
                self.attempts += 1
                report = self.workspace / "report" / "report.md"
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text("valid report" * 40, encoding="utf-8")
                if self.attempts == 1:
                    (self.workspace / "unauthorized").write_text("bad", encoding="utf-8")
                return InvocationResult(
                    capability_id,
                    "done",
                    before,
                    snapshot_workspace(self.workspace),
                    proposed_done=True,
                )

            def accept_invocation(self, result, evaluation):
                self.done = True

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bridge = TransactionalBridge(root)
            outcome = EpisodeRunner(bridge, "R5", JsonlLedger(root / "trace.jsonl")).run(hard_hop_limit=2)

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.hops, 2)
            self.assertFalse((root / "unauthorized").exists())
            self.assertTrue((root / "report" / "report.md").is_file())

    def test_recovery_retries_same_capability_without_consuming_control_hop(self):
        class RecoveringBridge(HostBridge):
            host_id = "recovering"

            def __init__(self, workspace):
                self.workspace = workspace
                self.attempts = 0

            def initialize(self, **kwargs):
                pass

            def checkpoint(self):
                return Checkpoint(
                    "ep",
                    self.attempts,
                    "run",
                    "run",
                    snapshot_workspace(self.workspace),
                    [Issue("native:run", "native_requirement", "run remains", required_tags=("experiment",))],
                    Budget(10, 10000, 10000, 10, 100, 10),
                    [CapabilityCard("run", "run", ("experiment",), (), ("outputs/**",), ("result",))],
                )

            def native_next(self, checkpoint):
                return "run"

            def transaction_workspace(self):
                return self.workspace

            def invoke(self, capability_id, contract):
                before = snapshot_workspace(self.workspace)
                self.attempts += 1
                output = self.workspace / "outputs" / "result.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(f'{{"attempt": {self.attempts}}}', encoding="utf-8")
                if self.attempts == 1:
                    (self.workspace / "unauthorized").write_text("bad", encoding="utf-8")
                    return InvocationResult(capability_id, "", before, snapshot_workspace(self.workspace), timed_out=True)
                return InvocationResult(capability_id, "done", before, snapshot_workspace(self.workspace), proposed_done=True)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            outcome = EpisodeRunner(
                RecoveringBridge(root),
                "R5",
                JsonlLedger(root / "trace.jsonl"),
            ).run(hard_hop_limit=2)

            self.assertEqual(outcome.status, "completed")
            self.assertEqual(outcome.hops, 2)
            self.assertTrue((root / "outputs" / "result.json").is_file())
            self.assertFalse((root / "unauthorized").exists())
