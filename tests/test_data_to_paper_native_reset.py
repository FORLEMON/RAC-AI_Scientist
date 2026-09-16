import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rac_ai_scientist.hosts.data_to_paper import DataToPaperBridge
from rac_ai_scientist.schemas import Budget


class FakeRunner:
    def __init__(self, returned):
        self.returned = returned
        self.stages_to_conversations_lens = {}

    def advance_stage(self, stage):
        pass

    def _run_stage(self, stage):
        return self.returned


def bridge_for(workspace, returned):
    bridge = DataToPaperBridge.__new__(DataToPaperBridge)
    bridge.workspace = workspace
    bridge.runner = FakeRunner(returned)
    bridge.initial_budget = Budget(20, 1000, 1000, 10, 100, 10)
    bridge.started = time.monotonic()
    bridge.completed = set()
    bridge.hop = 0
    bridge.terminal = False
    bridge.provider_calls = 0
    bridge.input_tokens = 0
    bridge.output_tokens = 0
    bridge.provider_cost_usd = 0.0
    bridge.cost_is_provider_reported = False
    return bridge


class DataToPaperNativeResetTests(unittest.TestCase):
    def test_native_termination_does_not_complete_analysis(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = bridge_for(Path(raw), False)
            with patch.object(DataToPaperBridge, "_available_cards", return_value=[SimpleNamespace(capability_id="data_analysis", available=True)]), patch.object(DataToPaperBridge, "_stages_for", return_value=("code",)):
                result = bridge.invoke("data_analysis", None)
            self.assertNotIn("data_analysis", bridge.completed)
            self.assertIn("terminated", result.error)

    def test_native_reset_keeps_analysis_unresolved(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = bridge_for(Path(raw), "data")
            with patch.object(DataToPaperBridge, "_available_cards", return_value=[SimpleNamespace(capability_id="data_analysis", available=True)]), patch.object(DataToPaperBridge, "_stages_for", return_value=("code",)):
                result = bridge.invoke("data_analysis", None)
            self.assertNotIn("data_analysis", bridge.completed)
            self.assertFalse(result.proposed_done)

    def test_compile_without_report_is_not_completion(self):
        with tempfile.TemporaryDirectory() as raw:
            bridge = bridge_for(Path(raw), None)
            with patch.object(DataToPaperBridge, "_available_cards", return_value=[SimpleNamespace(capability_id="compile", available=True)]), patch.object(DataToPaperBridge, "_stages_for", return_value=("compile",)):
                result = bridge.invoke("compile", None)
            self.assertFalse(bridge.terminal)
            self.assertFalse(result.proposed_done)
            self.assertIn("report", result.error)

    def test_compile_with_report_proposes_completion(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            native = workspace / "d2p_native"
            native.mkdir()
            (native / "paper.tex").write_text("A completed manuscript", encoding="utf-8")
            bridge = bridge_for(workspace, None)
            with patch.object(DataToPaperBridge, "_available_cards", return_value=[SimpleNamespace(capability_id="compile", available=True)]), patch.object(DataToPaperBridge, "_stages_for", return_value=("compile",)):
                result = bridge.invoke("compile", None)
            self.assertTrue(bridge.terminal)
            self.assertTrue(result.proposed_done)


if __name__ == "__main__":
    unittest.main()
