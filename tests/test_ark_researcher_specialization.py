import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.hosts.ark import ArkBridge


DOWNSTREAM = ("experimenter", "planner", "reviewer", "writer", "coder")


class FakeResearchCompiler:
    def __init__(self, workspace: Path, *, complete_in_phase: bool = True):
        self.workspace = workspace
        self.complete_in_phase = complete_in_phase
        self.phase_calls = 0
        self.specialize_calls = 0

    def _write_context(self) -> None:
        state = self.workspace / "auto_research" / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "project_context.md").write_text(
            "# Project Context\n\n## Experimental Protocol\nRun the declared experiment.",
            encoding="utf-8",
        )

    def _write_specializations(self) -> None:
        agents = self.workspace / ".rac" / "ark_project" / "agents"
        agents.mkdir(parents=True, exist_ok=True)
        for name in DOWNSTREAM:
            (agents / f"{name}.prompt").write_text(
                f"base {name}\n\n## Project-Specific Knowledge\n{name} details",
                encoding="utf-8",
            )

    def _run_research_phase(self) -> None:
        self.phase_calls += 1
        self._write_context()
        if self.complete_in_phase:
            self._write_specializations()

    def _specialize_agent_prompts(self) -> None:
        self.specialize_calls += 1
        self._write_specializations()


class ArkResearcherSpecializationTests(unittest.TestCase):
    def bridge(self, workspace: Path, orchestrator: FakeResearchCompiler) -> ArkBridge:
        bridge = object.__new__(ArkBridge)
        bridge.workspace = workspace
        bridge.orchestrator = orchestrator
        return bridge

    def test_native_research_phase_specializes_every_downstream_prompt(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            orchestrator = FakeResearchCompiler(workspace)
            bridge = self.bridge(workspace, orchestrator)

            output = bridge._run_native_research_specialization()

            self.assertEqual(orchestrator.phase_calls, 1)
            self.assertEqual(orchestrator.specialize_calls, 0)
            self.assertTrue(bridge._research_prompts_specialized())
            self.assertIn("Experimental Protocol", output)
            for name in DOWNSTREAM:
                prompt = workspace / ".rac" / "ark_project" / "agents" / f"{name}.prompt"
                self.assertIn("## Project-Specific Knowledge", prompt.read_text(encoding="utf-8"))

    def test_partial_native_research_phase_repairs_missing_prompt_specializations(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            orchestrator = FakeResearchCompiler(workspace, complete_in_phase=False)
            bridge = self.bridge(workspace, orchestrator)

            bridge._run_native_research_specialization()

            self.assertEqual(orchestrator.phase_calls, 1)
            self.assertEqual(orchestrator.specialize_calls, 1)
            self.assertTrue(bridge._research_prompts_specialized())


if __name__ == "__main__":
    unittest.main()
