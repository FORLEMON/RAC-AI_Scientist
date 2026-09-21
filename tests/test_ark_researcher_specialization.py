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


class DiskResearchCompiler(FakeResearchCompiler):
    """Observed host behavior: persist a section, return only its save receipt."""

    def __init__(
        self,
        workspace,
        *,
        section_body="Measured domain guidance",
        missing_base=None,
        section_filename="{role}_specialization.md",
    ):
        super().__init__(workspace)
        self.section_body = section_body
        self.missing_base = missing_base
        self.section_filename = section_filename

    def _write_specializations(self):
        agents = self.workspace / ".rac" / "ark_project" / "agents"
        agents.mkdir(parents=True, exist_ok=True)
        for role in DOWNSTREAM:
            section = f"## Project-Specific Knowledge\n{self.section_body}\n"
            filename = self.section_filename.format(role=role)
            (self.workspace / "auto_research" / "state" / filename).write_text(
                section,
                encoding="utf-8",
            )
            if role != self.missing_base:
                content = section if role in {"planner", "reviewer"} else "Saved the section to its specialization file."
                (agents / f"{role}.prompt").write_text(f"base {role}\n\n{content}", encoding="utf-8")


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

    def test_partial_native_research_phase_advances_without_prompt_gate_or_retry(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            orchestrator = FakeResearchCompiler(workspace, complete_in_phase=False)
            bridge = self.bridge(workspace, orchestrator)

            bridge._run_native_research_specialization()

            self.assertEqual(orchestrator.phase_calls, 1)
            self.assertEqual(orchestrator.specialize_calls, 0)
            self.assertFalse(bridge._research_prompts_specialized())

    def test_saved_native_sections_are_handed_to_existing_prompts_without_another_model_pass(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            orchestrator = DiskResearchCompiler(workspace)
            bridge = self.bridge(workspace, orchestrator)

            bridge._run_native_research_specialization()

            self.assertEqual(orchestrator.specialize_calls, 0)
            self.assertTrue(bridge._research_prompts_specialized())
            for role in DOWNSTREAM:
                prompt = workspace / ".rac" / "ark_project" / "agents" / f"{role}.prompt"
                text = prompt.read_text(encoding="utf-8")
                self.assertIn(f"base {role}", text)
                self.assertEqual(text.count("## Project-Specific Knowledge"), 1)
                self.assertIn("Measured domain guidance", text)

    def test_prompt_section_filename_is_restored_without_false_incomplete_error(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            orchestrator = DiskResearchCompiler(
                workspace,
                section_filename="{role}_prompt_section.md",
            )
            bridge = self.bridge(workspace, orchestrator)

            bridge._run_native_research_specialization()

            self.assertEqual(orchestrator.specialize_calls, 0)
            self.assertTrue(bridge._research_prompts_specialized())
            reviewer = workspace / ".rac/ark_project/agents/reviewer.prompt"
            self.assertIn(
                "Measured domain guidance",
                reviewer.read_text(encoding="utf-8"),
            )

    def test_knowledge_filename_is_restored_without_another_researcher_hop(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            orchestrator = DiskResearchCompiler(
                workspace,
                section_filename="{role}_knowledge.md",
            )
            bridge = self.bridge(workspace, orchestrator)

            bridge._run_native_research_specialization()

            self.assertEqual(orchestrator.phase_calls, 1)
            self.assertEqual(orchestrator.specialize_calls, 0)
            self.assertTrue(bridge._research_prompts_specialized())
            experimenter = workspace / ".rac/ark_project/agents/experimenter.prompt"
            self.assertIn(
                "Measured domain guidance",
                experimenter.read_text(encoding="utf-8"),
            )

    def test_outputs_specialization_is_restored_best_effort(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            orchestrator = DiskResearchCompiler(workspace)

            def write_outputs():
                orchestrator._write_context()
                agents = workspace / ".rac" / "ark_project" / "agents"
                outputs = workspace / "outputs"
                agents.mkdir(parents=True, exist_ok=True)
                outputs.mkdir(parents=True, exist_ok=True)
                for role in DOWNSTREAM:
                    (agents / f"{role}.prompt").write_text(
                        f"base {role}\n\nSaved specialization to outputs.",
                        encoding="utf-8",
                    )
                    (outputs / f"{role}_specialization.md").write_text(
                        f"## Project-Specific Knowledge\n{role} output guidance.",
                        encoding="utf-8",
                    )

            orchestrator._run_research_phase = write_outputs
            bridge = self.bridge(workspace, orchestrator)

            bridge._run_native_research_specialization()

            self.assertEqual(orchestrator.specialize_calls, 0)
            self.assertTrue(bridge._research_prompts_specialized())
            for role in DOWNSTREAM:
                prompt = workspace / ".rac/ark_project/agents" / f"{role}.prompt"
                self.assertIn(
                    f"{role} output guidance.",
                    prompt.read_text(encoding="utf-8"),
                )

    def test_role_labelled_project_context_sections_are_restored(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            orchestrator = DiskResearchCompiler(workspace)

            def write_context_sections():
                state = workspace / "auto_research" / "state"
                state.mkdir(parents=True, exist_ok=True)
                blocks = ["# Project Context", "Verified project facts."]
                for role in DOWNSTREAM:
                    blocks.extend(
                        (
                            f"### For the {role.title()} Agent",
                            "## Project-Specific Knowledge",
                            f"{role} context-only guidance.",
                        )
                    )
                (state / "project_context.md").write_text(
                    "\n\n".join(blocks),
                    encoding="utf-8",
                )
                agents = workspace / ".rac" / "ark_project" / "agents"
                agents.mkdir(parents=True, exist_ok=True)
                for role in DOWNSTREAM:
                    (agents / f"{role}.prompt").write_text(
                        f"base {role}\n",
                        encoding="utf-8",
                    )

            orchestrator._run_research_phase = write_context_sections
            bridge = self.bridge(workspace, orchestrator)

            bridge._run_native_research_specialization()

            self.assertEqual(orchestrator.specialize_calls, 0)
            for role in DOWNSTREAM:
                prompt = workspace / ".rac/ark_project/agents" / f"{role}.prompt"
                text = prompt.read_text(encoding="utf-8")
                self.assertEqual(text.count("## Project-Specific Knowledge"), 1)
                self.assertIn(f"{role} context-only guidance.", text)

    def test_unlabelled_context_section_is_not_assigned_to_a_role(self):
        text = (
            "# Project Context\n\n"
            "The planner is mentioned in prose.\n\n"
            "## Project-Specific Knowledge\nGeneric context only."
        )
        self.assertEqual(
            ArkBridge._research_specialization_from_context(text, "planner"),
            "",
        )

    def test_saved_sections_never_replace_missing_base_prompts_but_do_not_block_phase(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            bridge = self.bridge(workspace, DiskResearchCompiler(workspace, missing_base="coder"))
            bridge._run_native_research_specialization()
            self.assertFalse((workspace / ".rac/ark_project/agents/coder.prompt").exists())

    def test_empty_sections_and_inline_receipts_do_not_count_as_specialized_but_do_not_block(self):
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            bridge = self.bridge(workspace, DiskResearchCompiler(workspace, section_body=""))
            bridge._run_native_research_specialization()
            prompt = workspace / ".rac/ark_project/agents/experimenter.prompt"
            prompt.write_text('The file contains `## Project-Specific Knowledge`.\nSaved successfully.', encoding="utf-8")
            self.assertFalse(bridge._research_prompts_specialized())

    def test_specialization_body_ends_at_the_next_peer_or_parent_heading(self):
        header = "## Project-Specific Knowledge\n"
        for next_heading in ("# Other Section", "## Other Section"):
            with self.subTest(next_heading=next_heading):
                empty = header + "\n" + next_heading + "\nUnrelated instructions."
                self.assertEqual(ArkBridge._research_specialization_section(empty), "")
                valid = header + "Measured domain guidance.\n### Installation\nUse the existing environment."
                self.assertEqual(ArkBridge._research_specialization_section(valid + "\n" + next_heading + "\nUnrelated."), valid)
                self.assertEqual(ArkBridge._research_specialization_section(empty + "\n" + valid), valid)

    def test_missing_context_does_not_block_when_research_changed_other_files(self):
        with tempfile.TemporaryDirectory() as raw:
            orchestrator = FakeResearchCompiler(Path(raw))
            orchestrator._write_context = lambda: None
            bridge = self.bridge(Path(raw), orchestrator)
            output = bridge._run_native_research_specialization()
            self.assertIn("observable workspace changes", output)

    def test_native_terminal_error_stops_without_specialization_retry(self):
        with tempfile.TemporaryDirectory() as raw:
            orchestrator = FakeResearchCompiler(Path(raw), complete_in_phase=False)
            orchestrator._terminal_error = "provider rejected the research call"
            bridge = self.bridge(Path(raw), orchestrator)
            with self.assertRaisesRegex(RuntimeError, "provider rejected the research call"):
                bridge._run_native_research_specialization()
            self.assertEqual(orchestrator.specialize_calls, 0)

    def test_no_workspace_change_is_rejected_without_specialization_retry(self):
        with tempfile.TemporaryDirectory() as raw:
            orchestrator = FakeResearchCompiler(Path(raw), complete_in_phase=False)
            orchestrator._run_research_phase = lambda: None
            bridge = self.bridge(Path(raw), orchestrator)
            with self.assertRaisesRegex(RuntimeError, "no new or modified workspace file"):
                bridge._run_native_research_specialization()
            self.assertEqual(orchestrator.specialize_calls, 0)


if __name__ == "__main__":
    unittest.main()
