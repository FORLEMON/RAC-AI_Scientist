import json
import sys
import time
import tempfile
import types
import unittest
from enum import IntEnum, Enum
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.auto_research_claw import AutoResearchClawBridge, _raw_python_codegen_fallback
from rac_ai_scientist.schemas import Budget
from rac_ai_scientist.conditions import Condition
from rac_ai_scientist.policy import SharedPolicy, verify_result


class AutoResearchClawRuntimeTests(unittest.TestCase):
    def test_related_work_pdfs_are_translated_into_native_literature_artifacts(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            related_work = directory / "related_work"
            related_work.mkdir()
            (related_work / "paper_000.pdf").write_bytes(b"fake pdf")
            stage = directory / "auto_research_claw_native" / "stage-04"
            stage.mkdir(parents=True)
            candidates = stage / "candidates.jsonl"
            candidates.write_text(
                json.dumps({"id": "existing", "title": "Existing paper"}) + "\n",
                encoding="utf-8",
            )
            manifest = Path(__file__).resolve().parents[1] / "configs/hosts/auto_research_claw.json"
            bridge = AutoResearchClawBridge(
                directory, manifest, Budget(10, 1000, 1000, 5, 100, 5), "fake", "FAKE-ONLY"
            )
            bridge.workspace = directory
            bridge.run_dir = directory / "auto_research_claw_native"

            class PDFExtractor:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def extract(self, path):
                    return types.SimpleNamespace(
                        has_content=True,
                        title="Provided black-hole paper",
                        authors=["A. Researcher"],
                        abstract="A benchmark-provided paper about black-hole superradiance.",
                        text="Provided black-hole paper\nFull text",
                    )

            package = types.ModuleType("researchclaw")
            package.__path__ = []
            web = types.ModuleType("researchclaw.web")
            web.__path__ = []
            extractor = types.ModuleType("researchclaw.web.pdf_extractor")
            extractor.PDFExtractor = PDFExtractor
            with patch.dict(sys.modules, {
                "researchclaw": package,
                "researchclaw.web": web,
                "researchclaw.web.pdf_extractor": extractor,
            }):
                self.assertEqual(bridge._seed_related_work_candidates(), 1)
                self.assertEqual(bridge._seed_related_work_candidates(), 0)

            rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["source"], "provided_related_work")
            self.assertEqual(rows[1]["url"], "related_work/paper_000.pdf")
            self.assertEqual(rows[1]["cite_key"], "provided_paper_000")
            references = (stage / "references.bib").read_text(encoding="utf-8")
            self.assertEqual(references.count("@misc{provided_paper_000,"), 1)
            self.assertNotIn(str(directory), candidates.read_text(encoding="utf-8"))

    def test_native_research_decisions_route_and_bound_rollbacks(self):
        class Stage(IntEnum):
            SYNTHESIS = 7
            HYPOTHESIS_GEN = 8
            EXPERIMENT_RUN = 12
            ITERATIVE_REFINE = 13
            RESULT_ANALYSIS = 14
            RESEARCH_DECISION = 15
        class Status(Enum):
            DONE = "done"
        for decision, target, capability in (("refine", Stage.ITERATIVE_REFINE, "experiment"),
                                              ("pivot", Stage.HYPOTHESIS_GEN, "synthesis")):
            with self.subTest(decision=decision), tempfile.TemporaryDirectory() as raw:
                directory = Path(raw)
                manifest = Path(__file__).resolve().parents[1] / "configs/hosts/auto_research_claw.json"
                bridge = AutoResearchClawBridge(directory, manifest, Budget(10, 1000, 1000, 50, 100, 50), "fake", "FAKE-ONLY")
                bridge.workspace = directory
                bridge.run_dir = directory / "auto_research_claw_native"
                bridge.run_dir.mkdir()
                bridge.config = object()
                bridge.adapters = object()
                bridge.completed = {"scope", "literature", "synthesis", "design", "experiment"}
                code = directory / "code/auto_research_claw/main.py"
                code.parent.mkdir(parents=True)
                code.write_text("repaired")
                calls, versions = [], []
                def execute(stage, **kwargs):
                    calls.append(stage)
                    if stage == Stage.RESULT_ANALYSIS:
                        analysis = bridge.run_dir / "stage-14/analysis.md"
                        analysis.parent.mkdir(exist_ok=True)
                        analysis.write_text("Actual measurements show all conditions failed; refine the experiment.")
                        for path, text in (("stage-10/experiment/main.py", "old"), ("stage-13/experiment_final/main.py", "repaired")):
                            code = bridge.run_dir / path
                            code.parent.mkdir(parents=True, exist_ok=True)
                            code.write_text(text)
                    return types.SimpleNamespace(stage=stage, status=Status.DONE, error=None,
                                                 decision=decision if stage == Stage.RESEARCH_DECISION else "proceed")
                def version(run_dir, stage, attempt):
                    versions.append((stage, attempt))
                    for number in range(int(stage), 16):
                        source = run_dir / f"stage-{number:02d}"
                        if source.exists():
                            source.rename(run_dir / f"stage-{number:02d}_v{attempt}")
                def promote(run_dir, config):
                    (run_dir / "stage-14/analysis.md").write_text("Promoted analysis from the best measured iteration.")
                modules = {
                    "researchclaw.pipeline.executor": types.SimpleNamespace(execute_stage=execute),
                    "researchclaw.pipeline.stages": types.SimpleNamespace(Stage=Stage, StageStatus=Status,
                        DECISION_ROLLBACK={"refine": Stage.ITERATIVE_REFINE, "pivot": Stage.HYPOTHESIS_GEN}, MAX_DECISION_PIVOTS=2),
                    "researchclaw.pipeline.runner": types.SimpleNamespace(_version_rollback_stages=version,
                        _record_decision_history=lambda *args: None, _consecutive_empty_metrics=lambda *args: False,
                        _promote_best_stage14=promote),
                }
                with patch.dict(sys.modules, modules):
                    bridge.started = time.monotonic()
                    contract = SharedPolicy(Condition.R5).decide(bridge.checkpoint()).contract
                    self.assertEqual(contract.capability_id, "analysis")
                    result = bridge.invoke("analysis", contract)
                    self.assertIsNone(result.error)
                    self.assertEqual(verify_result(contract, result).verdict.value, "supported")
                    self.assertEqual((directory / "code/auto_research_claw/main.py").read_text(), "repaired")
                    self.assertEqual(bridge._next_native(), capability)
                    self.assertEqual(SharedPolicy(Condition.R5).decide(bridge.checkpoint()).capability_id, capability)
                    self.assertFalse(next(c for c in bridge._available_cards() if c.capability_id == "writing").available)
                    self.assertFalse(next(c for c in bridge._available_cards() if c.capability_id == "analysis").available)
                    bridge.invoke(capability, None)
                    self.assertEqual(calls[2], target)
                    self.assertEqual(bridge.open_issues, [])
                    self.assertEqual(versions, [(target, 1)])
                    # The upstream runner proceeds after its two rollback attempts.
                    bridge.completed.update({"scope", "literature", "synthesis", "design", "experiment"})
                    bridge.decision_attempts = 2
                    result = bridge.invoke("analysis", None)
                    self.assertIsNone(result.error)
                    self.assertIn("Native " + decision + " limit reached", result.output)
                    self.assertEqual((directory / "outputs/auto_research_claw/analysis.md").read_text(),
                                     "Promoted analysis from the best measured iteration.")
                    self.assertEqual((directory / "code/auto_research_claw/main.py").read_text(), "repaired")
                    self.assertEqual(bridge._next_native(), "writing")
                    self.assertTrue(next(c for c in bridge._available_cards() if c.capability_id == "writing").available)
                    self.assertEqual(len(versions), 1)

    def test_sandbox_uses_the_installed_python(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            upstream = directory / "upstream"
            runner = upstream / "researchclaw" / "pipeline" / "runner.py"
            runner.parent.mkdir(parents=True)
            runner.touch()
            manifest = Path(__file__).resolve().parents[1] / "configs" / "hosts" / "auto_research_claw.json"
            bridge = AutoResearchClawBridge(upstream, manifest, Budget(10, 1000, 1000, 5, 100, 5), "fake-model", "FAKE-ONLY")
            package = types.ModuleType("researchclaw")
            package.__path__ = []
            adapters = types.ModuleType("researchclaw.adapters")
            config = types.ModuleType("researchclaw.config")
            adapters.AdapterBundle = types.SimpleNamespace(from_config=lambda value: object())
            config.RCConfig = types.SimpleNamespace(from_dict=lambda value, **kwargs: value)
            workspace = directory / "workspace"
            workspace.mkdir()
            (workspace / "task_info.json").write_text(
                '{"data":[{"path":"data/sequence.json","description":"verified input"}]}',
                encoding="utf-8",
            )
            with patch.dict(sys.modules, {"researchclaw": package, "researchclaw.adapters": adapters, "researchclaw.config": config}), patch.object(
                bridge, "_install_usage_adapter"
            ), patch.object(bridge, "_install_codegen_parser_adapter"), patch.object(bridge, "_install_opencode_context"):
                bridge.initialize(episode_id="ep", workspace=workspace, objective="track objects", seed=0)
            self.assertEqual(bridge.config["experiment"]["sandbox"]["python_path"], sys.executable)
            topic = bridge.config["research"]["topic"]
            self.assertEqual(topic, "track objects")
            self.assertIn("data/sequence.json", bridge._model_instructions)
            self.assertIn("do not acquire or substitute an external dataset", bridge._model_instructions)

    def test_codegen_parser_accepts_complete_raw_python(self):
        source = "from pathlib import Path\n\nPath('result.txt').write_text('ok')\n"
        self.assertEqual(_raw_python_codegen_fallback(source), {"main.py": source.strip()})
        fragment = (
            "# In superradiance.py - fix the growth rate calculation\n"
            "def compute_growth_rate(self, mass):\n"
            "    return mass\n"
        )
        self.assertEqual(_raw_python_codegen_fallback(fragment), {})
        self.assertEqual(_raw_python_codegen_fallback("Here is the requested program:"), {})

    def test_codegen_calls_receive_the_larger_output_floor(self):
        calls = []

        class Response:
            raw = {}
            prompt_tokens = 10
            completion_tokens = 20

        class LLMClient:
            def chat(self, messages, **kwargs):
                calls.append(kwargs)
                return Response()

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest = Path(__file__).resolve().parents[1] / "configs/hosts/auto_research_claw.json"
            bridge = AutoResearchClawBridge(
                directory, manifest, Budget(10, 1000, 50000, 5, 100, 5), "fake", "FAKE-ONLY"
            )
            bridge.started = time.monotonic()
            module = types.SimpleNamespace(LLMClient=LLMClient)
            with patch.dict(sys.modules, {"researchclaw.llm.client": module}):
                bridge._install_usage_adapter()
                LLMClient().chat([], max_tokens=8192)
        self.assertEqual(calls[0]["max_tokens"], 16384)

    def test_native_stage_failure_is_reported_as_failed_with_detail(self):
        class Status(Enum):
            FAILED = "failed"

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            manifest = Path(__file__).resolve().parents[1] / "configs/hosts/auto_research_claw.json"
            bridge = AutoResearchClawBridge(directory, manifest, Budget(10, 1000, 1000, 5, 100, 5), "fake", "FAKE-ONLY")
            bridge.workspace = directory
            bridge.run_dir = directory / "auto_research_claw_native"
            bridge.run_dir.mkdir()
            bridge.config = object()
            bridge.adapters = object()
            bridge.started = time.monotonic()
            failed = types.SimpleNamespace(
                stage=types.SimpleNamespace(name="EXPERIMENT_DESIGN"),
                status=Status.FAILED,
                error="regeneration produced no main.py",
            )
            runner = types.SimpleNamespace(execute_pipeline=lambda **kwargs: [failed])
            with patch.dict(sys.modules, {"researchclaw.pipeline.runner": runner}):
                result = bridge.run_native()
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.native_status, "failed")
            self.assertIn("EXPERIMENT_DESIGN", result.reason)
            self.assertIn("no main.py", result.reason)


if __name__ == "__main__":
    unittest.main()
