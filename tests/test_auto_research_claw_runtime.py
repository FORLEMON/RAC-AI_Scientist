import sys
import time
import tempfile
import types
import unittest
from enum import IntEnum, Enum
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.auto_research_claw import AutoResearchClawBridge
from rac_ai_scientist.schemas import Budget
from rac_ai_scientist.conditions import Condition
from rac_ai_scientist.policy import SharedPolicy, verify_result


class AutoResearchClawRuntimeTests(unittest.TestCase):
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
            with patch.dict(sys.modules, {"researchclaw": package, "researchclaw.adapters": adapters, "researchclaw.config": config}), patch.object(
                bridge, "_install_usage_adapter"
            ):
                bridge.initialize(episode_id="ep", workspace=directory / "workspace", objective="track objects", seed=0)
            self.assertEqual(bridge.config["experiment"]["sandbox"]["python_path"], sys.executable)


if __name__ == "__main__":
    unittest.main()
