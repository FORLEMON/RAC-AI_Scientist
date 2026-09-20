import json
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rac_ai_scientist.conditions import Condition
from rac_ai_scientist.hosts.auto_research_claw import (
    AutoResearchClawBridge, _benchmark_execution_topic, render_contract_prompt,
)
from rac_ai_scientist.schemas import Budget, WorkContract


class AutoInstructionContextTests(unittest.TestCase):
    def test_scientific_topic_is_separate_from_both_model_instruction_routes(self):
        for condition in ("N0", "R1", "R2", "R3", "R4", "R5"):
            for fail in (False, True) if condition == "R5" else (False,):
                with self.subTest(condition=condition, fail=fail), tempfile.TemporaryDirectory() as raw:
                    root = Path(raw)
                    native_entry = root / "researchclaw/pipeline/runner.py"
                    native_entry.parent.mkdir(parents=True)
                    native_entry.touch()
                    workspace = root / "workspace"
                    workspace.mkdir()
                    (workspace / "task_info.json").write_text(json.dumps({"data": [{"path": "data/sample.dat"}]}))
                    objective = "Measure a scientific effect"
                    expected_execution = _benchmark_execution_topic(workspace, objective)
                    manifest = Path(__file__).parents[1] / "configs/hosts/auto_research_claw.json"
                    bridge = AutoResearchClawBridge(root, manifest, Budget(10, 10000, 10000, 50, 100, 20), "fake", "FAKE")
                    chats, beasts, queries = [], [], []
                    @dataclass
                    class Research:
                        topic: str
                    @dataclass
                    class Config:
                        research: Research
                    class LLMClient:
                        def chat(self, messages, *, system=None, **kwargs):
                            chats.append((messages, system))
                            return SimpleNamespace(raw={}, prompt_tokens=1, completion_tokens=1)
                    class OpenCodeBridge:
                        def generate(self, stage_dir, topic, exp_plan, metric, pkg_hint="", extra_guidance="", time_budget_sec=300):
                            beasts.append((topic, extra_guidance))
                    done = SimpleNamespace(value="done")
                    def execute(stage, **kwargs):
                        topic = kwargs["config"].research.topic
                        # Native novelty's first query is exactly config.research.topic.
                        queries.append(topic)
                        LLMClient().chat([{"role": "user", "content": topic}], system="native system")
                        OpenCodeBridge().generate(root, topic, "native plan", "metric", extra_guidance="native guidance")
                        if fail:
                            raise RuntimeError("native stage failure")
                        return SimpleNamespace(stage=SimpleNamespace(name=str(stage)), status=done, error=None)
                    modules = {
                        "researchclaw.adapters": SimpleNamespace(AdapterBundle=SimpleNamespace(from_config=lambda cfg: object())),
                        "researchclaw.config": SimpleNamespace(RCConfig=SimpleNamespace(from_dict=lambda data, **kwargs:
                            Config(Research(data["research"]["topic"])))),
                        "researchclaw.llm.client": SimpleNamespace(LLMClient=LLMClient),
                        "researchclaw.pipeline.opencode_bridge": SimpleNamespace(OpenCodeBridge=OpenCodeBridge),
                        "researchclaw.pipeline.executor": SimpleNamespace(execute_stage=execute),
                        "researchclaw.pipeline.stages": SimpleNamespace(Stage=int, StageStatus=SimpleNamespace(DONE=done)),
                        "researchclaw.pipeline.runner": SimpleNamespace(execute_pipeline=lambda **kwargs: [execute(1, **kwargs)]),
                    }
                    contract = WorkContract("test", "scope", objective, ("data/**",), ("state/**",), ())
                    coordination = "room context\n" + render_contract_prompt(objective, "scope", contract)
                    with patch.dict(sys.modules, modules), patch.object(bridge, "_install_codegen_parser_adapter"), \
                         patch.object(bridge, "_normalize_products"), \
                         patch.object(bridge, "communication_prompt", side_effect=lambda role, prompt, contract: "room context\n" + prompt):
                        bridge.initialize(episode_id="test", workspace=workspace, objective=objective, seed=0)
                        bridge.configure_condition(Condition.parse(condition))
                        if condition == "N0":
                            bridge.run_native()
                        else:
                            bridge.sharednet = object()
                            result = bridge.invoke("scope", contract)
                            self.assertEqual(bool(result.error), fail)
                        self.assertTrue(queries)
                        self.assertEqual(set(queries), {objective})
                        expected = expected_execution + ("\n\n" + coordination if condition != "N0" else "")
                        self.assertTrue(all(system == "native system\n\n" + expected for _, system in chats))
                        self.assertTrue(all(topic == objective and guidance == "native guidance\n\n" + expected for topic, guidance in beasts))
                        self.assertEqual(bridge.config.research.topic, objective)
                        # After success or exception, a later call has no stale room contract.
                        LLMClient().chat([], system="after")
                        OpenCodeBridge().generate(root, objective, "native plan", "metric", extra_guidance="after")
                        self.assertEqual(chats[-1][1], "after\n\n" + expected_execution)
                        self.assertEqual(beasts[-1][1], "after\n\n" + expected_execution)


if __name__ == "__main__":
    unittest.main()
