import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rac_ai_scientist.hosts.auto_research_claw import AutoResearchClawBridge
from rac_ai_scientist.schemas import Budget


class AutoResearchClawRuntimeTests(unittest.TestCase):
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
