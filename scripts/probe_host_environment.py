"""Deep offline host check. Run in a disposable container with --network none."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


class RuntimeProbe:
    def __init__(self):
        self.calls = []
        self.attestations = []

    def request(self, operation, payload):
        assert operation == "attest"
        self.attestations.append(payload)

    def execute(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return dict(stdout="remote-runtime-probe", stderr="", exit_code=0,
                    truncated=False, timed_out=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, choices=("ark", "agent_laboratory", "evo_scientist"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    host = Path(os.environ.get("RAC_HOST_ROOT", "/opt/host"))
    checks, packages = [], []
    from rac_ai_scientist.provenance import expected_tree_hash, tree_hash
    import rac_ai_scientist
    integration = Path(rac_ai_scientist.__file__).resolve().parents[2]
    spec = json.loads((integration / "upstream.lock.json").read_text())["upstreams"][args.host]
    actual = tree_hash(host)[0]
    assert expected_tree_hash(spec, actual, packaged=True) == actual, "host source differs from lock"
    checks.append("locked-source-tree")
    runtime = RuntimeProbe()
    with tempfile.TemporaryDirectory(prefix="rac-host-probe-") as raw:
        work = Path(raw)
        document = work / "probe.tex"
        document.write_text(r"\documentclass{article}\begin{document}Offline environment probe.\end{document}")
        compiled = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", document.name],
                                  cwd=work, capture_output=True, text=True, timeout=60)
        assert compiled.returncode == 0 and (work / "probe.pdf").is_file(), compiled.stdout[-1000:]
        checks.append("latex-pdf-compilation")
        if args.host == "agent_laboratory":
            copied = work / "host"
            shutil.copytree(host, copied)
            sys.path.insert(0, str(copied))
            os.chdir(work)
            import tiktoken
            assert tiktoken.get_encoding("cl100k_base").encode("offline smoke")
            from ai_lab_repo import LaboratoryWorkflow
            import torch
            import tensorflow as tf
            import app
            assert torch.version.cuda is None, "expected CPU-only PyTorch"
            assert int(torch.tensor([1, 2]).sum()) == 3
            assert int(tf.reduce_sum([1, 2]).numpy()) == 3
            assert app.model.encode(["offline smoke"]).shape == (1, 384)
            from rac_ai_scientist.task_runtime.hooks import install_agent_laboratory
            install_agent_laboratory(runtime)
            import tools
            assert tools.execute_code("raise RuntimeError('must execute remotely')") == "remote-runtime-probe"
            assert len(runtime.calls) == 1 and runtime.attestations
            checks += ["full-native-workflow-import", "torch-cpu", "tensorflow-cpu", "MiniLM-offline-embedding", "cl100k_base-offline"]
            packages += ["torch", "tensorflow-cpu", "sentence-transformers", "transformers", "tiktoken"]
        elif args.host == "evo_scientist":
            from EvoScientist.EvoScientist import create_cli_agent
            from EvoScientist.backends import CustomSandboxBackend
            from EvoScientist.middleware.code_interpreter import aclose_code_interpreters
            import langchain_quickjs
            backend = CustomSandboxBackend(root_dir=str(work), virtual_mode=True)
            result = backend.execute("python -c 'print(2 + 3)'", timeout=15)
            assert result.exit_code == 0 and "5" in result.output, result
            from rac_ai_scientist.task_runtime.hooks import install_evo_scientist
            install_evo_scientist(runtime)
            result = backend.execute("printf remote-probe", timeout=15)
            assert result.output == "remote-runtime-probe" and len(runtime.calls) == 1 and runtime.attestations
            checks += ["native-agent-factory-import", "quickjs-import", "native-shell-execution"]
            packages += ["EvoScientist", "deepagents", "langchain", "langchain-quickjs", "langgraph"]
        else:
            from ark.orchestrator import Orchestrator
            from ark.engines.cli import OpenHandsCLI
            from ark.latex.compiler import CompilerMixin
            from website.dashboard.db import SQLModel
            sys.path.insert(0, str(host / "submodules/PaperBanana"))
            # These are import-time placeholders only; --network none prevents
            # any provider request. No credentials are needed for this probe.
            os.environ.setdefault("OPENAI_API_KEY", "offline-probe")
            os.environ.setdefault("GOOGLE_API_KEY", "offline-probe")
            from agents.planner_agent import PlannerAgent
            from agents.visualizer_agent import VisualizerAgent
            from utils.paperviz_processor import PaperVizProcessor
            for executable, arguments in (("openhands", ["--help"]), ("conda", ["--version"]),
                                          ("pdflatex", ["--version"]), ("pandoc", ["--version"]), ("tmux", ["-V"])):
                process = subprocess.run([executable, *arguments], capture_output=True, text=True, timeout=60)
                assert process.returncode == 0, f"{executable}: {process.stderr[-1000:]}"
            from rac_ai_scientist.task_runtime.hooks import openhands_interpreter
            env = dict(os.environ, PYTHONPATH=str(integration / "src"))
            subprocess.run(["uv", "pip", "check", "--python", openhands_interpreter()], check=True, timeout=60)
            subprocess.run([openhands_interpreter(), str(Path(__file__).with_name("probe_openhands.py"))],
                           env=env, check=True, timeout=90)
            subprocess.run(["conda", "run", "--no-capture-output", "-n", "ark-base", "python", "-c",
                            "import numpy, scipy, pandas, matplotlib, sklearn, ark; print('conda science imports OK')"],
                           check=True, timeout=60)
            checks += ["native-orchestrator-import", "PaperBanana-import", "OpenHands-CLI", "conda", "latex", "pandoc", "tmux"]
            packages += ["ark-research", "litellm", "google-genai", "anthropic"]
    checks.append("native-task-runtime-hook")
    versions = {name: importlib.metadata.version(name) for name in packages}
    report = {"host": args.host, "status": "passed", "python": sys.version.split()[0],
              "checks": checks, "versions": versions, "source_tree_sha256": actual}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
