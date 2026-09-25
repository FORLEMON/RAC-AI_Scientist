import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from rac_ai_scientist.task_runtime.docker import COMMAND_RUNNER, DockerTaskRuntime
from rac_ai_scientist.task_runtime.hooks import install_agent_laboratory, install_evo_scientist, install_ark
from rac_ai_scientist.task_runtime.openhands_entry import install as install_openhands


class FakeRuntime:
    url, token = "http://controller:8765", "test-token"

    def __init__(self):
        self.calls = []

    def request(self, operation, payload):
        self.calls.append((operation, payload))
        return {"ok": True}

    def execute(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return {"stdout": "container result", "stderr": "", "exit_code": 0, "timed_out": False, "truncated": False}


class NativeHookTests(unittest.TestCase):
    def test_agent_laboratory_imported_aliases_use_runtime(self):
        tools = ModuleType("tools")
        solver = ModuleType("mlesolver")
        def local_execute(code, timeout=600, MAX_LEN=1000):
            raise AssertionError("must never execute task code in host")
        tools.execute_code = solver.execute_code = local_execute
        runtime = FakeRuntime()
        with patch.dict(sys.modules, {"tools": tools, "mlesolver": solver}):
            install_agent_laboratory(runtime)
            self.assertEqual(solver.execute_code("print('hi')", timeout=7), "container result")
            self.assertEqual(runtime.calls[-1][1]["timeout"], 7)
            self.assertNotIn("from utils", runtime.calls[-1][0])
            self.assertIs(tools.execute_code, solver.execute_code)

    def test_evo_preserves_validator_and_replaces_process_runner(self):
        module = ModuleType("EvoScientist.backends")
        class Backend:
            cwd, _default_timeout = "/workspace", 30
            def execute(self, command):
                return self._execute_prepared_command("validated " + command)
            def _execute_prepared_command(self, command, timeout=None):
                raise AssertionError("local execution forbidden")
        module.CustomSandboxBackend = Backend
        module.ExecuteResponse = SimpleNamespace
        runtime = FakeRuntime()
        with patch.dict(sys.modules, {"EvoScientist.backends": module}):
            install_evo_scientist(runtime)
            self.assertEqual(Backend().execute("test").output, "container result")
        self.assertEqual(runtime.calls[-1][0], "validated test")

    def test_ark_uses_same_openhands_cli_instrumented_entrypoint(self):
        class CLI:
            def build_command(self, prompt, path_boundary, code_dir):
                return ["openhands", "--headless", "--json", "-t", prompt]
            def build_env(self, code_dir=None):
                return {"LLM_MODEL": "fake"}
        with patch("rac_ai_scientist.task_runtime.hooks.openhands_interpreter", return_value="/venv/bin/python"):
            install_ark(CLI, FakeRuntime())
            command = CLI().build_command("task", "boundary", Path("/workspace"))
        self.assertEqual(command[:3], ["/venv/bin/python", "-m", "rac_ai_scientist.task_runtime.openhands_entry"])
        self.assertEqual(command[3:], ["--headless", "--json", "-t", "task"])
        self.assertEqual(CLI().build_env()["RAC_TASK_RUNTIME_URL"], "http://controller:8765")

    def test_openhands_terminal_and_workspace_commands_both_use_runtime(self):
        class Terminal:
            def __call__(self, action):
                raise AssertionError("local terminal forbidden")
            def close(self):
                pass
        class Workspace:
            working_dir = "/workspace"
            def execute_command(self, *args, **kwargs):
                raise AssertionError("local execution forbidden")
        observation = SimpleNamespace(from_text=lambda **kwargs: SimpleNamespace(**kwargs))
        modules = {"openhands.tools.terminal.impl": SimpleNamespace(TerminalExecutor=Terminal),
                   "openhands.tools.terminal.definition": SimpleNamespace(TerminalObservation=observation),
                   "openhands.sdk.workspace.local": SimpleNamespace(LocalWorkspace=Workspace),
                   "openhands.sdk.workspace.models": SimpleNamespace(CommandResult=SimpleNamespace)}
        runtime = FakeRuntime()
        with patch.dict(sys.modules, modules):
            install_openhands(runtime)
            terminal = Terminal(working_dir="/workspace")
            self.assertEqual(terminal(SimpleNamespace(command="python main.py", timeout=2, is_input=False)).exit_code, 0)
            self.assertEqual(Workspace().execute_command("pwd").stdout, "container result")
            terminal(SimpleNamespace(command="", timeout=2, is_input=True))
        self.assertEqual(len([x for x in runtime.calls if x[0] != "attest"]), 2)


class ControllerBoundaryTests(unittest.TestCase):
    def test_rejects_invalid_cpuset_before_start(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace, logs = Path(directory) / 'workspace', Path(directory) / 'logs'
            with patch('rac_ai_scientist.task_runtime.docker.os.name', 'posix'):
                with self.assertRaisesRegex(ValueError, 'CPU set'):
                    DockerTaskRuntime(workspace, 'sha256:' + 'a' * 64, logs,
                                      wall_seconds=1, cpuset_cpus='2-7;rm')

    def test_file_api_rejects_escape_and_shares_workspace(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = object.__new__(DockerTaskRuntime)
            runtime.active = True
            runtime.workspace, runtime.deadline = Path(raw), time.monotonic() + 30
            runtime.dispatch("/write", {"path": "outputs/result.txt", "content": "42"})
            self.assertEqual(runtime.dispatch("/read", {"path": "outputs/result.txt"})["content"], "42")
            with self.assertRaises(ValueError):
                runtime.dispatch("/read", {"path": "../private.json"})

    def test_execute_rejects_bad_cwd_and_timeout_without_starting_a_process(self):
        with tempfile.TemporaryDirectory() as raw:
            runtime = object.__new__(DockerTaskRuntime)
            runtime.active = True
            runtime.workspace, runtime.deadline = Path(raw), time.monotonic() + 30
            for params in ({"cwd": ".."}, {"timeout": -1}, {"timeout": float("nan")}):
                with self.subTest(params=params), patch("subprocess.run") as run, self.assertRaises(ValueError):
                    runtime.dispatch("/execute", {"command": "pwd", **params})
                run.assert_not_called()

    @unittest.skipUnless(os.name == "posix" and Path("/bin/bash").exists(), "Linux Bash process-group smoke")
    def test_command_timeout_kills_child_processes(self):
        with tempfile.TemporaryDirectory() as raw:
            command = "(sleep 1; echo leaked > survived.txt) & sleep 30"
            response = subprocess.run([sys.executable, "-c", COMMAND_RUNNER], input=json.dumps(
                {"command": command, "cwd": raw, "timeout": .1}), capture_output=True, text=True, timeout=5)
            self.assertTrue(json.loads(response.stdout)["timed_out"])
            time.sleep(1.1)
            self.assertFalse((Path(raw) / "survived.txt").exists())
