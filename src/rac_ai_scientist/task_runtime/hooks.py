"""Attach execution at native tool boundaries without replacing schedulers."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import sys
import inspect


def attest(runtime, host, function):
    from ..benchmarks.base import digest
    source = Path(inspect.getsourcefile(function))
    runtime.request("attest", {"host": host, "native_execution_source_sha256": digest(source)})


def install_agent_laboratory(runtime):
    import tools
    original = tools.execute_code
    attest(runtime, "agent_laboratory", original)

    def execute_code(code_str, timeout=600, MAX_LEN=1000):
        # Native execute_code prepends `from utils import *`, which imports
        # the thick host environment. Scientific snippets instead execute
        # unchanged in the capsule runtime and install their own dependencies.
        result = runtime.execute("python3 -c " + shlex.quote(code_str), timeout=timeout)
        output = result["stdout"] + result["stderr"]
        if result["exit_code"]:
            output += f"\n[CODE EXECUTION ERROR]: exit {result['exit_code']}"
        return output

    # ai_lab_repo imports functions by value into several native modules.
    # Replace every existing reference to the original, then tools itself,
    # so later imports also receive the adapter.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "execute_code", None) is original:
            module.execute_code = execute_code


def install_evo_scientist(runtime):
    from EvoScientist.backends import CustomSandboxBackend, ExecuteResponse

    # Preserve the public execute() validation path. Override only its final
    # process runner; inherited async execution reaches the same method.
    def execute_prepared(self, command, *, timeout=None):
        result = runtime.execute(command, cwd=str(self.cwd), timeout=self._default_timeout if timeout is None else timeout)
        return ExecuteResponse(output=result["stdout"] + result["stderr"],
                               exit_code=result["exit_code"], truncated=result["truncated"])

    if not hasattr(CustomSandboxBackend, "_execute_prepared_command"):
        raise RuntimeError("unsupported EvoScientist execution API; refusing local fallback")
    attest(runtime, "evo_scientist", CustomSandboxBackend._execute_prepared_command)
    CustomSandboxBackend._execute_prepared_command = execute_prepared


def openhands_interpreter() -> str:
    executable = shutil.which("openhands")
    if not executable:
        raise RuntimeError("OpenHands Python console entrypoint not found")
    with open(executable, "rb") as stream:
        first = stream.readline(1024).decode("utf-8", errors="replace").strip()
    if first.startswith("#!"):
        interpreter = first[2:]
        if Path(interpreter).is_file() and "python" in Path(interpreter).name:
            return interpreter
    raise RuntimeError("runtime adapter requires a Python-installed OpenHands CLI, not a standalone binary")


def install_ark(cli_class, runtime):
    interpreter = openhands_interpreter()
    original_command = cli_class.build_command
    original_env = cli_class.build_env
    if getattr(original_command, "_rac_runtime", False):
        raise RuntimeError("ARK runtime already installed; use one episode per process")

    def build_command(self, prompt, path_boundary, code_dir):
        original = original_command(self, prompt, path_boundary, code_dir)
        return [interpreter, "-m", "rac_ai_scientist.task_runtime.openhands_entry", *original[1:]]

    def build_env(self, code_dir=None):
        env = original_env(self, code_dir)
        env["RAC_TASK_RUNTIME_URL"] = runtime.url
        env["RAC_TASK_RUNTIME_TOKEN"] = runtime.token
        package_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = package_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    build_command._rac_runtime = True
    cli_class.build_command, cli_class.build_env = build_command, build_env
