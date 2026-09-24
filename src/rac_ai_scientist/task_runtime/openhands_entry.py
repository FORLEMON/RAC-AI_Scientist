"""OpenHands console wrapper: install remote execution before CLI imports.

The SDK interfaces are checked at startup. An unsupported SDK fails closed;
it must never silently execute a CORE capsule in the host environment.
"""
from __future__ import annotations

from importlib.metadata import entry_points
import os
from pathlib import Path
import subprocess

from .client import RuntimeClient


def install(runtime):
    # file:// upload-pack intentionally strips GIT_CONFIG_COUNT. Trust only
    # this controller-mounted, read-only public dependency in the container's
    # global config, so its Windows UID mapping does not block local cloning.
    if Path('/opt/openhands-extensions/.git').is_dir():
        subprocess.run(['git','config','--global','--add','safe.directory','/opt/openhands-extensions/.git'],check=True)
    from openhands.tools.terminal.impl import TerminalExecutor
    from openhands.tools.terminal.definition import TerminalObservation
    from openhands.sdk.workspace.local import LocalWorkspace
    from openhands.sdk.workspace.models import CommandResult

    for cls, names in ((TerminalExecutor, ("__call__", "close")), (LocalWorkspace, ("execute_command",))):
        if any(not hasattr(cls, name) for name in names):
            raise RuntimeError("unsupported OpenHands execution API; local fallback is forbidden")
    from .hooks import attest
    attest(runtime, "ark", TerminalExecutor.__call__)

    def initialize(self, working_dir, **kwargs):
        self._working_dir = str(working_dir)
        self._pool = None
        self._session = None
        self.full_output_save_dir = None

    def execute(self, action, conversation=None, **kwargs):
        if getattr(action, "is_input", False) or not action.command.strip():
            return TerminalObservation.from_text(text="Task runtime commands are synchronous. No interactive process is pending.",
                command=action.command, exit_code=1, is_error=True)
        result = runtime.execute(action.command, cwd=self._working_dir, timeout=action.timeout or 600)
        return TerminalObservation.from_text(text=result["stdout"] + result["stderr"],
            command=action.command, exit_code=result["exit_code"], is_error=result["exit_code"] != 0)

    def workspace_execute(self, command, cwd=None, timeout=30):
        result = runtime.execute(command, cwd=str(cwd or self.working_dir), timeout=timeout)
        return CommandResult(command=command, exit_code=result["exit_code"], stdout=result["stdout"],
                             stderr=result["stderr"], timeout_occurred=result["timed_out"])

    TerminalExecutor.__init__ = initialize
    TerminalExecutor.__call__ = execute
    TerminalExecutor.close = lambda self: None
    LocalWorkspace.execute_command = workspace_execute


def main():
    runtime = RuntimeClient(os.environ["RAC_TASK_RUNTIME_URL"], os.environ["RAC_TASK_RUNTIME_TOKEN"])
    runtime.info()
    install(runtime)
    entries = list(entry_points(group="console_scripts", name="openhands"))
    if len(entries) != 1:
        raise RuntimeError("ambiguous OpenHands console entrypoint")
    return entries[0].load()()


if __name__ == "__main__":
    raise SystemExit(main())
