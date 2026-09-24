"""Exercise the installed OpenHands SDK interfaces without a provider call."""
from probe_host_environment import RuntimeProbe
from rac_ai_scientist.task_runtime.openhands_entry import install
from openhands.tools.terminal.impl import TerminalExecutor
from openhands.tools.terminal.definition import TerminalAction
from openhands.sdk.workspace.local import LocalWorkspace

runtime = RuntimeProbe()
install(runtime)
terminal = TerminalExecutor(working_dir="/tmp")
result = terminal(TerminalAction(command="echo remote-probe", timeout=5))
assert result.exit_code == 0 and len(runtime.calls) == 1
workspace = LocalWorkspace(working_dir="/tmp")
result = workspace.execute_command("echo remote-probe", timeout=5)
assert result.stdout == "remote-runtime-probe" and len(runtime.calls) == 2
assert runtime.attestations
terminal.close()
print("OpenHands terminal/workspace runtime hooks OK")
