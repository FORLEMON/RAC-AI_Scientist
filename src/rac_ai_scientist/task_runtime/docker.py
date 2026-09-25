from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
import time
import uuid

from ..benchmarks.base import safe_file, write_json
from .client import RuntimeClient

# Executed by the base image's interpreter, not by a workspace script. Kill
# the whole session on timeout, including children inheriting output pipes.
COMMAND_RUNNER = '''import json, os, signal, subprocess, sys, tempfile
p = json.load(sys.stdin)
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
    proc = subprocess.Popen(["/bin/bash", "-c", p["command"]], cwd=p["cwd"],
        stdout=out, stderr=err, stdin=subprocess.DEVNULL, start_new_session=True)
    timed_out = False
    try:
        proc.wait(timeout=p["timeout"])
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        proc.wait()
    out.seek(0); err.seek(0)
    stdout=out.read(1048577); stderr=err.read(1048577)
    print(json.dumps(dict(stdout=stdout[:1048576].decode("utf-8", "replace"),
        stderr=stderr[:1048576].decode("utf-8", "replace"),
        exit_code=124 if timed_out else proc.returncode, timed_out=timed_out,
        truncated=len(stdout)>1048576 or len(stderr)>1048576)))
'''


class DockerTaskRuntime:
    """One fresh dependency environment per episode, outside the host image.

    The controller is the only component that invokes Docker. No socket,
    private reference tree, host environment or credentials is mounted into
    the task container. Host tools use the returned RuntimeClient.
    """
    def __init__(self, workspace: Path, image: str, log_dir: Path, *,
                 wall_seconds: float, memory: str = "8g", cpus: float = 4,
                 bind: str = "127.0.0.1", port: int = 0, token: str | None = None,
                 advertised_url: str | None = None):
        if os.name != "posix":
            raise RuntimeError("task runtime requires a Linux Docker controller")
        if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image):
            raise ValueError("task image must be frozen by image ID or registry sha256 digest")
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("runtime wall budget must be positive")
        self.workspace = workspace.resolve()
        self.log_dir = log_dir.resolve()
        if self.log_dir.is_relative_to(self.workspace):
            raise ValueError("runtime logs must be outside the agent workspace")
        self.image, self.memory, self.cpus = image, memory, cpus
        self.name = "rac-task-" + uuid.uuid4().hex
        self.bind, self.port = bind, port
        self.token = token or secrets.token_urlsafe(32)
        self.advertised_url = advertised_url
        self.deadline = time.monotonic() + wall_seconds
        self.server = None
        self.timer = None
        self.lock = threading.Lock()
        self.calls = 0
        self.active = False

    def start(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        command = ["docker", "run", "--detach", "--name", self.name,
            "--init", "--memory", self.memory, "--cpus", str(self.cpus), "--pids-limit", "256",
            "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
            "--cap-add", "FOWNER", "--cap-add", "DAC_OVERRIDE", "--network", "bridge",
            "--mount", f"type=bind,source={self.workspace},target={self.workspace}",
            "--workdir", str(self.workspace), "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint", "/usr/local/bin/python3", self.image,
            "-c", "import time; time.sleep(10**9)"]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
            inspected = subprocess.run(["docker", "inspect", self.name], check=True, capture_output=True, text=True, timeout=20)
            image_id = json.loads(inspected.stdout)[0]["Image"]
            inventory = subprocess.run(["docker", "exec", self.name, "/usr/local/bin/python3", "-I", "-c",
                "import importlib.metadata as m,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in m.distributions())))"],
                check=True, capture_output=True, text=True, timeout=30)
            packages = json.loads(inventory.stdout)
            if any(name.lower() not in {"pip", "setuptools", "wheel"} for name, _ in packages):
                raise ValueError("task runtime image contains preinstalled non-bootstrap Python packages")
            owner = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + owner.token):
                        self.send_error(403)
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 8 * 1024 * 1024:
                            raise ValueError("invalid request size")
                        payload = json.loads(self.rfile.read(length))
                        with owner.lock:
                            result = owner.dispatch(self.path, payload)
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}
                    body = json.dumps(result, allow_nan=False).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            self.server = ThreadingHTTPServer((self.bind, self.port), Handler)
            self.server.daemon_threads = True
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            self.provenance = {"kind": "docker", "image": self.image, "image_id": image_id,
                               "workspace": str(self.workspace), "memory": self.memory, "cpus": self.cpus,
                               "initial_python_packages": packages, "bootstrap_only": True,
                               "execution_protocol": "synchronous-bash-v1"}
            write_json(self.log_dir / "runtime.json", self.provenance)
            self.timer = threading.Timer(max(0, self.deadline - time.monotonic()), self._remove_container)
            self.timer.daemon = True
            self.timer.start()
            url = self.advertised_url or f"http://{self.bind}:{self.server.server_port}"
            return RuntimeClient(url, self.token)
        except Exception:
            self.close()
            raise

    def dispatch(self, operation: str, payload: dict):
        if operation == "/info":
            return self.provenance
        if operation == "/begin":
            if self.active:
                raise ValueError("runtime already assigned to an episode; start a fresh container")
            duration = float(payload["wall_seconds"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("invalid episode wall budget")
            self.deadline = min(self.deadline, time.monotonic() + duration)
            if self.deadline <= time.monotonic():
                raise TimeoutError("runtime expired before episode start")
            self.timer.cancel()
            self.timer = threading.Timer(self.deadline - time.monotonic(), self._remove_container)
            self.timer.daemon = True
            self.timer.start()
            self.active = True
            self.provenance.update(episode_id=payload["episode_id"], task_spec_sha256=payload["task_spec_sha256"],
                                   lifecycle_wall_seconds=duration)
            write_json(self.log_dir / "runtime.json", self.provenance)
            return self.provenance
        if not self.active:
            raise RuntimeError("runtime is not assigned to an episode")
        if operation == "/attest":
            if payload.get("host") not in {"ark", "agent_laboratory", "evo_scientist"}:
                raise ValueError("unknown host attestation")
            write_json(self.log_dir / (payload["host"] + ".hook.json"), payload)
            return {"ok": True}
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("task runtime lifecycle budget exhausted")
        if operation in {"/read", "/write"}:
            path = safe_file(self.workspace, payload["path"])
            if operation == "/read":
                if path.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("file exceeds 8 MiB")
                return {"content": path.read_text(encoding="utf-8")}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload["content"], encoding="utf-8")
            return {"ok": True}
        if operation != "/execute":
            raise ValueError("unknown runtime operation")
        command = payload["command"]
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be nonempty text")
        timeout = float(payload.get("timeout", 600))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("command timeout must be positive")
        timeout = min(timeout, remaining)
        cwd = Path(payload.get("cwd") or self.workspace)
        cwd = cwd if cwd.is_absolute() else self.workspace / cwd
        if not cwd.resolve().is_relative_to(self.workspace):
            raise ValueError("command cwd escapes workspace")
        self.calls += 1
        call_id = self.calls
        write_json(self.log_dir / f"{call_id:06d}.request.json", {"command": command, "cwd": str(cwd), "timeout": timeout})
        try:
            process = subprocess.run(["docker", "exec", "--interactive", self.name,
                "/usr/local/bin/python3", "-I", "-c", COMMAND_RUNNER],
                input=json.dumps({"command": command, "cwd": str(cwd), "timeout": timeout}),
                capture_output=True, text=True, timeout=timeout + 15)
            if process.returncode:
                raise RuntimeError(f"task container exec failed ({process.returncode}): {process.stderr[-2000:]}")
            result = json.loads(process.stdout)
        except Exception as exc:
            write_json(self.log_dir / f"{call_id:06d}.result.json", {"error": f"{type(exc).__name__}: {exc}"})
            # A transport timeout leaves command completion uncertain. Remove
            # the entire task container rather than leave an orphan process.
            self._remove_container()
            raise
        write_json(self.log_dir / f"{call_id:06d}.result.json", result)
        return result

    def _remove_container(self):
        try:
            subprocess.run(["docker", "rm", "--force", self.name], capture_output=True, timeout=30)
        except FileNotFoundError:
            pass

    def close(self):
        if self.timer:
            self.timer.cancel()
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self._remove_container()
