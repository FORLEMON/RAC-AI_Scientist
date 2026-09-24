"""Offline acceptance checks and package inventory for local Docker hosts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGES = {"ark": "rac-local/ark:validated",
          "agent_laboratory": "rac-local/agent-laboratory:validated",
          "evo_scientist": "rac-local/evo-scientist:validated"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=tuple(IMAGES), action="append")
    parser.add_argument("--task-runtime", action="store_true")
    args = parser.parse_args()
    report_dir = ROOT / "runs/environment-check" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir.mkdir(parents=True)
    summary = {"checked_at": datetime.now(timezone.utc).isoformat(), "hosts": {}}
    for host in args.host or IMAGES:
        image = IMAGES[host]
        inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image], text=True))[0]
        image_id = inspected["Id"]
        command = ["docker", "run", "--rm", "--network", "none",
                   "--mount", f"type=bind,source={ROOT / 'scripts'},target=/verification,readonly",
                   "--mount", f"type=bind,source={report_dir},target=/reports",
                   "--entrypoint", "python", image_id,
                   "/verification/probe_host_environment.py", "--host", host,
                   "--output", f"/reports/{host}.json"]
        print(f"Checking {host} ({image_id})", flush=True)
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        (report_dir / f"{host}.log").write_text(process.stdout, encoding="utf-8")
        if process.returncode:
            print(process.stdout)
            raise SystemExit(f"{host} failed; report: {report_dir}")
        report = json.loads((report_dir / f"{host}.json").read_text())
        report.update(image=image, image_id=image_id, image_size_bytes=inspected["Size"])
        summary["hosts"][host] = report
        frozen = subprocess.check_output(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
                                          image_id, "-m", "pip", "freeze"], text=True)
        (report_dir / f"{host}-packages.txt").write_text(frozen, encoding="utf-8")
        if host == "ark":
            frozen = subprocess.check_output(["docker", "run", "--rm", "--network", "none", "--entrypoint", "cat",
                                              image_id, "/opt/integration/openhands-environment.txt"], text=True)
            (report_dir / "openhands-packages.txt").write_text(frozen, encoding="utf-8")
        print(f"PASS: {host}", flush=True)
    if args.task_runtime:
        image = json.loads(subprocess.check_output(["docker", "image", "inspect", "rac-local/task-runtime:validated"], text=True))[0]["Id"]
        result = subprocess.run([sys.executable, str(ROOT / "scripts/smoke_task_runtime.py"), "--image", image],
                                capture_output=True, text=True, timeout=120)
        (report_dir / "task-runtime.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        result.check_returncode()
        summary["task_runtime"] = json.loads(result.stdout)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Reports: {report_dir}", flush=True)


if __name__ == "__main__":
    main()
