"""Linux/Docker acceptance smoke; no model calls or task dependency installs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rac_ai_scientist.task_runtime import DockerTaskRuntime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="rac-runtime-smoke-") as raw:
        root = Path(raw)
        runtime = DockerTaskRuntime(root / "workspace", args.image, root / "logs", wall_seconds=60)
        try:
            client = runtime.start()
            client.request("begin", {"episode_id": "smoke", "task_spec_sha256": "smoke", "wall_seconds": 60})
            info = client.info()
            assert info["bootstrap_only"]
            client.write_file("data/input.txt", "42")
            result = client.execute("test ! -S /var/run/docker.sock && cat data/input.txt > output.txt", timeout=5)
            assert result["exit_code"] == 0, result
            assert client.read_file("output.txt") == "42"
            assert (root / "workspace/output.txt").read_text() == "42"
            result = client.execute("python3 -c 'import numpy'", timeout=5)
            assert result["exit_code"] != 0, "task image unexpectedly includes NumPy"
            result = client.execute("(sleep 2; echo orphan > orphan.txt) & sleep 30", timeout=.2)
            assert result["timed_out"] and result["exit_code"] == 124, result
            result = client.execute("sleep 3; test ! -e orphan.txt", timeout=5)
            assert result["exit_code"] == 0, "child process survived timeout"
            try:
                client.request("begin", {"episode_id": "reuse", "task_spec_sha256": "reuse", "wall_seconds": 60})
            except RuntimeError:
                pass
            else:
                raise AssertionError("runtime accepted a second episode")
        finally:
            runtime.close()
        check = subprocess.run(["docker", "inspect", runtime.name], capture_output=True)
        assert check.returncode != 0, "task container leaked after cleanup"
        print(json.dumps({"status": "passed", "image": args.image,
            "checks": ["bootstrap-only", "shared-files", "no-docker-socket", "timeout-kills-children", "no-reuse", "cleanup"]}))


if __name__ == "__main__":
    main()
