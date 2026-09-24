"""Run on the Docker controller; host containers need no Docker socket."""
import argparse
import os
from pathlib import Path
import signal
import threading

from .docker import DockerTaskRuntime


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--wall-seconds", required=True, type=float)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    token = os.environ.get("RAC_TASK_RUNTIME_TOKEN")
    if not token:
        raise ValueError("set RAC_TASK_RUNTIME_TOKEN for controller and host")
    runtime = DockerTaskRuntime(Path(args.workspace), args.image, Path(args.log_dir),
        wall_seconds=args.wall_seconds, bind=args.bind, port=args.port, token=token)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *args: stop.set())
    signal.signal(signal.SIGTERM, lambda *args: stop.set())
    try:
        runtime.start()
        print(f"Task runtime ready on {args.bind}:{args.port}", flush=True)
        stop.wait(args.wall_seconds)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
