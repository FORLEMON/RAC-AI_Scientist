"""Run with WSL Python on the Docker controller, not inside a host container."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rac_ai_scientist.queue_runner import main

if __name__ == '__main__':
    main()
