import unittest
from pathlib import Path

from rac_ai_scientist.matrix import expand_matrix


class MatrixTests(unittest.TestCase):
    def test_cartesian_product_is_deterministic(self):
        config = {
            "experiment_id": "pilot",
            "hosts": ["ark", "data_to_paper"],
            "conditions": ["N0", "R5"],
            "tasks": ["Task_001"],
            "seeds": [7],
            "repeats": 2,
            "model": {"name": "model"},
            "budget": {"max_hops": 3},
            "paths": {"benchmark": "bench"},
        }
        rows = list(expand_matrix(config, Path("/repo/configs/experiment.json")))
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["episode_id"], "pilot__Task_001__ark__N0__s7__r0")
        self.assertEqual(rows[-1]["ordinal"], 8)


if __name__ == "__main__":
    unittest.main()
