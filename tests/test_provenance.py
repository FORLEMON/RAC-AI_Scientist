import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.provenance import tree_hash


class ProvenanceTests(unittest.TestCase):
    def test_tree_hash_depends_on_paths_and_contents(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a").write_text("same", encoding="utf-8")
            first = tree_hash(root)[0]
            (root / "a").rename(root / "b")
            second = tree_hash(root)[0]
            self.assertNotEqual(first, second)
