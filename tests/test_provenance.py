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

    def test_generated_package_metadata_does_not_change_runtime_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "source.py").write_text("value = 1\n", encoding="utf-8")
            before = tree_hash(root)
            metadata = root / "package.egg-info"
            metadata.mkdir()
            (metadata / "PKG-INFO").write_text("generated", encoding="utf-8")
            self.assertEqual(tree_hash(root), before)
