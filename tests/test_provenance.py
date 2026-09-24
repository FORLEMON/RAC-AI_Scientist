import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.provenance import expected_tree_hash, tree_hash


class ProvenanceTests(unittest.TestCase):
    def test_runtime_variants_require_explicit_lock_and_packaged_location(self):
        spec = {"tree_sha256": "canonical", "runtime_tree_sha256": "server",
                "runtime_tree_sha256s": ["with-submodule"], "snapshot_tree_sha256": "snapshot"}
        for actual in ("server", "with-submodule"):
            self.assertEqual(expected_tree_hash(spec, actual, packaged=True), actual)
            self.assertNotEqual(expected_tree_hash(spec, actual), actual)
        for actual in ("tampered", "canonical"):
            self.assertNotEqual(expected_tree_hash(spec, actual, packaged=True), actual)
        self.assertEqual(expected_tree_hash(spec, "snapshot", snapshot=True), "snapshot")
        self.assertEqual(expected_tree_hash(spec, "canonical"), "canonical")

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
