import json
import tempfile
import unittest
from pathlib import Path

from rac_ai_scientist.provenance import tree_hash


class ProvenanceTests(unittest.TestCase):
    def test_ai_researcher_runtime_pin_uses_reviewed_native_revision(self):
        lock = json.loads((Path(__file__).parents[1] / "upstream.lock.json").read_text(encoding="utf-8"))
        spec = lock["upstreams"]["ai_researcher"]
        self.assertEqual(spec["revision"], "e9c3294cee5dc85ce70cb1e30b7a2f86f69d0502")
        self.assertEqual(spec["runtime_tree_sha256"], "d38fddf6197d888a951de7c09165dbda0ecdd9a92e569225ae292f6d95002d06")

    def test_auto_runtime_pin_uses_clean_647_archive(self):
        lock = json.loads((Path(__file__).parents[1] / "upstream.lock.json").read_text(encoding="utf-8"))
        spec = lock["upstreams"]["auto_research_claw"]
        self.assertEqual(spec["revision"], "7c264656d895ed1f41acd662b9e6195858bf5e37")
        self.assertEqual(spec["runtime_tree_sha256"], "01b369c6d0f5e6a43a1a2517b377fe0a0b24e09a5bb123358a67f08eaa9dbd0a")

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
