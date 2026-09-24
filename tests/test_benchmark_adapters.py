import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rac_ai_scientist.benchmarks import get_adapter, load_prepared, copy_prepared
from rac_ai_scientist.benchmarks.base import BoundaryError, InvalidSubmission, TaskSpec, digest, read_json, write_json
from rac_ai_scientist.benchmarks.discoverybench import REVISION, private_reference
from rac_ai_scientist.benchmarks.episode import score_episode
from rac_ai_scientist.benchmarks.scoring import score_core, score_discovery
from rac_ai_scientist.benchmarks.upstream import core_harness, definitions
from rac_ai_scientist.matrix import expand_matrix
from rac_ai_scientist.policy import SharedPolicy
from rac_ai_scientist.schemas import Budget, CapabilityCard, Checkpoint, Issue, InvocationResult, Usage, Action, Verdict

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "upstreams" / "hal_harness"
DISCOVERY = ROOT / "upstreams" / "discoverybench"


def discovery_fixture(source):
    topic = source / "discoverybench/real/train/topic"
    topic.mkdir(parents=True)
    (topic / "data.csv").write_text("x,y\n1,2\n2,4\n", encoding="utf-8")
    metadata = {"domain_knowledge": "SECRET_DOMAIN", "workflow": "SECRET_WORKFLOW",
        "hypotheses": {"main": "SECRET_HYPOTHESIS"},
        "datasets": [{"name": "data.csv", "description": "Measurements",
                      "columns": {"raw": [{"name": "x", "description": "Input"}],
                                  "derived": [{"name": "SECRET_DERIVED"}]}}],
        "queries": [[{"qid": 7, "question": "What relates x and y?", "true_hypothesis": "SECRET_ANSWER"}],
                    [{"qid": 8, "question": "Second question", "true_hypothesis": "OTHER_ANSWER"}]]}
    write_json(topic / "metadata_0.json", metadata)
    return metadata


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        discovery_fixture(self.source)
        self.workspace = self.root / "prepared"
        self.adapter = get_adapter("discoverybench")
        self.spec = self.adapter.prepare(self.source, self.workspace, task_id="topic/metadata_0/0/0", split="train")

    def test_discovery_public_whitelist_and_unique_nested_query_identity(self):
        combined = "".join(path.read_text(encoding="utf-8") for path in self.workspace.rglob("*") if path.is_file())
        self.assertNotIn("SECRET", combined)
        self.assertNotIn("OTHER_ANSWER", combined)
        self.assertNotIn("Second question", combined)
        self.assertEqual(load_prepared(self.workspace), self.spec)
        other = self.adapter.prepare(self.source, self.root / "other", task_id="topic/metadata_0/1/0", split="train")
        self.assertNotEqual(self.spec.task_id, other.task_id)
        self.assertEqual(other.objective, "Second question")

    def test_secret_env_is_never_copied(self):
        (self.workspace / ".env").write_text("SHAREDNET_INVITE=secret", encoding="utf-8")
        destination = self.root / "episode"
        copy_prepared(self.workspace, destination)
        self.assertFalse((destination / ".env").exists())

    def test_detects_tampering_and_undeclared_inputs(self):
        (self.workspace / "target_study.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(BoundaryError):
            load_prepared(self.workspace)
        (self.workspace / "target_study.json").unlink()
        (self.workspace / "data/data.csv").write_text("tampered", encoding="utf-8")
        with self.assertRaises(BoundaryError):
            load_prepared(self.workspace)

    def test_rejects_portable_path_traversal(self):
        for name in ("../secret.csv", "C:/private.txt", "a\\b.csv", "a/../b", "a//b"):
            payload = read_json(self.source / "discoverybench/real/train/topic/metadata_0.json")
            payload["datasets"][0]["name"] = name
            write_json(self.source / "discoverybench/real/train/topic/metadata_0.json", payload)
            with self.subTest(name=name), self.assertRaises(BoundaryError):
                self.adapter.prepare(self.source, self.root / "bad", task_id=self.spec.task_id, split="train")

    def test_json_submission_cannot_be_replaced_by_markdown(self):
        (self.workspace / "report/report.md").write_text("Complete report", encoding="utf-8")
        with self.assertRaises(InvalidSubmission):
            self.adapter.read_submission(self.workspace, self.spec)
        write_json(self.workspace / "discovery_result.json", {"hypothesis": "y increases", "workflow": "fit y~x"})
        self.assertEqual(self.adapter.read_submission(self.workspace, self.spec)["workflow"], "fit y~x")

    def test_invalid_json_is_not_repaired(self):
        for value in ('{"hypothesis":"x", "hypothesis":"y", "workflow":"z"}',
                      '{"hypothesis":"x","workflow":""}', '{"hypothesis":NaN}',
                      '{"hypothesis":"x","workflow":"y","n":1e400}', '```json\n{}\n```'):
            (self.workspace / "discovery_result.json").write_text(value, encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(InvalidSubmission):
                self.adapter.read_submission(self.workspace, self.spec)

    def test_test_answer_key_uses_qid_not_query_position(self):
        payload = read_json(self.source / "discoverybench/real/train/topic/metadata_0.json")
        del payload["queries"][0][0]["true_hypothesis"]
        write_json(self.source / "discoverybench/real/test/topic/metadata_0.json", payload)
        (self.source / "eval").mkdir()
        (self.source / "eval/answer_key_real.csv").write_text("dataset,metadataid,query_id,gold_hypo\ntopic,0,7,correct\ntopic,0,0,wrong\n", encoding="utf-8")
        spec = TaskSpec.from_dict({**self.spec.to_dict(), "split": "test"})
        self.assertEqual(private_reference(self.source, spec)[0], "correct")

    def test_failed_submission_produces_null_score_without_loading_private_scorer(self):
        episode = self.root / "episode"
        copy_prepared(self.workspace, episode / "workspace")
        write_json(episode / "task_spec.json", self.spec.to_dict())
        write_json(episode / "episode.json", {"benchmark_id": "discoverybench", "task_id": self.spec.task_id,
            "task_spec_sha256": digest(episode / "task_spec.json")})
        with patch("rac_ai_scientist.benchmarks.upstream.assert_revision") as verify:
            self.assertEqual(score_episode(episode, self.source), 2)
            verify.assert_not_called()
        result = read_json(episode / "score.json")
        self.assertEqual(result["status"], "invalid_submission")
        self.assertIsNone(result["total_score"])

    def test_manifest_cannot_whitelist_credentials(self):
        with self.assertRaises(BoundaryError):
            TaskSpec.from_dict({**self.spec.to_dict(), "public_files": {".env": "0" * 64}})

    def test_workspace_spec_cannot_change_scorer_identity(self):
        episode = self.root / "episode"
        copy_prepared(self.workspace, episode / "workspace")
        write_json(episode / "task_spec.json", self.spec.to_dict())
        write_json(episode / "episode.json", {"benchmark_id": "discoverybench", "task_id": self.spec.task_id,
            "task_spec_sha256": digest(episode / "task_spec.json")})
        write_json(episode / "workspace/task_spec.json", {"benchmark_id": "attacker"})
        with patch("rac_ai_scientist.benchmarks.upstream.assert_revision") as verify:
            self.assertEqual(score_episode(episode, self.source), 2)
            verify.assert_not_called()
        self.assertEqual(read_json(episode / "score.json")["status"], "invalid_submission")
        write_json(episode / "task_spec.json", {**self.spec.to_dict(), "objective": "tampered"})
        self.assertEqual(score_episode(episode, self.source), 2)
        self.assertEqual(read_json(episode / "score.json")["status"], "scorer_failed")

    def test_scoring_worker_failure_remains_null_and_loses_agent_credentials(self):
        episode = self.root / "episode"
        copy_prepared(self.workspace, episode / "workspace")
        write_json(episode / "workspace/discovery_result.json", {"hypothesis": "x relates y", "workflow": "fit"})
        write_json(episode / "task_spec.json", self.spec.to_dict())
        write_json(episode / "episode.json", {"benchmark_id": "discoverybench", "task_id": self.spec.task_id,
            "task_spec_sha256": digest(episode / "task_spec.json")})
        def fail_worker(command, **kwargs):
            self.assertNotIn("AGENT_API_KEY", kwargs["env"])
            self.assertNotIn("SHAREDNET_INVITE", kwargs["env"])
            self.assertNotEqual(Path(kwargs["cwd"]), episode / "workspace")
            self.assertNotIn(str(episode / "workspace"), command)
            return SimpleNamespace(returncode=7, stdout="worker began", stderr="worker unavailable")
        with patch("rac_ai_scientist.benchmarks.upstream.assert_revision"), \
             patch("rac_ai_scientist.benchmarks.episode.subprocess.run", side_effect=fail_worker), \
             patch.dict("os.environ", {"AGENT_API_KEY": "private", "SHAREDNET_INVITE": "private"}):
            self.assertEqual(score_episode(episode, self.source), 2)
        result = read_json(episode / "score.json")
        self.assertIsNone(result["total_score"])
        self.assertIn("worker unavailable", result["error"])
        self.assertEqual((episode / "scoring/worker.stdout.log").read_text(), "worker began")
        self.assertEqual((episode / "scoring/worker.stderr.log").read_text(), "worker unavailable")


class ProtocolTests(unittest.TestCase):
    def test_core_exact_question_keys_and_types(self):
        spec = TaskSpec("corebench", "capsule", "dev", "answer", "rev", "report.json", questions=("Q?",))
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            for value in ({"Q": 3}, {"Q?": None}, {"Q?": {"answer": 3}}):
                write_json(workspace / "report.json", value)
                with self.assertRaises(InvalidSubmission):
                    get_adapter("corebench").read_submission(workspace, spec)
            write_json(workspace / "report.json", {"Q?": 3})
            self.assertEqual(get_adapter("corebench").read_submission(workspace, spec), {"Q?": 3})

    def test_final_json_contract_is_advisory_and_not_required_during_analysis(self):
        spec = TaskSpec("discoverybench", "topic/metadata_0/0/0", "train", "answer", REVISION, "discovery_result.json")
        cards = [CapabilityCard("run", "experiment", ("experiment",), (), ("outputs/*",), ("result",)),
                 CapabilityCard("finish", "finish", ("finalize",), (), ("discovery_result.json",))]
        cp = Checkpoint("e", 0, "answer", "work", [], [], Budget(1, 100, 100, 5, 100, 5), cards)
        policy = SharedPolicy("R3", task_spec=spec)
        decision = policy.decide(cp)
        self.assertIn("benchmark_submission", [x.kind for x in decision.contract.required_evidence])
        evaluation = policy.evaluate(cp, decision, InvocationResult("finish", "done", [], [], Usage(), metrics={"submission_valid": 0.0}))
        self.assertIs(evaluation.action, Action.REVERIFY)
        self.assertIs(evaluation.verification.verdict, Verdict.REFUTED)
        cp.issues = [Issue("native:run", "native_requirement", "run")]
        self.assertNotIn("benchmark_submission", [x.kind for x in policy.decide(cp).contract.required_evidence])

    def test_matrix_ids_include_benchmark_and_split_without_path_separators(self):
        config = {"experiment_id": "test", "hosts": ["ark", "agent_laboratory", "evo_scientist"],
            "conditions": ["N0", "R1", "R2", "R3"], "tasks": ["topic/metadata_0/0/0"],
            "model": {"name": "model"}, "budget": {}, "paths": {"benchmark": "source"},
            "benchmark": {"id": "discoverybench", "split": "train"}}
        rows = list(expand_matrix(config, Path("/repo/configs/test.json")))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["episode_id"] for r in rows}), 12)
        self.assertTrue(all("discoverybench__train" in r["episode_id"] and "/" not in r["episode_id"] for r in rows))


@unittest.skipUnless((CORE / "hal/benchmarks/corebench.py").exists(), "pinned CORE checkout is optional; bootstrap_benchmarks.py enables this smoke")
class OfficialCoreSmoke(unittest.TestCase):
    def test_official_hard_filter_and_no_answer_leak(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            capsule = root / "capsules/capsule"
            names = ["code/main.py", "data/test.csv", "README.md", "results/answer.txt", "environment/Dockerfile",
                     "REPRODUCING.md", "code/run", "code/run.sh", "data/train.csv", "cache/stuff", "code/model.ckpt", "code/checkpoint1"]
            for name in names:
                path = capsule / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("public" if name in names[:3] else "SECRET", encoding="utf-8")
            write_json(root / "private.json", [{"capsule_id": "capsule", "task_prompt": "compute", "results": [{"Q?": "SECRET_ANSWER"}]}])
            spec = get_adapter("corebench").prepare(CORE, root / "prepared", task_id="capsule", split="dev",
                dataset=root / "private.json", capsules=root / "capsules")
            for name in names[:3]:
                self.assertTrue((root / "prepared" / name).exists())
            for name in names[3:]:
                self.assertFalse((root / "prepared" / name).exists())
            self.assertNotIn("SECRET", "".join(p.read_text(encoding="utf-8") for p in (root / "prepared").rglob("*") if p.is_file()))
            self.assertEqual(spec.questions, ("Q?",))

    def test_official_numeric_string_list_and_vision_grading(self):
        try:
            import scipy
        except ImportError:
            self.skipTest("install benchmark-scoring extra for official numeric scorer smoke")
        with tempfile.TemporaryDirectory() as raw:
            dataset = Path(raw) / "private.json"
            task = {"capsule_id": "capsule", "task_prompt": "compute", "results": [
                {"mean?": 10, "label?": "yes", "fig values?": [1, 2]},
                {"mean?": 11, "label?": "yes", "fig values?": [1, 2]},
                {"mean?": 9, "label?": "yes", "fig values?": [1, 2]}]}
            write_json(dataset, [task])
            spec = TaskSpec("corebench", "capsule", "dev", core_harness(CORE)._construct_prompt(task), "rev", "report.json",
                            questions=tuple(task["results"][0]))
            correct = score_core(CORE, dataset, spec, {"mean?": 10, "label?": "YES", "fig values?": [1, 2]})
            wrong = score_core(CORE, dataset, spec, {"mean?": 1000, "label?": "no", "fig values?": [1]})
            self.assertEqual(correct["accuracy"], 1)
            self.assertEqual(correct["raw"]["capsule"]["correct_vision_answers"], 1)
            self.assertEqual(wrong["accuracy"], 0)
            self.assertEqual(wrong["raw"]["capsule"]["correct_written_answers"], 0)


@unittest.skipUnless((DISCOVERY / "eval/new_eval.py").exists(), "pinned Discovery checkout is optional")
class OfficialDiscoverySmoke(unittest.TestCase):
    def test_official_formula_and_judge_failure_are_distinct(self):
        judge = SimpleNamespace(client=object(), model="fake")
        judge.get_response = lambda *a, **k: {"sub_hypo": [{"text": "x relates y", "context": "all"}]}
        def chat(**kwargs):
            text = kwargs["messages"][-1]["content"]
            payload = {"sizeA": 2, "sizeB": 2, "intersection": 2, "explanation": "equal"} if "different variables found" in text else {"answer": "A) very similar"}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
        judge.chat = chat
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw)
            discovery_fixture(source)
            spec = TaskSpec("discoverybench", "topic/metadata_0/0/0", "train", "What relates x and y?", REVISION, "discovery_result.json")
            # Keep official functions from the verified checkout and synthetic
            # private references separate; no model request is made.
            with patch("rac_ai_scientist.benchmarks.scoring.definitions", side_effect=lambda source, name, ns: definitions(DISCOVERY, name, ns)):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = score_discovery(source, spec, {"hypothesis": "x relates y", "workflow": "fit"}, judge)
                self.assertEqual(result["final_score"], 1)
                judge.chat = lambda **kwargs: (_ for _ in ()).throw(TimeoutError("judge unavailable"))
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
                    score_discovery(source, spec, {"hypothesis": "x relates y", "workflow": "fit"}, judge)
