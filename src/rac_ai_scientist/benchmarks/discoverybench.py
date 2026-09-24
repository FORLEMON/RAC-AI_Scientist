from __future__ import annotations

import csv
from pathlib import Path
import shutil

from .base import BenchmarkAdapter, TaskSpec, InvalidSubmission, finish_preparation, safe_file, read_json, write_json

REVISION = "c31fcf011e070f021a5f5b906896d0821f6880e8"


def locate(source: Path, task_id: str, split: str):
    """IDs are topic/metadata_N/group-index/query-index (not qid alone)."""
    if split not in {"train", "test"}:
        raise ValueError("DiscoveryBench real split must be train or test")
    parts = task_id.split("/")
    if len(parts) != 4:
        raise ValueError("task-id must be topic/metadata_N/group-index/query-index")
    topic, metadata, group, query = parts
    root = safe_file(source, f"discoverybench/real/{split}/{topic}")
    payload = read_json(safe_file(root, metadata + ".json"))
    gi, qi = int(group), int(query)
    if gi < 0 or qi < 0:
        raise ValueError("query indices must be nonnegative")
    selected = payload["queries"][gi][qi]
    if not isinstance(selected.get("question"), str) or not selected["question"].strip():
        raise ValueError("missing discovery question")
    return root, payload, selected


def public_metadata(payload: dict) -> dict:
    # The no-domain-knowledge track matches the default agent inputs. Never
    # copy hypotheses, workflows, queries, derived columns or entire metadata.
    return {"datasets": [
        {"name": item["name"], "description": item.get("description", ""),
         "columns": {"raw": [{"name": col["name"], "description": col.get("description", "")}
                             for col in item["columns"]["raw"]]}}
        for item in payload["datasets"]
    ]}


class DiscoveryBenchAdapter(BenchmarkAdapter):
    benchmark_id = "discoverybench"

    def prepare(self, source: Path, destination: Path, **options) -> TaskSpec:
        task_id, split = options["task_id"], options["split"]
        root, payload, query = locate(source, task_id, split)
        public = public_metadata(payload)
        files = [(item["name"], safe_file(root, item["name"])) for item in public["datasets"]]
        for _, path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
        destination.mkdir(parents=True, exist_ok=False)
        for name, path in files:
            target = safe_file(destination, "data/" + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        write_json(destination / "metadata.json", public)
        spec = TaskSpec(self.benchmark_id, task_id, split, query["question"],
                        REVISION, "discovery_result.json", profile="real-no-domain-knowledge")
        return finish_preparation(destination, spec)

    def read_submission(self, workspace: Path, spec: TaskSpec):
        value = super().read_submission(workspace, spec)
        for key in ("hypothesis", "workflow"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise InvalidSubmission(f"{key} must be a nonempty string")
        # Evidence is supplementary and must not change the official score.
        return {"hypothesis": value["hypothesis"], "workflow": value["workflow"]}


def private_reference(source: Path, spec: TaskSpec):
    _, payload, query = locate(source, spec.task_id, spec.split)
    gold = query.get("true_hypothesis")
    workflow = payload.get("workflow", "")
    if not gold:
        # Test answer keys are deliberately absent from prepared workspaces.
        # The official key uses topic, metadata id and qid (not our flattened
        # list position). Reject ambiguous entries instead of guessing.
        key_path = source / "eval" / "answer_key_real.csv"
        with key_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        topic, metadata, _, _ = spec.task_id.split("/")
        matches = [row for row in rows if row.get("dataset") == topic
                   and row.get("metadataid") == metadata.removeprefix("metadata_")
                   and row.get("query_id") == str(query["qid"])]
        if len(matches) != 1:
            raise ValueError("test reference is missing or ambiguous for this exact question")
        row = matches[0]
        gold = row["gold_hypo"]
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("missing private gold hypothesis")
    return gold, workflow, public_metadata(payload)
