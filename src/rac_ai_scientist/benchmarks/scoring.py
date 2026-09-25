"""Private scorer entry point. Never import a host or execute submitted code."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import time

from .base import TaskSpec, ScoreResult, read_json, write_json, digest
from .upstream import core_harness, definitions, SOURCES


class DiscoveryJudge:
    def __init__(self):
        from openai import OpenAI, AzureOpenAI
        self.model = os.environ.get("JUDGE_MODEL_NAME", "").strip()
        key = os.environ.get("JUDGE_API_KEY", "")
        if not self.model or not key:
            raise ValueError("set JUDGE_MODEL_NAME and JUDGE_API_KEY")
        self.provider = os.environ.get("JUDGE_PROVIDER", "openai")
        options = dict(api_key=key, timeout=float(os.environ.get("JUDGE_TIMEOUT", "120")), max_retries=0)
        if self.provider == "azure":
            self.client = AzureOpenAI(azure_endpoint=os.environ["JUDGE_API_BASE"],
                                     api_version=os.environ["JUDGE_API_VERSION"], **options)
        elif self.provider == "openai":
            self.client = OpenAI(base_url=os.environ.get("JUDGE_API_BASE") or None, **options)
        else:
            raise ValueError("JUDGE_PROVIDER must be openai or azure")
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def chat(self, *, messages, model_name=None, max_tokens=4096, temperature=1.0, json_response=True):
        self.usage["calls"] += 1
        options = dict(model=self.model, messages=messages)
        if self.model.rsplit("/", 1)[-1].startswith("gpt-5"):
            options["max_completion_tokens"] = max_tokens
            # Current GPT-5 Azure deployments accept only their default
            # temperature. Upstream DiscoveryBench requests temperature=0,
            # so forwarding it turns an otherwise valid score into HTTP 400.
        else:
            options.update(temperature=temperature, max_tokens=max_tokens)
        if json_response:
            options["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**options)
        if response.usage:
            self.usage["input_tokens"] += response.usage.prompt_tokens
            self.usage["output_tokens"] += response.usage.completion_tokens
        if not response.choices or response.choices[0].finish_reason != "stop" or not response.choices[0].message.content:
            raise ValueError("judge returned empty/truncated/refused output")
        return response

    def get_response(self, client, prompt, model=None, max_retry=5, **kwargs):
        # All three upstream judge paths share this client. Transport errors
        # propagate; there is no answer repair or silent zero-score fallback.
        response = self.chat(messages=[{"role": "system", "content": "You are a helpful assistant who is not talkative. You only respond with the exact answer to a query without additional conversation."},
                                       {"role": "user", "content": prompt}], json_response=False)
        value = json.loads(response.choices[0].message.content.strip().strip("```json").strip("```"))
        if not isinstance(value, dict):
            raise ValueError("judge response must be an object")
        if "sub_hypo" in value:
            if not isinstance(value["sub_hypo"], list):
                raise ValueError("invalid sub-hypothesis list")
            for item in value["sub_hypo"]:
                if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not isinstance(item.get("context"), str):
                    raise ValueError("invalid sub-hypothesis")
        elif type(value.get("match")) is not bool:
            raise ValueError("judge context decision is not a boolean")
        return value


def score_core(source: Path, dataset: Path, spec: TaskSpec, submission: dict) -> dict:
    from .corebench import reference
    task = reference(dataset, spec.task_id)
    if set(task["results"][0]) != set(spec.questions):
        raise ValueError("private reference questions do not match frozen task")
    harness = core_harness(source, scoring=True)
    if harness._construct_prompt(task) != spec.objective:
        raise ValueError("private task prompt does not match frozen task")
    harness.benchmark_answers = {spec.task_id: task["results"]}
    raw = harness.evaluate_output({spec.task_id: submission}, "rac")
    if raw[spec.task_id].get("error"):
        raise ValueError(raw[spec.task_id]["error"])
    return {"raw": raw, **harness.get_metrics(raw)}


def score_discovery(source: Path, spec: TaskSpec, submission: dict, judge) -> dict:
    from .discoverybench import locate, private_reference
    _, _, question = locate(source, spec.task_id, spec.split)
    if question["question"] != spec.objective:
        raise ValueError("private question does not match frozen task")
    gold, workflow, metadata = private_reference(source, spec)
    module = definitions(source, "discoverybench", dict(json=json, OpenAI=lambda: judge.client,
        get_response=judge.get_response, run_chatgpt_query_multi_turn=judge.chat))
    raw = module["run_eval_gold_vs_gen_NL_hypo_workflow"](
        spec.objective, gold, workflow, submission["hypothesis"], submission["workflow"],
        metadata, judge.model, "real", use_column_metadata=True)
    for item in raw["matched_gold_gen_subh_evals"].values():
        variable_score = item["var"]["score"]
        for value in (variable_score["p"], variable_score["r"], variable_score["f1"], item["rel"]["score"], item["context"]["score"]):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("official judge returned an invalid/sentinel facet score")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("spec", "submission", "source", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset")
    args = parser.parse_args(argv)
    spec = TaskSpec.from_dict(read_json(Path(args.spec)))
    source = Path(args.source)
    result = ScoreResult(spec.benchmark_id, spec.task_id, "scorer_failed")
    judge = None
    started = time.monotonic()
    result.provenance = {"source_revision": spec.source_revision,
        "scorer_source_sha256": SOURCES[spec.benchmark_id][1],
        "submission_sha256": digest(Path(args.submission)), "split": spec.split, "profile": spec.profile}
    try:
        submission = read_json(Path(args.submission))
        if spec.benchmark_id == "corebench":
            result.metrics = score_core(source, Path(args.dataset), spec, submission)
            result.total_score = result.metrics["accuracy"]
            result.provenance["private_dataset_sha256"] = digest(Path(args.dataset))
        elif spec.benchmark_id == "discoverybench":
            judge = DiscoveryJudge()
            # Official diagnostic output contains gold; it belongs only in
            # the controller-owned scoring directory, never the workspace.
            log = Path(args.output).with_suffix(".private.log")
            with log.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
                result.metrics = score_discovery(source, spec, submission, judge)
            result.total_score = result.metrics["final_score"]
        if result.total_score is None or not math.isfinite(result.total_score) or not 0 <= result.total_score <= 1:
            raise ValueError("invalid official total score")
        result.status = "scored"
    except Exception as exc:
        result.status, result.total_score = "scorer_failed", None
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.provenance["wall_seconds"] = time.monotonic() - started
        if judge:
            result.provenance.update(judge_provider=judge.provider, judge_model=judge.model, judge_usage=judge.usage)
        write_json(Path(args.output), result.to_dict())
    return 0 if result.status == "scored" else 2


if __name__ == "__main__":
    raise SystemExit(main())
