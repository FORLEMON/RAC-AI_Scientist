"""Offline audit of existing DiscoveryBench outputs; never calls a model or host.

Run with the bundled Python (numpy required). Original run artifacts are read-only.
Only audit.json beside this script is written.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXPORT = ROOT / "output/discoverybench-results"
RUN = ROOT / "runs/smoke-20260923"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-12), (a, b)


source = read(EXPORT / "results.json")
data_path = ROOT / "prepared_tasks/discoverybench/train/nls_bmi/metadata_0/0/0/data/nls_bmi_processed.csv"
input_hash = sha(data_path)
episodes = []
for row in source["episodes"]:
    original = RUN / "episodes" / row["episode_id"]
    exported = EXPORT / row["debug_episode"]
    score = read(original / "score.json")
    assert score == read(exported / "score.json")
    assert score.get("total_score") == row["score"]
    assert sha(original / "workspace/data/nls_bmi_processed.csv") == input_hash
    item = {key: row[key] for key in (
        "episode_id", "host", "condition", "score", "run_status", "scoring_status",
        "submission_status", "model", "judge_model_used", "elapsed_minutes",
        "model_calls", "conservative_cost_usd_excluding_judge", "end_reason_zh_CN",
    )}
    item["score_path"] = relative(original / "score.json")
    item["score_sha256"] = sha(original / "score.json")
    item["input_sha256"] = input_hash
    if score["status"] == "scored":
        m = score["metrics"]
        gold = m["gold_sub_hypo"]["sub_hypo"]
        pred = m["gen_sub_hypo"]["sub_hypo"]
        pairs = []
        for key, pair in m["matched_gold_gen_subh_evals"].items():
            v = pair["var"]["score"]
            f1 = 2 * v["intersection"] / (v["sizeA"] + v["sizeB"])
            close(f1, v["f1"])
            accuracy = f1 * pair["rel"]["score"] * pair["context"]["score"]
            close(accuracy, pair["accuracy_score"])
            pairs.append({
                "pair": key, "gold_variable_count": v["sizeA"],
                "pred_variable_count": v["sizeB"], "intersection": v["intersection"],
                "variable_f1": f1, "relationship": pair["rel"]["score"],
                "context": pair["context"]["score"], "accuracy": accuracy,
                "relationship_judge_answer": pair["rel"]["answer"],
                "facet_uses_full_gold_hypothesis": pair["HypoA"] == m["HypoA"],
                "facet_uses_full_pred_hypothesis": pair["HypoB"] == m["HypoB"],
            })
        recall = len(m["gold_subh_covered"]) / len(gold)
        mean = sum(p["accuracy"] for p in pairs) / len(pred)
        total = recall * mean
        close(recall, m["recall_context"])
        close(mean, m["mean_accuracy_score"])
        close(total, m["final_score"])
        close(total, score["total_score"])
        item["decomposition"] = {
            "gold_count": len(gold), "pred_count": len(pred), "matched_count": len(pairs),
            "recall_context": recall, "mean_accuracy": mean, "recomputed_total": total,
            "gold_subhypotheses": gold, "pred_subhypotheses": pred,
            "mapping": m["gen_subh_to_gold_subh"], "pairs": pairs,
        }
    else:
        assert row["score"] is None
        item["decomposition"] = None
    events_path = original / "coordination.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()] if events_path.exists() else []
    item["route"] = [e["payload"]["capability_id"] for e in events if e["type"] == "decision" and e["payload"].get("capability_id")]
    item["verification"] = [
        {"hop": e.get("hop"), "verdict": e["payload"]["verification"]["verdict"],
         "checks": [{"name": c["name"], "passed": c["passed"]} for c in e["payload"]["verification"]["checks"]]}
        for e in events if e["type"] == "evaluation" and e["payload"].get("verification")
    ]
    runtime_paths = sorted((RUN / "logs" / row["episode_id"] / "task-runtime").glob("*.result.json"))
    runtime = [read(p) for p in runtime_paths]
    counts = Counter(str(r.get("exit_code")) for r in runtime)
    item["task_runtime"] = {
        "requests": len(runtime), "exit_codes": dict(counts),
        "nonzero_exit": sum(r.get("exit_code") != 0 for r in runtime),
        "timeout_flag": sum(bool(r.get("timed_out")) for r in runtime),
        "note": "External task-runtime requests, not model calls or independent experiments.",
    }
    episodes.append(item)

with data_path.open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
columns = list(rows[0])
arr = {k: np.array([1.0 if r[k] == "MALE" else 0.0 for r in rows]) if k == "GENDER"
       else np.array([float(r[k]) for r in rows]) for k in columns}
assert all(np.isfinite(v).all() for v in arr.values())


def ols(names, arrays=None):
    arrays = arrays or arr
    x = np.column_stack([np.ones(len(rows)), *[arrays[n] for n in names]])
    coef, _, rank, _ = np.linalg.lstsq(x, arr["BMI"], rcond=None)
    assert rank == len(names) + 1
    return dict(zip(["Intercept", *names], map(float, coef)))


groups = []
for d, s in sorted(set(zip(arr["DISSAVED"], arr["SAMESAVE"]))):
    mask = (arr["DISSAVED"] == d) & (arr["SAMESAVE"] == s)
    groups.append({"DISSAVED": int(d), "SAMESAVE": int(s), "n": int(mask.sum()), "mean_bmi": float(arr["BMI"][mask].mean())})
simple = ols(["DISSAVED", "SAMESAVE"])
assert round(simple["DISSAVED"], 4) == 0.3596
assert round(simple["SAMESAVE"], 4) == 0.4858
with_future = dict(arr, future_oriented=((arr["DISSAVED"] == 0) & (arr["SAMESAVE"] == 0)).astype(float))
future = ols(["future_oriented", "INCOME", "AGE", "GENDER", "BLACK", "HISPANIC"], with_future)
assert round(future["future_oriented"], 3) == -0.431

scorer = ROOT / "upstreams/discoverybench/eval/new_eval.py"
tree = ast.parse(scorer.read_text(encoding="utf-8"))
workflow = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_eval_gold_vs_gen_NL_hypo_workflow")
call = next(n for n in ast.walk(workflow) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_eval_gold_vs_gen_NL_subhypo")
facet_args = [ast.unparse(a) for a in call.args]
assert facet_args[1:5] == ["gold_hypo", "gold_workflow", "gen_hypo", "gen_workflow"]
source_checks = []
for name in ["policy.py", "benchmarks/scoring.py"]:
    current = ROOT / "src/rac_ai_scientist" / name
    frozen = RUN / "controller-source/rac_ai_scientist" / name
    assert sha(current) == sha(frozen)
    source_checks.append({"name": name, "current_sha256": sha(current), "frozen_sha256": sha(frozen), "identical": True})
normalized_scorer_hash = hashlib.sha256(scorer.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
assert normalized_scorer_hash == "83406ddbe6894429cc5012b17cb82a209d5397aa7863b0aba1bb51edf1dd6c65"
lab_r1 = next(e for e in episodes if e["host"] == "agent_laboratory" and e["condition"] == "R1")
lab_r3 = next(e for e in episodes if e["host"] == "agent_laboratory" and e["condition"] == "R3")
assert lab_r1["route"] == lab_r3["route"] and len(lab_r3["route"]) == 7
assert len(lab_r3["verification"]) == 7 and all(v["verdict"] == "supported" for v in lab_r3["verification"])
c = lab_r1["decomposition"]
context_inconsistency = {
    "gold_contexts_equal": c["gold_subhypotheses"][0]["context"] == c["gold_subhypotheses"][1]["context"],
    "pred_contexts_equal": c["pred_subhypotheses"][0]["context"] == c["pred_subhypotheses"][1]["context"],
    "mapping": c["mapping"],
    "limit": "Prompt also contains hypothesis text. False context responses are not retained; no repeat judging was performed.",
}
assert context_inconsistency["gold_contexts_equal"] and context_inconsistency["pred_contexts_equal"]
assert c["mapping"]["0"] == 0 and c["mapping"]["1"] == -1
payload = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Existing six retained DiscoveryBench runs, one task and seed. Offline auditing only; no judge or host calls.",
    "original_runs_modified": False,
    "formula": "(matched / gold) * sum(context * variable_f1 * relationship) / predicted",
    "episodes": episodes,
    "data_audit": {
        "path": relative(data_path), "sha256": input_hash, "rows": len(rows), "columns": columns,
        "all_six_inputs_identical": True, "groups": groups,
        "simple_ols": simple,
        "adjusted_ols": ols(["DISSAVED", "SAMESAVE", "AGE", "AGE_2", "GENDER", "INCOME", "BLACK", "HISPANIC"]),
        "future_orientation_ols": future,
        "future_orientation_definition": "1 iff DISSAVED == 0 and SAMESAVE == 0, else 0",
        "future_ols_note": "Raw income and age units; original agent standardized these covariates. This leaves the future_oriented coefficient unchanged.",
        "gold_rounded_coefficients_reproduced": True,
        "agent_r3_rounded_future_coefficient_reproduced": True,
        "scientific_limit": "Only these OLS coefficients were independently recomputed. No verification of all reported tests, mediation, causal interpretation, figures, or p-values.",
    },
    "scorer_audit": {
        "discovery_revision": "c31fcf011e070f021a5f5b906896d0821f6880e8",
        "lf_normalized_sha256": normalized_scorer_hash,
        "facet_call_line": call.lineno, "facet_call_arguments": facet_args,
        "facets_use_whole_hypotheses_not_matched_subhypotheses": True,
        "agent_r1_context_inconsistency": context_inconsistency,
        "judge_protocol": {
            "official_entrypoint_default": "gpt-4-1106-preview",
            "current_run_judge": "gpt-5.4",
            "decomposition_context_temperature": 1,
            "decomposition_context_output_cap": 4096,
            "official_facet_helper_temperature": 0,
            "official_facet_helper_output_cap": 250,
            "adapter_facet_temperature": 0,
            "adapter_facet_output_cap": 512,
            "note": "Official helper ignores the caller's 512 argument; adapter honors it. This does not establish the causal size of any score change.",
        },
        "frozen_code_checks": source_checks,
    },
    "researchclawbench": {
        "revision": "ed664513287a1fc60ae319e6bf7dbe7a62144135",
        "source": "https://github.com/InternScience/ResearchClawBench/blob/ed664513287a1fc60ae319e6bf7dbe7a62144135/evaluation/score.py",
        "local_download": "tmp/benchmark-analysis/researchclawbench-score.py",
        "source_sha256": sha(ROOT / "tmp/benchmark-analysis/researchclawbench-score.py"),
        "historical_raw_scores_available_in_this_audit": False,
        "formula": "round(sum(item_weight * item_score) / sum(item_weight), 2); item_score: 0..100",
    },
    "validation": {
        "six_original_score_files_equal_exports": True,
        "five_numeric_scores_recomputed_exactly": True,
        "one_invalid_submission_preserved_as_null": True,
        "six_input_hashes_match": True,
        "agent_laboratory_r1_r3_same_seven_step_route": True,
    },
}
OUT.mkdir(exist_ok=True)
(OUT / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"episodes": len(episodes), "scored": sum(e["score"] is not None for e in episodes), "validation": payload["validation"], "path": str(OUT / "audit.json")}, ensure_ascii=False, indent=2))
