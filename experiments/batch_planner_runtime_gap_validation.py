"""Fresh-process runtime-boundary validation for the batch planner."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import jax

import jaxoom
from batch_planner_boundary_validation import workloads


SCENARIOS = (
    ("mlp-32m", "mlp", 32 * 1024**2),
    ("training-32m", "training_like", 32 * 1024**2),
    ("attention-32m", "attention", 32 * 1024**2),
    ("convolution-32m", "convolution", 32 * 1024**2),
    ("attention-auto", "attention", "auto"),
    ("mlp-auto", "mlp", "auto"),
)


def child(family: str, batch: int) -> None:
    fn, factory = workloads()[family]
    values = factory(batch, concrete=True)
    started = time.perf_counter()
    try:
        compiled = jax.jit(fn).lower(*values).compile()
    except Exception as exc:
        print(json.dumps({"outcome": "COMPILE_OOM" if _is_oom(exc) else "OTHER_FAILURE", "phase": "compile", "error": f"{type(exc).__name__}: {exc}", "latency_ms": (time.perf_counter() - started) * 1000}), flush=True)
        return
    compile_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    try:
        result = compiled(*values)
        jax.tree_util.tree_map(lambda value: value.block_until_ready(), result)
        print(json.dumps({"outcome": "FIT", "compile_latency_ms": compile_ms, "execution_latency_ms": (time.perf_counter() - started) * 1000}), flush=True)
    except Exception as exc:
        print(json.dumps({"outcome": "EXECUTION_OOM" if _is_oom(exc) else "OTHER_FAILURE", "phase": "execution", "compile_latency_ms": compile_ms, "error": f"{type(exc).__name__}: {exc}"}), flush=True)


def _is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "resource_exhausted" in text or "resource exhausted" in text


def trial_payload(trial):
    if trial is None:
        return None
    if trial.assessment is None:
        return {"batch": trial.batch_size, "error": trial.error}
    assessment = trial.assessment
    return {"batch": trial.batch_size, "upper_bytes": assessment.interval.upper_bytes, "budget_bytes": assessment.memory_limit_bytes, "risk": assessment.risk.value if assessment.risk else None, "calibrated": assessment.calibrated}


def run_probe(family: str, batch: int, timeout: int) -> dict:
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    command = [sys.executable, str(Path(__file__)), "--child", "--family", family, "--batch", str(batch)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"batch": batch, "outcome": "EXECUTION_TIMEOUT"}
    for line in reversed(completed.stdout.splitlines()):
        try:
            row = json.loads(line)
            row["batch"] = batch
            if completed.returncode and row.get("outcome") == "FIT":
                row["outcome"] = "OTHER_FAILURE"
            return row
        except json.JSONDecodeError:
            continue
    return {"batch": batch, "outcome": "OTHER_FAILURE", "stderr": completed.stderr[-2000:]}


def geometric_points(start: int, cap: int) -> list[int]:
    points, current = [], start
    while current <= cap:
        points.append(current)
        current = max(current + 1, math.ceil(current * 1.35))
    if points[-1] != cap:
        points.append(cap)
    return list(dict.fromkeys(points))


def assess_scenario(scenario_id: str, family: str, budget, timeout: int, checkpoint: dict) -> dict:
    fn, factory = workloads()[family]
    started = time.perf_counter()
    plan = jaxoom.plan_batch_size(fn, lambda batch: factory(batch), memory_limit=budget, min_batch_size=1, max_batch_size=8192, max_evaluations=64)
    planner_ms = (time.perf_counter() - started) * 1000
    if plan.recommended_batch_size is None:
        return {"scenario_id": scenario_id, "family": family, "budget_type": "auto" if budget == "auto" else "explicit", "budget_bytes": plan.memory_limit_bytes, "planner_recommended_batch": None, "planner_recommended_upper": None, "runtime_batches_tested": [], "runtime_boundary_reached": False, "status": "NOTHING_FITS", "planner_latency_ms": planner_ms, "evaluations": plan.evaluations}
    recommendation = plan.recommended_batch_size
    cap = min(16384, max(recommendation + 1, recommendation * 8))
    tested = checkpoint.setdefault("probe_cache", {}).setdefault(scenario_id, {})
    def probe(batch):
        key = str(batch)
        if key not in tested:
            result = run_probe(family, batch, timeout)
            try:
                report = jaxoom.estimate(fn, *factory(batch))
                assessment = jaxoom.assess(report, budget, device_budget=plan.device_budget)
                result.update({"predicted_structural_bytes": assessment.structural_peak_bytes, "predicted_upper_bytes": assessment.interval.upper_bytes, "predicted_excess_bytes": (assessment.interval.upper_bytes - assessment.memory_limit_bytes) if assessment.memory_limit_bytes is not None else None})
            except Exception as exc:
                result["prediction_error"] = f"{type(exc).__name__}: {exc}"
            tested[key] = result
            checkpoint["_write"] = True
        return tested[key]
    recommended_result = probe(recommendation)
    low = recommendation if recommended_result["outcome"] == "FIT" else None
    if low is None:
        return {"scenario_id": scenario_id, "family": family, "budget_type": "auto" if budget == "auto" else "explicit", "budget_bytes": plan.memory_limit_bytes, "planner_recommended_batch": recommendation, "planner_recommended_upper": (next((item for item in plan.trials if item.batch_size == recommendation), None).assessment.interval.upper_bytes if next((item for item in plan.trials if item.batch_size == recommendation), None) and next((item for item in plan.trials if item.batch_size == recommendation), None).assessment else None), "runtime_batches_tested": list(tested.values()), "highest_known_fit": None, "first_known_oom": recommendation, "exact_runtime_boundary": False, "runtime_boundary_reached": True, "runtime_gap_absolute": None, "runtime_gap_ratio": None, "planner_utilization": None, "next_batch": recommendation, "planner_latency_ms": planner_ms, "evaluations": plan.evaluations, "status": "RECOMMENDED_NOT_FIT"}
    points = geometric_points(recommendation + 1, cap)
    high = None
    for batch in points:
        row = probe(batch)
        if row["outcome"] == "FIT":
            low = batch
        elif row["outcome"] in {"COMPILE_OOM", "EXECUTION_OOM"}:
            high = batch
            break
        else:
            break
    if high is not None:
        while high - low > 1:
            middle = (low + high) // 2
            row = probe(middle)
            if row["outcome"] == "FIT": low = middle
            elif row["outcome"] in {"COMPILE_OOM", "EXECUTION_OOM"}: high = middle
            else: break
    recommended_trial = next((item for item in plan.trials if item.batch_size == recommendation), None)
    next_batch = high if high is not None else None
    return {"scenario_id": scenario_id, "family": family, "budget_type": "auto" if budget == "auto" else "explicit", "budget_bytes": plan.memory_limit_bytes, "planner_recommended_batch": recommendation, "planner_recommended_upper": recommended_trial.assessment.interval.upper_bytes if recommended_trial and recommended_trial.assessment else None, "runtime_batches_tested": list(tested.values()), "highest_known_fit": low, "first_known_oom": high, "exact_runtime_boundary": high == low + 1 if high is not None else False, "runtime_boundary_reached": high is not None, "runtime_gap_absolute": low - recommendation, "runtime_gap_ratio": low / recommendation, "planner_utilization": recommendation / low, "next_batch": next_batch, "planner_latency_ms": planner_ms, "evaluations": plan.evaluations, "status": "COMPLETE" if high is not None else "RUNTIME_BOUNDARY_NOT_REACHED"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--family")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.child:
        child(args.family, args.batch)
        return
    if args.output is None:
        parser.error("--output is required")
    checkpoint = json.loads(args.output.read_text()) if args.output.exists() else {"status": "OBSERVED", "scenarios": {}, "probe_cap": 60}
    scenarios = checkpoint.setdefault("scenarios", {})
    for scenario_id, family, budget in SCENARIOS:
        if scenario_id in scenarios and scenarios[scenario_id].get("status") in {"COMPLETE", "RUNTIME_BOUNDARY_NOT_REACHED", "NOTHING_FITS"}:
            continue
        scenarios[scenario_id] = assess_scenario(scenario_id, family, budget, args.timeout, scenarios)
        scenarios.pop("_write", None)
        checkpoint["completed_scenarios"] = len(scenarios)
        args.output.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint.pop("_write", None)
    args.output.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
