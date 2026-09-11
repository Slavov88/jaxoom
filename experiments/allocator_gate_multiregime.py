"""Frozen multi-regime RTX study for the experimental allocator gate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import jaxoom

BUDGET = 3 * 1024**3
CANDIDATES = [
    {"id": "ATT-B1-H8-S4096-D64-F32", "family": "attention", "B": 1, "H": 8, "S": 4096, "D": 64, "dtype": "float32"},
    {"id": "ATT-B2-H4-S3072-D64-F32", "family": "attention", "B": 2, "H": 4, "S": 3072, "D": 64, "dtype": "float32"},
    {"id": "ATT-B1-H4-S4608-D64-F32", "family": "attention", "B": 1, "H": 4, "S": 4608, "D": 64, "dtype": "float32"},
    {"id": "ATT-B1-H16-S4096-D32-F32", "family": "attention", "B": 1, "H": 16, "S": 4096, "D": 32, "dtype": "float32"},
    {"id": "ATT-B1-H4-S4096-D128-F32", "family": "attention", "B": 1, "H": 4, "S": 4096, "D": 128, "dtype": "float32"},
    {"id": "ATT-B1-H8-S3072-D64-F16", "family": "attention", "B": 1, "H": 8, "S": 3072, "D": 64, "dtype": "float16"},
    {"id": "ATT-B1-H8-S4096-D64-F16", "family": "attention", "B": 1, "H": 8, "S": 4096, "D": 64, "dtype": "float16"},
    {"id": "ATT-B1-H8-S4608-D64-F16", "family": "attention", "B": 1, "H": 8, "S": 4608, "D": 64, "dtype": "float16"},
    {"id": "ATT-B2-H8-S3072-D32-F16", "family": "attention", "B": 2, "H": 8, "S": 3072, "D": 32, "dtype": "float16"},
    {"id": "ATT-B1-H16-S3072-D32-F16", "family": "attention", "B": 1, "H": 16, "S": 3072, "D": 32, "dtype": "float16"},
    {"id": "MLP-2-CONTROL", "family": "MLP", "control": "MLP-2", "batch": 9632},
    {"id": "TRAIN-1-CONTROL", "family": "training", "control": "TRAIN-1", "batch": 5986},
    {"id": "CONV-1-CONTROL", "family": "convolution", "control": "CONV-1", "batch": 409},
    {"id": "ATT-B1-H8-S3584-D64-F32", "family": "attention", "B": 1, "H": 8, "S": 3584, "D": 64, "dtype": "float32"},
    {"id": "ATT-B4-H4-S2048-D64-F32", "family": "attention", "B": 4, "H": 4, "S": 2048, "D": 64, "dtype": "float32"},
]


def attention(q, k, v):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def attention_args(case, concrete=False):
    dtype = getattr(jnp, case["dtype"])
    shape = (case["B"], case["H"], case["S"], case["D"])
    return tuple(jnp.ones(shape, dtype) if concrete else jax.ShapeDtypeStruct(shape, dtype) for _ in range(3))


def load_workload(case):
    if case.get("control"):
        from planner_boundary_validation_v1 import workloads
        workload = workloads()[case["control"]]
        return workload.fn, lambda concrete=False: workload.args(workload.args(1, False)[0].shape[0] if False else case.get("batch", 1), concrete)
    return attention, lambda concrete=False: attention_args(case, concrete)


def static_row(case):
    if case.get("control"):
        from planner_boundary_validation_v1 import workloads
        workload = workloads()[case["control"]]
        args = workload.args(case["batch"], False)
        report = jaxoom.estimate(workload.fn, *args)
    else:
        report = jaxoom.estimate(attention, *attention_args(case))
    interval = jaxoom.calibrate(report, backend="gpu", jax_version=jax.__version__)
    live = sorted((buffer.nbytes for buffer in report.peak.live_buffers), reverse=True)
    proxy = sum(live[:2])
    return {**case, "budget_bytes": BUDGET, "structural_peak_bytes": report.estimated_peak_bytes, "calibrated_upper_bytes": interval.upper_bytes, "largest_buffer_bytes": live[0] if live else 0, "top_two_peak_live_bytes": proxy, "gate_score": (interval.upper_bytes + proxy) / BUDGET, "aggregate_pass": interval.upper_bytes <= BUDGET, "allocator_gate_pass": interval.upper_bytes + proxy <= BUDGET, "split": "held_out"}


def child(case_id: str):
    case = next(item for item in CANDIDATES if item["id"] == case_id)
    if case.get("control"):
        from planner_boundary_validation_v1 import workloads
        workload = workloads()[case["control"]]
        args = workload.args(case["batch"], True)
        fn = workload.fn
    else:
        fn = attention
        args = attention_args(case, True)
    started = time.perf_counter()
    try:
        compiled = jax.jit(fn).lower(*args).compile()
        compile_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        output = compiled(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
        print(json.dumps({"outcome": "FIT", "compile_latency_ms": compile_ms, "execution_latency_ms": (time.perf_counter() - started) * 1000}), flush=True)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        oom = "out of memory" in text.lower() or "resource_exhausted" in text.lower()
        print(json.dumps({"outcome": "EXECUTION_OOM" if oom else "OTHER_FAILURE", "error": text}), flush=True)


def probe(case_id: str, timeout: int = 60) -> dict:
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        result = subprocess.run([sys.executable, str(Path(__file__)), "--child", "--case-id", case_id], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"outcome": "EXECUTION_TIMEOUT"}
    for line in reversed(result.stdout.splitlines()):
        try: return json.loads(line)
        except json.JSONDecodeError: pass
    return {"outcome": "OTHER_FAILURE", "stderr": result.stderr[-2000:]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--plan-output", type=Path); parser.add_argument("--results-output", type=Path); parser.add_argument("--run", action="store_true"); parser.add_argument("--child", action="store_true"); parser.add_argument("--case-id"); parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.child: child(args.case_id); return
    rows = [static_row(case) for case in CANDIDATES]
    plan = {"status": "FROZEN_PLAN", "environment": {"device": str(jax.devices()[0]), "backend": jax.default_backend(), "jax_version": jax.__version__}, "budget_bytes": BUDGET, "candidate_formula": "calibrated_upper + top_two_peak_live <= budget", "rows": rows}
    if args.plan_output: args.plan_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if args.run:
        results = []
        for row in rows:
            # Freeze the prediction before probing. Repeat near gate boundary.
            repeats = 2 if 0.85 <= row["gate_score"] <= 1.15 else 1
            outcomes = [probe(row["id"], args.timeout) for _ in range(repeats)]
            results.append({"configuration_id": row["id"], "prediction": row["allocator_gate_pass"], "gate_score": row["gate_score"], "outcomes": outcomes, "stable_outcome": outcomes[0]["outcome"] if all(item.get("outcome") == outcomes[0].get("outcome") for item in outcomes) else "UNSTABLE"})
        output = {"status": "COMPUTATIONALLY_VERIFIED", "plan": plan, "results": results}
        if args.results_output: args.results_output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
