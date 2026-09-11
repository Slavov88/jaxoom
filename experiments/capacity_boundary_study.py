"""Bounded fresh-process allocator-capacity threshold study.

Diagnostic only: capacity probes are labels and are never used by production
prediction code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxoom

ROOT = Path(__file__).resolve().parent
BUDGET = 3 * 1024**3
PANEL = [
    {"configuration_id": "CAP-ATT-S1024-F32", "family": "attention", "config": {"sequence": 1024, "heads": 8, "head_dim": 64}, "dtype": "float32", "selection_reason": "small clear FIT control", "fractions": [0.15, 0.25, 0.35, 0.45, 0.55]},
    {"configuration_id": "CAP-ATT-S2048-F32", "family": "attention", "config": {"sequence": 2048, "heads": 8, "head_dim": 64}, "dtype": "float32", "selection_reason": "historical FIT control near lower transition", "fractions": [0.25, 0.35, 0.45, 0.55, 0.65]},
    {"configuration_id": "CAP-ATT-S4096-F32", "family": "attention", "config": {"sequence": 4096, "heads": 8, "head_dim": 64}, "dtype": "float32", "selection_reason": "historical FIT/unstable transition", "fractions": [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]},
    {"configuration_id": "CAP-ATT-S4608-F32", "family": "attention", "config": {"sequence": 4608, "heads": 8, "head_dim": 64}, "dtype": "float32", "selection_reason": "historical execution-OOM boundary", "fractions": [0.55, 0.65, 0.75, 0.85, 0.95]},
    {"configuration_id": "CAP-ATT-S5120-F32", "family": "attention", "config": {"sequence": 5120, "heads": 8, "head_dim": 64}, "dtype": "float32", "selection_reason": "ATT-2 historical execution-OOM", "fractions": [0.65, 0.75, 0.85, 0.95]},
    {"configuration_id": "CAP-ATT-B2-H12-S2560-D64-F32", "family": "attention", "config": {"sequence": 2560, "heads": 12, "head_dim": 64}, "dtype": "float32", "selection_reason": "discriminator execution-OOM with diagnostic request", "fractions": [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]},
    {"configuration_id": "CAP-ATT-B2-H16-S2560-D96-F16", "family": "attention", "config": {"sequence": 2560, "heads": 16, "head_dim": 96}, "dtype": "float16", "selection_reason": "discriminator FIT with same request scale", "fractions": [0.25, 0.35, 0.45, 0.55, 0.65]},
    {"configuration_id": "CAP-CONV1-B2048", "family": "convolution", "config": {"batch": 2048, "height": 64, "width": 64, "channels": 3, "out_channels": 16}, "dtype": "float32", "selection_reason": "known compile-OOM discriminator", "fractions": [0.35, 0.55, 0.75]},
    {"configuration_id": "CAP-CONV2-B1024", "family": "convolution", "config": {"batch": 1024, "height": 96, "width": 96, "channels": 3, "out_channels": 16}, "dtype": "float32", "selection_reason": "known execution-OOM discriminator", "fractions": [0.55, 0.65, 0.75, 0.85, 0.95]},
]


def attention_fn(q, k, v):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def workload_args(panel: dict[str, Any], concrete: bool = False):
    if panel["family"] == "attention":
        c = panel["config"]
        dtype = getattr(jnp, panel["dtype"])
        shape = (1, c["heads"], c["sequence"], c["head_dim"])
        factory = jnp.ones if concrete else jax.ShapeDtypeStruct
        return attention_fn, tuple(factory(shape, dtype) for _ in range(3))
    from planner_boundary_validation_v1 import workloads
    workload_id = "CONV-1" if panel["configuration_id"].endswith("CONV1-B2048") else "CONV-2"
    workload = workloads()[workload_id]
    return workload.fn, workload.args(panel["config"]["batch"], concrete)


def static_features(panel: dict[str, Any]) -> dict[str, Any]:
    fn, abstract_args = workload_args(panel)
    report = jaxoom.estimate(fn, *abstract_args)
    interval = jaxoom.calibrate(report, backend="gpu", jax_version=jax.__version__)
    live = sorted((b.nbytes for b in report.peak.live_buffers), reverse=True)
    dtype_bytes = 2 if panel["dtype"] == "float16" else 4
    return {"structural_peak": report.estimated_peak_bytes, "calibrated_central": interval.central_bytes, "calibrated_upper": interval.upper_bytes, "largest_buffer": live[0] if live else 0, "top_two_peak_live": sum(live[:2]), "peak_live": report.peak.live_bytes, "request_upper": (live[0] if live else 0) * dtype_bytes // 2, "dtype_bytes": dtype_bytes}


def classify(row: dict[str, Any]) -> str:
    if row.get("compile_status") == "COMPILE_OOM": return "COMPILE_OOM"
    if row.get("execution_status") == "EXECUTION_OOM": return "EXECUTION_OOM"
    if row.get("execution_status") == "FIT": return "FIT"
    return row.get("status", "OTHER_FAILURE")


def child_probe(panel: dict[str, Any]) -> None:
    import time
    fn, abstract_args = workload_args(panel, concrete=False)
    row: dict[str, Any] = {"configuration_id": panel["configuration_id"], "compile_status": None, "execution_status": None, "allocator_limit_bytes": None, "snapshots": [], "timings": {}}
    try:
        device = jax.devices()[0]
        row["device"] = str(device)
        row["jax_version"] = jax.__version__
        row["allocator_environment"] = {k: os.environ.get(k) for k in ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR")}
        row["allocator_limit_bytes"] = jaxoom.device_memory().allocator_limit_bytes
        started = time.perf_counter()
        concrete_args = workload_args(panel, concrete=True)[1]
        compiled = jax.jit(fn).lower(*concrete_args).compile()
        row["compile_status"] = "SUCCESS"
        row["timings"]["compile_seconds"] = time.perf_counter() - started
        try:
            memory = compiled.memory_analysis()
            row["compiler_memory"] = {k: getattr(memory, k, None) for k in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")}
        except Exception:
            row["compiler_memory"] = None
        row["allocator_limit_bytes"] = jaxoom.device_memory().allocator_limit_bytes
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        row["compile_status"] = "COMPILE_OOM" if "out of memory" in text.lower() or "resource_exhausted" in text.lower() else "OTHER_FAILURE"
        row["error"] = text[:3000]
        print(json.dumps(row), flush=True)
        return
    try:
        started = time.perf_counter()
        output = compiled(*concrete_args)
        jax.tree_util.tree_map(lambda value: value.block_until_ready(), output)
        row["execution_status"] = "FIT"
        row["timings"]["execution_seconds"] = time.perf_counter() - started
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        row["execution_status"] = "EXECUTION_OOM" if "out of memory" in text.lower() or "resource_exhausted" in text.lower() else "OTHER_FAILURE"
        row["error"] = text[:3000]
    print(json.dumps(row), flush=True)


def run_probe(panel: dict[str, Any], fraction: float, timeout: int) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--child-json", json.dumps(panel, sort_keys=True)]
    env = os.environ.copy()
    env["XLA_CLIENT_MEM_FRACTION"] = str(fraction)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"configuration_id": panel["configuration_id"], "fraction": fraction, "outcome": "COMPILE_TIMEOUT", "timeout_seconds": timeout}
    for line in reversed(completed.stdout.splitlines()):
        try:
            raw = json.loads(line)
            raw.update({"configuration_id": panel["configuration_id"], "fraction": fraction, "outcome": classify(raw), "returncode": completed.returncode})
            return raw
        except json.JSONDecodeError:
            continue
    return {"configuration_id": panel["configuration_id"], "fraction": fraction, "outcome": "OTHER_FAILURE", "stderr": (completed.stderr or completed.stdout)[-3000:], "returncode": completed.returncode}


def threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("outcome") in {"FIT", "COMPILE_OOM", "EXECUTION_OOM"} and r.get("allocator_limit_bytes") is not None]
    usable.sort(key=lambda r: r["allocator_limit_bytes"])
    inversions = []
    for low, high in zip(usable, usable[1:]):
        if low["outcome"] == "FIT" and high["outcome"] != "FIT": inversions.append({"lower": low, "higher": high})
    oom = [r for r in usable if r["outcome"] in {"COMPILE_OOM", "EXECUTION_OOM"}]
    fit = [r for r in usable if r["outcome"] == "FIT"]
    return {"last_oom_capacity": max((r["allocator_limit_bytes"] for r in oom), default=None), "first_fit_capacity": min((r["allocator_limit_bytes"] for r in fit), default=None), "compile_oom_capacities": [r["allocator_limit_bytes"] for r in oom if r["outcome"] == "COMPILE_OOM"], "execution_oom_capacities": [r["allocator_limit_bytes"] for r in oom if r["outcome"] == "EXECUTION_OOM"], "monotonicity_violations": len(inversions), "violations": inversions, "bracket_width": (min((r["allocator_limit_bytes"] for r in fit), default=0) - max((r["allocator_limit_bytes"] for r in oom), default=0)) if oom and fit else None}


def main() -> None:
    if "--child-json" in sys.argv:
        panel = json.loads(sys.argv[sys.argv.index("--child-json") + 1])
        child_probe(panel)
        return
    parser = argparse.ArgumentParser(); parser.add_argument("--plan-output", type=Path); parser.add_argument("--probes-output", type=Path); parser.add_argument("--results-output", type=Path); parser.add_argument("--timeout", type=int, default=60); parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    planned = []
    for panel in PANEL:
        planned.append({**panel, "static_features": static_features(panel), "baseline_outcome": "SEE_FROZEN_EVIDENCE"})
    plan = {"status": "FROZEN_CAPACITY_PANEL", "environment_scope": "RTX 3050 Laptop GPU, JAX 0.11.0, CUDA, BFC, PREALLOCATE=false", "capacity_control": "XLA_CLIENT_MEM_FRACTION", "panel": planned, "target_bracket_width_bytes": 64 * 1024**2}
    if args.plan_output: args.plan_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if not args.run: return
    probes = []
    for panel in PANEL:
        for fraction in panel["fractions"]:
            row = run_probe(panel, fraction, args.timeout)
            probes.append({**row, "family": panel["family"], "config": panel["config"], "dtype": panel["dtype"], "static_features": next(p["static_features"] for p in planned if p["configuration_id"] == panel["configuration_id"])})
            snapshots = row.get("snapshots") or [{}]
            print(json.dumps({"id": panel["configuration_id"], "fraction": fraction, "outcome": row.get("outcome"), "limit": row.get("allocator_limit_bytes") or snapshots[0].get("allocator_limit_bytes")}, sort_keys=True), flush=True)
    groups = {}
    for panel in planned:
        values = [r for r in probes if r["configuration_id"] == panel["configuration_id"]]
        groups[panel["configuration_id"]] = {"configuration_id": panel["configuration_id"], "family": panel["family"], "dtype": panel["dtype"], "static_features": panel["static_features"], "threshold": threshold(values), "probes": values}
    if args.probes_output: args.probes_output.write_text(json.dumps({"status": "COMPUTATIONALLY_VERIFIED", "probes": probes}, indent=2, sort_keys=True) + "\n")
    if args.results_output: args.results_output.write_text(json.dumps({"status": "COMPUTATIONALLY_VERIFIED", "workloads": groups}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
