"""Focused boundary validation harness for the compilation-free batch planner.

Planning itself never compiles. ``--validate-execution`` is an explicit
post-planning harness step and is the only mode that compiles selected cases.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import jaxoom


def workloads():
    def mlp(b, concrete=False):
        width = 1024
        factory = jnp.ones if concrete else lambda s, d: jax.ShapeDtypeStruct(s, d)
        return (factory((b, width), jnp.float32),) + tuple(factory((width, width), jnp.float32) for _ in range(3))

    def training(b, concrete=False):
        width = 1024
        factory = jnp.ones if concrete else lambda s, d: jax.ShapeDtypeStruct(s, d)
        return (factory((b, width), jnp.float32), factory((b, width), jnp.float32), factory((4, width, width), jnp.float32))

    def attention(b, concrete=False):
        factory = jnp.ones if concrete else lambda s, d: jax.ShapeDtypeStruct(s, d)
        return tuple(factory(s, jnp.float32) for s in ((b, 4, 64), (b, 4, 64), (b, 4, 64)))

    def convolution(b, concrete=False):
        factory = jnp.ones if concrete else lambda s, d: jax.ShapeDtypeStruct(s, d)
        return factory((b, 32, 32, 3), jnp.float32), factory((3, 3, 3, 8), jnp.float32)

    def mlp_fn(x, w1, w2, w3):
        y1 = jnp.tanh(x @ w1)
        y2 = jnp.tanh(y1 @ w2)
        y3 = jnp.tanh(y2 @ w3)
        return y1, y2, y3

    return {
        "mlp": (mlp_fn, mlp),
        "training_like": (lambda x, target, params: ((x + params[0, 0, :] - target) ** 2, params), training),
        "attention": (lambda q, k, v: jnp.einsum("bhd,bhe->bhde", q, k) + v[:, :, :, None], attention),
        "convolution": (lambda x, kernel: jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")), convolution),
    }


def serialize_trial(trial):
    if trial.assessment is None:
        return {"batch_size": trial.batch_size, "error": trial.error}
    a = trial.assessment
    return {"batch_size": trial.batch_size, "structural_peak_bytes": a.structural_peak_bytes, "central_bytes": a.interval.central_bytes, "upper_bytes": a.interval.upper_bytes, "memory_limit_bytes": a.memory_limit_bytes, "risk": a.risk.value if a.risk else None, "calibrated": a.calibrated}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-execution", action="store_true")
    parser.add_argument("--auto-pressure", action="store_true", help="run bounded auto-budget pressure scenarios")
    parser.add_argument("--max-evaluations", type=int, default=64)
    args = parser.parse_args()
    results = []
    for name, (fn, factory) in workloads().items():
        for budget in (16 * 1024**2, 32 * 1024**2, 64 * 1024**2):
            started = time.perf_counter()
            plan = jaxoom.plan_batch_size(fn, lambda b, f=factory: f(b), memory_limit=budget, min_batch_size=1, max_batch_size=2048, max_evaluations=args.max_evaluations)
            row = {"workload": name, "budget_bytes": budget, "planner_latency_ms": (time.perf_counter() - started) * 1000, "recommended_batch_size": plan.recommended_batch_size, "next_tested": serialize_trial(plan.next_failing_or_riskier) if plan.next_failing_or_riskier else None, "evaluations": plan.evaluations, "status": plan.status, "monotonic": plan.monotonic, "trials": [serialize_trial(t) for t in plan.trials]}
            if args.validate_execution and plan.recommended_batch_size is not None:
                for label, batch in (("recommended", plan.recommended_batch_size), ("next", plan.next_failing_or_riskier.batch_size if plan.next_failing_or_riskier else None)):
                    if batch is None: continue
                    try:
                        values = factory(batch, concrete=True)
                        compile_started = time.perf_counter()
                        compiled = jax.jit(fn).lower(*values).compile()
                        row[label + "_compile_latency_ms"] = (time.perf_counter() - compile_started) * 1000
                        execute_started = time.perf_counter()
                        jax.tree_util.tree_map(lambda x: x.block_until_ready(), compiled(*values))
                        row[label + "_execution_latency_ms"] = (time.perf_counter() - execute_started) * 1000
                        row[label + "_actual_outcome"] = "FIT"
                    except Exception as exc:
                        row[label + "_actual_outcome"] = "EXECUTION_OOM" if "out of memory" in str(exc).lower() else "OTHER_FAILURE"
                        row[label + "_actual_error"] = f"{type(exc).__name__}: {exc}"
            results.append(row)
    if args.auto_pressure:
        fn, factory = workloads()["attention"]
        for pressure_bytes in (0, 512 * 1024**2, 1024 * 1024**2):
            pressure = None
            try:
                if pressure_bytes:
                    pressure = jnp.ones((pressure_bytes // 4,), jnp.float32)
                    pressure.block_until_ready()
                started = time.perf_counter()
                plan = jaxoom.plan_batch_size(fn, lambda b, f=factory: f(b), memory_limit="auto", min_batch_size=1, max_batch_size=8192, max_evaluations=args.max_evaluations)
                row = {"workload": "attention", "scenario": "auto_pressure", "pressure_bytes": pressure_bytes, "budget_bytes": plan.memory_limit_bytes, "planner_latency_ms": (time.perf_counter() - started) * 1000, "recommended_batch_size": plan.recommended_batch_size, "next_tested": serialize_trial(plan.next_failing_or_riskier) if plan.next_failing_or_riskier else None, "evaluations": plan.evaluations, "status": plan.status, "monotonic": plan.monotonic, "device_snapshot": {"timestamp": plan.device_budget.snapshot.timestamp if plan.device_budget else None, "effective_available_bytes": plan.device_budget.effective_available_bytes if plan.device_budget else None}, "trials": [serialize_trial(t) for t in plan.trials]}
                results.append(row)
            finally:
                del pressure
                jax.clear_caches()
    snapshot = jaxoom.device_memory()
    payload = {"status": "COMPUTATIONALLY_VERIFIED" if args.validate_execution else "OBSERVED", "environment": {"jax_version": jax.__version__, "backend": jax.default_backend(), "device": str(jax.devices()[0])}, "device_snapshot": {"backend": snapshot.backend, "device_kind": snapshot.device_kind, "effective_available_bytes": snapshot.effective_available_bytes, "timestamp": snapshot.timestamp}, "target_compilation_during_planning": False, "target_execution_during_planning": False, "validation_execution_enabled": args.validate_execution, "rows": results}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
