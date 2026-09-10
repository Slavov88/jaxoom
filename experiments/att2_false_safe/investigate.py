"""Reproduce and diagnose the ATT-2 false-safe without changing planner logic."""
from __future__ import annotations

import argparse
import dataclasses
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
from planner_boundary_validation_v1 import workloads

OUTPUT_DIR = Path(__file__).resolve().parent
BUDGET_BYTES = 3 * 1024**3
WORKLOAD_ID = "ATT-2"


def snapshot() -> dict[str, Any]:
    try:
        return dataclasses.asdict(jaxoom.device_memory())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", None),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "allocator_environment": {
            key: os.environ.get(key)
            for key in (
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "XLA_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "TF_GPU_ALLOCATOR",
            )
        },
    }


def planner_breakdown(workload_id: str = WORKLOAD_ID, budget_bytes: int = BUDGET_BYTES) -> dict[str, Any]:
    workload = workloads()[workload_id]
    abstract_args = workload.args(1, False)
    report = jaxoom.estimate(workload.fn, *abstract_args)
    assessment = jaxoom.assess(report, memory_limit=budget_bytes)
    plan = jaxoom.plan_batch_size(
        workload.fn,
        lambda batch: workload.args(batch, False),
        memory_limit=budget_bytes,
        min_batch_size=1,
        max_batch_size=4,
        max_evaluations=8,
    )

    def buffer_info(buffer: Any) -> dict[str, Any]:
        return {
            "variable_id": buffer.variable_id,
            "shape": list(buffer.shape),
            "dtype": buffer.dtype,
            "nbytes": buffer.nbytes,
            "producer": buffer.producer,
            "birth": buffer.birth,
            "last_use": buffer.last_use,
            "is_input": buffer.is_input,
            "is_output": buffer.is_output,
        }

    return {
        "workload_id": workload_id,
        "batch_size": 1,
        "budget_bytes": budget_bytes,
        "recommendation": plan.recommended_batch_size,
        "plan_status": plan.status,
        "plan_trials": [
            {
                "batch_size": trial.batch_size,
                "error": trial.error,
                "upper_bytes": trial.assessment.interval.upper_bytes if trial.assessment else None,
                "central_bytes": trial.assessment.interval.central_bytes if trial.assessment else None,
                "risk": trial.assessment.risk.value if trial.assessment and trial.assessment.risk else None,
            }
            for trial in plan.trials
        ],
        "static": {
            "estimated_peak_bytes": report.estimated_peak_bytes,
            "equations_analyzed": report.equations_analyzed,
            "confidence": report.confidence,
            "model": report.model,
            "unsupported_constructs": report.unsupported_constructs,
            "largest_buffers": [buffer_info(item) for item in report.largest_buffers],
            "peak": {
                "equation": report.peak.equation,
                "primitive": report.peak.primitive,
                "live_bytes": report.peak.live_bytes,
                "live_buffers": [buffer_info(item) for item in report.peak.live_buffers],
            },
        },
        "calibrated": {
            "lower_bytes": assessment.interval.lower_bytes,
            "central_bytes": assessment.interval.central_bytes,
            "upper_bytes": assessment.interval.upper_bytes,
            "calibration_method": assessment.interval.calibration_method,
            "calibration_scope": assessment.interval.calibration_scope,
            "dataset_version": assessment.interval.dataset_version,
            "sample_count": assessment.interval.sample_count,
            "applicability": assessment.interval.applicability,
            "risk": assessment.risk.value if assessment.risk else None,
            "headroom_to_upper_bytes": assessment.headroom_to_upper_bytes,
        },
        "input_shapes": [list(arg.shape) for arg in abstract_args],
        "dtype": "float32",
        "environment": environment(),
    }


def memory_analysis(compiled: Any) -> dict[str, Any]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {"available": False}
    values = {
        "argument_bytes": getattr(stats, "argument_size_in_bytes", None),
        "output_bytes": getattr(stats, "output_size_in_bytes", None),
        "temporary_bytes": getattr(stats, "temp_size_in_bytes", None),
        "alias_bytes": getattr(stats, "alias_size_in_bytes", None),
    }
    values = {key: int(value) if value is not None else None for key, value in values.items()}
    if all(value is not None for value in values.values()):
        values["compiler_accounted_bytes"] = (
            values["argument_bytes"] + values["output_bytes"] + values["temporary_bytes"] - values["alias_bytes"]
        )
    else:
        values["compiler_accounted_bytes"] = None
    values["available"] = True
    return values


def is_oom(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "out of memory",
            "resource exhausted",
            "resource_exhausted",
            "cuda_error_out_of_memory",
            "failed to allocate",
        )
    )


def requested_allocation(text: str) -> dict[str, Any] | None:
    matches = re.findall(r"(?:allocate|allocation)[^\n]{0,100}?([0-9]+(?:\.[0-9]+)?)\s*(KiB|MiB|GiB|KB|MB|GB)", text, re.I)
    if not matches:
        return None
    value, unit = matches[-1]
    factor = {"kib": 2**10, "mib": 2**20, "gib": 2**30, "kb": 10**3, "mb": 10**6, "gb": 10**9}[unit.lower()]
    return {"value": float(value), "unit": unit, "bytes_approx": round(float(value) * factor), "source": "exception text"}


def block(value: Any) -> None:
    jax.tree_util.tree_map(lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item, value)


def probe(rep: int, workload_id: str = WORKLOAD_ID) -> None:
    workload = workloads()[workload_id]
    values = workload.args(1, True)
    started = time.perf_counter()
    before = snapshot()
    phase = "INPUT_CREATION"
    compile_seconds = None
    execute_seconds = None
    compiler = None
    try:
        for value in values:
            if hasattr(value, "block_until_ready"):
                value.block_until_ready()
        phase = "COMPILE"
        compile_start = time.perf_counter()
        lowered = jax.jit(workload.fn).lower(*values)
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - compile_start
        after_compile = snapshot()
        compiler = memory_analysis(compiled)
        phase = "FIRST_EXECUTION"
        execute_start = time.perf_counter()
        output = compiled(*values)
        block(output)
        execute_seconds = time.perf_counter() - execute_start
        after_execute = snapshot()
        payload = {
            "rep": rep,
            "workload_id": workload_id,
            "batch_size": 1,
            "outcome": "FIT",
            "failure_phase": None,
            "error_type": None,
            "error": None,
            "requested_allocation": None,
            "compile_seconds": compile_seconds,
            "execute_seconds": execute_seconds,
            "wall_seconds": time.perf_counter() - started,
            "before": before,
            "after_compile": after_compile,
            "after_execute": after_execute,
            "compiler_memory": compiler,
            "environment": environment(),
        }
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        payload = {
            "rep": rep,
            "workload_id": workload_id,
            "batch_size": 1,
            "outcome": "OOM" if is_oom(text) else "OTHER_FAILURE",
            "failure_phase": phase,
            "error_type": type(exc).__name__,
            "error": text[-8000:],
            "requested_allocation": requested_allocation(text),
            "compile_seconds": compile_seconds,
            "execute_seconds": execute_seconds,
            "wall_seconds": time.perf_counter() - started,
            "before": before,
            "after_compile": snapshot() if phase in {"COMPILE", "FIRST_EXECUTION"} else None,
            "after_execute": None,
            "compiler_memory": compiler,
            "environment": environment(),
        }
    print(json.dumps(payload, sort_keys=True), flush=True)
    raise SystemExit(0 if payload["outcome"] == "FIT" else 2)


def run_repetitions(
    repetitions: int, timeout: int, workload_id: str = WORKLOAD_ID, budget_bytes: int = BUDGET_BYTES
) -> dict[str, Any]:
    rows = []
    command_base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe",
        "--workload-id",
        workload_id,
        "--budget-bytes",
        str(budget_bytes),
    ]
    for rep in range(1, repetitions + 1):
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command_base + ["--rep", str(rep)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            row = json.loads(lines[-1]) if lines else {
                "rep": rep,
                "outcome": "OOM" if is_oom(completed.stderr) else "OTHER_FAILURE",
                "failure_phase": "UNKNOWN",
                "error": completed.stderr[-8000:],
            }
            row["parent_returncode"] = completed.returncode
            row["parent_wall_seconds"] = time.perf_counter() - started
        except subprocess.TimeoutExpired as exc:
            row = {
                "rep": rep,
                "outcome": "TIMEOUT",
                "failure_phase": "UNKNOWN",
                "error": str(exc),
                "parent_wall_seconds": time.perf_counter() - started,
            }
        rows.append(row)
        print(rep, row.get("outcome"), row.get("failure_phase"), flush=True)
    return {
        "status": "OBSERVED",
        "repetitions": repetitions,
        "rows": rows,
        "outcome_counts": {outcome: sum(row.get("outcome") == outcome for row in rows) for outcome in sorted({row.get("outcome") or "UNKNOWN" for row in rows})},
        "failure_phase_counts": {phase: sum((row.get("failure_phase") or "UNKNOWN") == phase for row in rows) for phase in sorted({row.get("failure_phase") or "UNKNOWN" for row in rows})},
        "environment": environment(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--workload-id", choices=sorted(workloads()), default=WORKLOAD_ID)
    parser.add_argument("--budget-bytes", type=int, default=BUDGET_BYTES)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.probe:
        probe(args.rep, args.workload_id)
    if args.output is None:
        parser.error("--output is required in parent mode")
    payload = {
        "status": "OBSERVED",
        "workload_id": args.workload_id,
        "budget_bytes": args.budget_bytes,
        "planner_commit": "135a646",
        "planner_breakdown": planner_breakdown(args.workload_id, args.budget_bytes),
        "reproduction": run_repetitions(args.repetitions, args.timeout, args.workload_id, args.budget_bytes) if args.reproduce else None,
        "environment": environment(),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
