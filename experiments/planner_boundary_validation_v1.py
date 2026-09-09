"""Empirical validation of the existing JaxOOM batch-size planner.

Planning uses abstract inputs and does not compile or execute targets.  Every
runtime probe is delegated to a fresh subprocess through this same module's
``--probe`` mode.  This is intentionally a benchmark harness, not planner
logic.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

import jaxoom


ArrayFactory = Callable[[tuple[int, ...], Any, bool], Any]


def _array(shape: tuple[int, ...], dtype: Any, concrete: bool) -> Any:
    if concrete:
        return jnp.ones(shape, dtype)
    return jax.ShapeDtypeStruct(shape, dtype)


@dataclasses.dataclass(frozen=True)
class Workload:
    workload_id: str
    family: str
    dimensions: dict[str, int]
    fn: Callable[..., Any]
    args: Callable[[int, bool], tuple[Any, ...]]
    dtype: str = "float32"


def _mlp_1_fn(x, w1, w2, w3):
    y = jnp.tanh(x @ w1)
    y = jnp.tanh(y @ w2)
    return y @ w3


def _mlp_2_fn(x, w1, w2):
    return jnp.tanh(jnp.tanh(x @ w1) @ w2)


def _train_1_fn(x, target, w1, w2, b1, b2):
    def loss(w1, w2, b1, b2):
        hidden = jnp.tanh(x @ w1 + b1)
        prediction = hidden @ w2 + b2
        return jnp.mean((prediction - target) ** 2)

    return jax.value_and_grad(loss, argnums=(0, 1, 2, 3))(w1, w2, b1, b2)


def _train_2_fn(x, target, w):
    def loss(w):
        prediction = jnp.tanh(x @ w)
        return jnp.mean((prediction - target) ** 2)

    return jax.value_and_grad(loss)(w)


def _attention_1_fn(q, k, v):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def _attention_2_fn(q, k, v):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    return output + q


def _conv_1_fn(x, k1, k2):
    y = jax.lax.conv_general_dilated(
        x, k1, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    y = jax.nn.relu(y)
    return jax.lax.conv_general_dilated(
        y, k2, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )


def _conv_2_fn(x, k1, k2, k3):
    y = jax.nn.relu(
        jax.lax.conv_general_dilated(
            x, k1, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
    )
    y = jax.nn.relu(
        jax.lax.conv_general_dilated(
            y, k2, (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        )
    )
    return jax.lax.conv_general_dilated(
        y, k3, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )


def workloads() -> dict[str, Workload]:
    f32 = jnp.float32

    def mlp1(b, concrete):
        return (
            _array((b, 256), f32, concrete),
            _array((256, 768), f32, concrete),
            _array((768, 768), f32, concrete),
            _array((768, 256), f32, concrete),
        )

    def mlp2(b, concrete):
        return (
            _array((b, 384), f32, concrete),
            _array((384, 1024), f32, concrete),
            _array((1024, 384), f32, concrete),
        )

    def train1(b, concrete):
        return (
            _array((b, 256), f32, concrete),
            _array((b, 256), f32, concrete),
            _array((256, 768), f32, concrete),
            _array((768, 256), f32, concrete),
            _array((768,), f32, concrete),
            _array((256,), f32, concrete),
        )

    def train2(b, concrete):
        return (
            _array((b, 512), f32, concrete),
            _array((b, 512), f32, concrete),
            _array((512, 512), f32, concrete),
        )

    def att1(b, concrete):
        return tuple(_array((b, 8, 4096, 64), f32, concrete) for _ in range(3))

    def att2(b, concrete):
        return tuple(_array((b, 8, 5120, 64), f32, concrete) for _ in range(3))

    def conv1(b, concrete):
        return (
            _array((b, 64, 64, 3), f32, concrete),
            _array((3, 3, 3, 16), f32, concrete),
            _array((3, 3, 16, 32), f32, concrete),
        )

    def conv2(b, concrete):
        return (
            _array((b, 96, 96, 3), f32, concrete),
            _array((5, 5, 3, 16), f32, concrete),
            _array((3, 3, 16, 32), f32, concrete),
            _array((1, 1, 32, 16), f32, concrete),
        )

    return {
        "MLP-1": Workload("MLP-1", "MLP", {"input": 256, "hidden": 768, "depth": 3}, _mlp_1_fn, mlp1),
        "MLP-2": Workload("MLP-2", "MLP", {"input": 384, "hidden": 1024, "depth": 2}, _mlp_2_fn, mlp2),
        "TRAIN-1": Workload("TRAIN-1", "training", {"input": 256, "hidden": 768, "output": 256}, _train_1_fn, train1),
        "TRAIN-2": Workload("TRAIN-2", "training", {"input": 512, "output": 512}, _train_2_fn, train2),
        "ATT-1": Workload("ATT-1", "attention", {"sequence": 4096, "heads": 8, "head_dim": 64}, _attention_1_fn, att1),
        "ATT-2": Workload("ATT-2", "attention", {"sequence": 5120, "heads": 8, "head_dim": 64}, _attention_2_fn, att2),
        "CONV-1": Workload("CONV-1", "convolution", {"height": 64, "width": 64, "channels": 3, "filters": 32}, _conv_1_fn, conv1),
        "CONV-2": Workload("CONV-2", "convolution", {"height": 96, "width": 96, "channels": 3, "filters": 32}, _conv_2_fn, conv2),
    }


def _snapshot() -> dict[str, Any]:
    try:
        return dataclasses.asdict(jaxoom.device_memory())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
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


def _is_oom(text: str) -> bool:
    text = text.lower()
    return any(
        token in text
        for token in (
            "out of memory",
            "resource exhausted",
            "resource_exhausted",
            "failed to allocate",
            "cuda_error_out_of_memory",
            "allocator",
        )
    )


def _probe(workload_id: str, batch_size: int) -> None:
    workload = workloads()[workload_id]
    started = time.perf_counter()
    before = _snapshot()
    compile_started = time.perf_counter()
    phase = "input_creation"
    try:
        values = workload.args(batch_size, True)
        phase = "compile"
        compiled = jax.jit(workload.fn).lower(*values).compile()
        compile_seconds = time.perf_counter() - compile_started
        after_compile = _snapshot()
        phase = "execute"
        execute_started = time.perf_counter()
        output = compiled(*values)
        jax.tree_util.tree_map(lambda value: value.block_until_ready(), output)
        execute_seconds = time.perf_counter() - execute_started
        after_execute = _snapshot()
        payload = {
            "workload_id": workload_id,
            "batch_size": batch_size,
            "outcome": "FIT",
            "failure_phase": None,
            "error": None,
            "compile_seconds": compile_seconds,
            "execute_seconds": execute_seconds,
            "wall_seconds": time.perf_counter() - started,
            "before": before,
            "after_compile": after_compile,
            "after_execute": after_execute,
            "environment": _environment(),
        }
    except Exception as exc:
        payload = {
            "workload_id": workload_id,
            "batch_size": batch_size,
            "outcome": "OOM" if _is_oom(f"{type(exc).__name__}: {exc}") else "OTHER_FAILURE",
            "failure_phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "compile_seconds": time.perf_counter() - compile_started if phase == "compile" else None,
            "execute_seconds": None,
            "wall_seconds": time.perf_counter() - started,
            "before": before,
            "after_compile": _snapshot() if phase in {"compile", "execute"} else None,
            "after_execute": None,
            "environment": _environment(),
        }
    print(json.dumps(payload, sort_keys=True), flush=True)
    raise SystemExit(0 if payload["outcome"] == "FIT" else 2)


def _run_probe(workload_id: str, batch_size: int, timeout_seconds: float) -> dict[str, Any]:
    command = [sys.executable, __file__, "--probe", "--workload", workload_id, "--batch", str(batch_size)]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "workload_id": workload_id,
            "batch_size": batch_size,
            "outcome": "TIMEOUT",
            "failure_phase": "unknown",
            "error": str(exc),
            "wall_seconds": time.perf_counter() - started,
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload: dict[str, Any]
    if lines:
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    if not payload:
        text = completed.stdout + "\n" + completed.stderr
        payload = {
            "workload_id": workload_id,
            "batch_size": batch_size,
            "outcome": "OOM" if _is_oom(text) else "OTHER_FAILURE",
            "failure_phase": "unknown",
            "error": text[-4000:],
        }
    payload["parent_returncode"] = completed.returncode
    payload["parent_wall_seconds"] = time.perf_counter() - started
    return payload


def _serialize_trial(trial: Any) -> dict[str, Any]:
    if trial.assessment is None:
        return {"batch_size": trial.batch_size, "error": trial.error}
    assessment = trial.assessment
    return {
        "batch_size": trial.batch_size,
        "structural_peak_bytes": assessment.structural_peak_bytes,
        "central_bytes": assessment.interval.central_bytes,
        "upper_bytes": assessment.interval.upper_bytes,
        "memory_limit_bytes": assessment.memory_limit_bytes,
        "risk": assessment.risk.value if assessment.risk else None,
        "calibrated": assessment.calibrated,
    }


def _plan(workload: Workload, budget_bytes: int, max_batch: int, max_evaluations: int) -> Any:
    return jaxoom.plan_batch_size(
        workload.fn,
        lambda batch: workload.args(batch, False),
        memory_limit=budget_bytes,
        min_batch_size=1,
        max_batch_size=max_batch,
        max_evaluations=max_evaluations,
    )


def _classification(planned: dict[str, Any]) -> str:
    if not planned.get("planned_batch_fits", False):
        return "INVALID/INCONCLUSIVE"
    next_outcome = planned.get("next_tested_outcome")
    if next_outcome == "OOM":
        return "EXACT_BOUNDARY"
    if next_outcome == "FIT":
        return "CONSERVATIVE_SAFE"
    return "INVALID/INCONCLUSIVE"


def _probe_sequence(
    workload_id: str, planned_batch: int, max_batch: int, timeout: float
) -> tuple[list[dict[str, Any]], int | None, bool]:
    probes: list[dict[str, Any]] = []
    cache: dict[int, dict[str, Any]] = {}

    def probe(batch: int) -> dict[str, Any]:
        if batch not in cache:
            cache[batch] = _run_probe(workload_id, batch, timeout)
            probes.append(cache[batch])
        return cache[batch]

    first = probe(planned_batch)
    if first["outcome"] != "FIT":
        return probes, None, False
    next_batch = planned_batch + 1
    if next_batch > max_batch:
        return probes, planned_batch, False
    next_result = probe(next_batch)
    if next_result["outcome"] != "FIT":
        return probes, planned_batch, True

    low = next_batch
    high = min(max_batch, max(next_batch + 1, planned_batch * 2))
    while high <= max_batch:
        result = probe(high)
        if result["outcome"] == "FIT":
            low = high
            if high == max_batch:
                return probes, low, False
            high = min(max_batch, high * 2)
        else:
            break
    if high > max_batch or probe(high)["outcome"] == "FIT":
        return probes, low, False
    while high - low > 1:
        middle = (low + high) // 2
        if probe(middle)["outcome"] == "FIT":
            low = middle
        else:
            high = middle
    return probes, low, True


def _run_boundary(args: argparse.Namespace) -> dict[str, Any]:
    workload = workloads()[args.workload]
    planning_started = time.perf_counter()
    plan = _plan(workload, args.budget_bytes, args.max_batch, args.max_evaluations)
    planning_seconds = time.perf_counter() - planning_started
    planned_batch = plan.recommended_batch_size
    row: dict[str, Any] = {
        "id": workload.workload_id,
        "family": workload.family,
        "dimensions": workload.dimensions,
        "dtype": workload.dtype,
        "memory_mode": "explicit",
        "budget_bytes": args.budget_bytes,
        "planned_batch": planned_batch,
        "planned_estimated_bytes": (
            next((trial.assessment.interval.upper_bytes for trial in plan.trials if trial.batch_size == planned_batch and trial.assessment), None)
            if planned_batch is not None
            else None
        ),
        "planner_status": plan.status,
        "planner_evaluations": plan.evaluations,
        "planner_monotonic": plan.monotonic,
        "planner_seconds": planning_seconds,
        "planner_next_riskier": _serialize_trial(plan.next_failing_or_riskier) if plan.next_failing_or_riskier else None,
        "trials": [_serialize_trial(trial) for trial in plan.trials],
        "probe_timeout_seconds": args.probe_timeout,
    }
    if planned_batch is None or args.plan_only:
        row.update({"probes": [], "planned_batch_fits": None, "false_safe": None, "empirical_max_batch": None, "classification": "NOT_RUN"})
        return row
    probes, empirical_max, empirical_max_is_exact = _probe_sequence(
        workload.workload_id, planned_batch, args.max_probe_batch, args.probe_timeout
    )
    next_probe = next((probe for probe in probes if probe["batch_size"] == planned_batch + 1), None)
    planned_probe = next((probe for probe in probes if probe["batch_size"] == planned_batch), None)
    row.update(
        {
            "probes": probes,
            "planned_batch_fits": planned_probe is not None and planned_probe.get("outcome") == "FIT",
            "false_safe": planned_probe is None or planned_probe.get("outcome") != "FIT",
            "next_tested_batch": planned_batch + 1 if next_probe else None,
            "next_tested_outcome": next_probe.get("outcome") if next_probe else None,
            "empirical_max_batch": empirical_max,
            "empirical_max_is_exact": empirical_max_is_exact,
            "classification": "INVALID/INCONCLUSIVE",
        }
    )
    row["classification"] = _classification(row)
    if empirical_max:
        row["planned_empirical_ratio"] = planned_batch / empirical_max
        row["absolute_gap"] = empirical_max - planned_batch
        row["relative_gap"] = (empirical_max - planned_batch) / empirical_max
    else:
        row["planned_empirical_ratio"] = None
        row["absolute_gap"] = None
        row["relative_gap"] = None
    return row


def _run_auto_pressure(args: argparse.Namespace) -> list[dict[str, Any]]:
    workload = workloads()[args.pressure_workload]
    rows = []
    for pressure_bytes in args.pressure_bytes:
        allocation = None
        try:
            before = _snapshot()
            if pressure_bytes:
                allocation = jnp.ones((pressure_bytes // 4,), jnp.float32)
                allocation.block_until_ready()
            occupied = _snapshot()
            started = time.perf_counter()
            plan = jaxoom.plan_batch_size(
                workload.fn,
                lambda batch: workload.args(batch, False),
                memory_limit="auto",
                min_batch_size=1,
                max_batch_size=args.auto_max_batch,
                max_evaluations=args.max_evaluations,
            )
            rows.append(
                {
                    "pressure_bytes_requested": pressure_bytes,
                    "before": before,
                    "after_allocation": occupied,
                    "frozen_auto_budget_bytes": plan.memory_limit_bytes,
                    "recommended_batch": plan.recommended_batch_size,
                    "planner_status": plan.status,
                    "planner_seconds": time.perf_counter() - started,
                    "device_budget": dataclasses.asdict(plan.device_budget) if plan.device_budget else None,
                }
            )
        finally:
            del allocation
            jax.clear_caches()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workload", choices=sorted(workloads()))
    parser.add_argument("--budget-bytes", type=int)
    parser.add_argument("--max-batch", type=int, default=65536)
    parser.add_argument("--max-evaluations", type=int, default=30)
    parser.add_argument("--max-probe-batch", type=int, default=65536)
    parser.add_argument("--probe-timeout", type=float, default=240.0)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--auto-pressure", action="store_true")
    parser.add_argument("--pressure-workload", default="ATT-1", choices=sorted(workloads()))
    parser.add_argument("--pressure-bytes", type=int, nargs="+", default=[0, 256 * 1024**2, 512 * 1024**2])
    parser.add_argument("--auto-max-batch", type=int, default=65536)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.probe:
        if args.workload is None or args.batch is None:
            parser.error("--probe requires --workload and --batch")
        _probe(args.workload, args.batch)
    if args.output is None:
        parser.error("--output is required outside probe mode")
    payload: dict[str, Any] = {
        "status": "OBSERVED",
        "environment": _environment(),
        "commit": _git_commit(),
        "planning_target_compilation": False,
        "planning_target_execution": False,
        "rows": [],
    }
    if args.workload and args.budget_bytes is not None:
        row = _run_boundary(args)
        payload["rows"] = [row]
    elif args.auto_pressure:
        payload["auto_pressure"] = _run_auto_pressure(args)
        payload["pressure_workload"] = args.pressure_workload
    else:
        parser.error("provide --workload/--budget-bytes or --auto-pressure")
    payload["status"] = "COMPUTATIONALLY_VERIFIED"
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    main()
