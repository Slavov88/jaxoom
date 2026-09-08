"""Validate current-device budgets without compiling the assessment target.

The default mode is a bounded occupancy and latency pilot. ``--helper`` is an
internal subprocess used to hold a separate JAX allocation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import jaxoom
from jaxoom.device import device_budget


MiB = 1024**2


def workload(x):
    q = x @ x.T
    return q @ x


def pressure_workload(query, key, value):
    scores = jnp.einsum("bhid,bhjd->bhij", query, key)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhij,bhjd->bhid", weights, value)


def snapshot_record(label: str) -> dict:
    snapshot = jaxoom.device_memory()
    budget = device_budget(snapshot)
    assessment = jaxoom.assess(
        pressure_workload,
        *[jax.ShapeDtypeStruct((1, 8, 6144, 64), "float32")] * 3,
        memory_limit="auto",
        device_budget=budget,
    )
    return {
        "label": label,
        "timestamp": snapshot.timestamp,
        "physical_total_bytes": snapshot.physical_total_bytes,
        "driver_used_bytes": snapshot.driver_used_bytes,
        "driver_free_bytes": snapshot.driver_free_bytes,
        "jax_bytes_in_use": snapshot.jax_bytes_in_use,
        "jax_pool_bytes": snapshot.jax_pool_bytes,
        "external_used_bytes": snapshot.external_used_bytes,
        "effective_available_bytes": snapshot.effective_available_bytes,
        "safety_reserve_bytes": budget.safety_reserve_bytes,
        "assessment_budget_bytes": budget.assessment_budget_bytes,
        "structural_bytes": assessment.structural_peak_bytes,
        "central_bytes": assessment.interval.central_bytes,
        "upper_bytes": assessment.interval.upper_bytes,
        "headroom_to_upper_bytes": assessment.headroom_to_upper_bytes,
        "risk": assessment.risk.value if assessment.risk else None,
        "limitations": list(snapshot.limitations) + list(budget.limitations),
    }


def start_holder(bytes_to_hold: int, ready: Path):
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    command = [sys.executable, str(Path(__file__).resolve()), "--helper", str(bytes_to_hold), str(ready)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, env=env, text=True)
    deadline = time.monotonic() + 30
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not ready.exists():
        process.kill()
        raise RuntimeError("GPU allocation helper did not become ready")
    return process


def run_occupancy() -> dict:
    if jax.default_backend() != "gpu":
        return {"status": "SKIPPED", "reason": "CUDA GPU is unavailable"}
    records = [snapshot_record("before")]
    baseline_free = records[0]["driver_free_bytes"] or 0
    levels = [256 * MiB, 768 * MiB, 1280 * MiB]
    levels = [level for level in levels if level < baseline_free // 2]
    with tempfile.TemporaryDirectory(prefix="jaxoom-device-budget-") as temporary:
        for level in levels:
            ready = Path(temporary) / f"ready-{level}"
            holder = start_holder(level, ready)
            try:
                records.append(snapshot_record(f"external_{level // MiB}MiB"))
            finally:
                if holder.stdin:
                    holder.stdin.write("release\n")
                    holder.stdin.flush()
                holder.wait(timeout=30)
            records.append(snapshot_record(f"after_release_{level // MiB}MiB"))
    return {"status": "OBSERVED", "records": records}


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def timed(callable_, repetitions: int = 3) -> float:
    values = []
    for _ in range(repetitions):
        start = time.perf_counter()
        callable_()
        values.append((time.perf_counter() - start) * 1000)
    return median(values)


def run_latency() -> dict:
    abstract = jax.ShapeDtypeStruct((128, 64), "float32")
    concrete = jnp.ones((128, 64), dtype=jnp.float32)
    return {
        "estimate_ms_median": timed(lambda: jaxoom.estimate(workload, abstract), 5),
        "assess_auto_ms_median": timed(lambda: jaxoom.assess(workload, abstract, memory_limit="auto"), 5),
        "device_query_ms_median": timed(jaxoom.device_memory, 5),
        "compile_analyze_ms_median": timed(lambda: jaxoom.compile_analyze(workload, concrete), 3),
        "analyze_donation_ms_median": timed(lambda: jaxoom.analyze_donation(workload, concrete), 3),
    }


def helper(bytes_to_hold: int, ready: Path):
    count = max(1, bytes_to_hold // 4)
    value = jnp.zeros((count,), dtype=jnp.float32)
    value.block_until_ready()
    ready.write_text("ready", encoding="utf-8")
    sys.stdin.readline()
    del value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", nargs=2, metavar=("BYTES", "READY"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.helper:
        helper(int(args.helper[0]), Path(args.helper[1]))
        return
    result = {
        "status": "OBSERVED",
        "environment": {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]) if jax.devices() else None,
        },
        "occupancy": run_occupancy(),
        "latency": run_latency(),
    }
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
