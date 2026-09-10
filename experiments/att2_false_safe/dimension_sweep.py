"""Small one-axis ATT-2 diagnostic sweep with isolated execution probes."""
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

BUDGET = 3 * 1024**3


def attention(q, k, v):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    return output + q


def args_for(sequence: int, heads: int, head_dim: int, concrete: bool):
    shape = (1, heads, sequence, head_dim)
    return tuple(
        jnp.ones(shape, jnp.float32) if concrete else jax.ShapeDtypeStruct(shape, jnp.float32)
        for _ in range(3)
    )


def snapshot() -> dict[str, Any]:
    try:
        return dataclasses.asdict(jaxoom.device_memory())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def is_oom(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("out of memory", "resource_exhausted", "resource exhausted", "failed to allocate"))


def memory_analysis(compiled: Any) -> dict[str, Any]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {"available": False}
    values = {
        "argument_bytes": int(stats.argument_size_in_bytes),
        "output_bytes": int(stats.output_size_in_bytes),
        "temporary_bytes": int(stats.temp_size_in_bytes),
        "alias_bytes": int(stats.alias_size_in_bytes),
    }
    values["compiler_accounted_bytes"] = values["argument_bytes"] + values["output_bytes"] + values["temporary_bytes"] - values["alias_bytes"]
    values["available"] = True
    return values


def probe(sequence: int, heads: int, head_dim: int) -> None:
    values = args_for(sequence, heads, head_dim, True)
    before = snapshot()
    phase = "INPUT_CREATION"
    try:
        for value in values:
            value.block_until_ready()
        phase = "COMPILE"
        compile_start = time.perf_counter()
        compiled = jax.jit(attention).lower(*values).compile()
        compile_seconds = time.perf_counter() - compile_start
        after_compile = snapshot()
        compiler = memory_analysis(compiled)
        phase = "FIRST_EXECUTION"
        execute_start = time.perf_counter()
        output = compiled(*values)
        jax.tree_util.tree_map(lambda item: item.block_until_ready(), output)
        payload = {"sequence": sequence, "heads": heads, "head_dim": head_dim, "outcome": "FIT", "failure_phase": None, "compile_seconds": compile_seconds, "execute_seconds": time.perf_counter() - execute_start, "before": before, "after_compile": after_compile, "compiler_memory": compiler}
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        payload = {"sequence": sequence, "heads": heads, "head_dim": head_dim, "outcome": "OOM" if is_oom(text) else "OTHER_FAILURE", "failure_phase": phase, "error": text[-8000:], "before": before, "after_compile": snapshot() if phase in {"COMPILE", "FIRST_EXECUTION"} else None, "compiler_memory": locals().get("compiler")}
    print(json.dumps(payload, sort_keys=True), flush=True)
    raise SystemExit(0 if payload["outcome"] == "FIT" else 2)


def run_probe(sequence: int, heads: int, head_dim: int, timeout: int) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--probe", "--sequence", str(sequence), "--heads", str(heads), "--head-dim", str(head_dim)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=os.environ.copy())
    except subprocess.TimeoutExpired as exc:
        return {"sequence": sequence, "heads": heads, "head_dim": head_dim, "outcome": "TIMEOUT", "failure_phase": "UNKNOWN", "error": str(exc)}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if lines:
        try:
            row = json.loads(lines[-1])
        except json.JSONDecodeError:
            row = {"outcome": "OTHER_FAILURE", "failure_phase": "UNKNOWN", "error": lines[-1]}
    else:
        text = (completed.stderr or completed.stdout)[-8000:]
        row = {"sequence": sequence, "heads": heads, "head_dim": head_dim, "outcome": "OOM" if is_oom(text) else "OTHER_FAILURE", "failure_phase": "UNKNOWN", "error": text}
    row["returncode"] = completed.returncode
    return row


def plan(sequence: int, heads: int, head_dim: int) -> dict[str, Any]:
    abstract = args_for(sequence, heads, head_dim, False)
    report = jaxoom.estimate(attention, *abstract)
    assessment = jaxoom.assess(report, memory_limit=BUDGET)
    return {"structural_bytes": report.estimated_peak_bytes, "upper_bytes": assessment.interval.upper_bytes, "central_bytes": assessment.interval.central_bytes, "risk": assessment.risk.value if assessment.risk else None, "input_bytes": sum(int(x.size) * 4 for x in abstract), "peak_primitive": report.peak.primitive}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--axis", choices=("sequence", "heads", "head_dim", "all"), default="sequence")
    args = parser.parse_args()
    if args.probe:
        probe(args.sequence, args.heads, args.head_dim)
    if args.output is None:
        parser.error("--output is required")
    grids = {
        "sequence": [(sequence, 8, 64) for sequence in (2048, 3072, 4096, 4608, 5120)],
        "heads": [(4096, heads, 64) for heads in (4, 6, 8)],
        "head_dim": [(4096, 8, head_dim) for head_dim in (32, 48, 64)],
    }
    rows = []
    selected_grids = grids if args.axis == "all" else {args.axis: grids[args.axis]}
    for axis, points in selected_grids.items():
        for sequence, heads, head_dim in points:
            rows.append({"axis": axis, "sequence": sequence, "heads": heads, "head_dim": head_dim, "planner": plan(sequence, heads, head_dim), "actual": run_probe(sequence, heads, head_dim, args.timeout)})
            print(axis, sequence, heads, head_dim, rows[-1]["actual"].get("outcome"), flush=True)
    args.output.write_text(json.dumps({"status": "OBSERVED", "budget_bytes": BUDGET, "environment": {"jax": jax.__version__, "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()], "allocator": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")}, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
