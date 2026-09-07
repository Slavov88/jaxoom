"""Realistic-scale, backend-portable compiler calibration.

Use abstract inputs for analysis/compilation; this script does not execute the
workloads. Results include host, device, allocator, dtype, and scale metadata.
Run with an explicit dated output prefix so earlier experiments are preserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

import jaxoom


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    dtype: str
    config: dict[str, Any]
    fn: Callable[..., Any]
    args: tuple[Any, ...]


def elementwise(depth: int, shape: tuple[int, ...], dtype: str) -> Case:
    def fn(x):
        y = x
        for _ in range(depth):
            y = jnp.tanh(jnp.sin(jnp.exp(y) * 2) + 1)
        return y

    return Case(f"elementwise-d{depth}-{shape[0]}-{dtype}", "elementwise", dtype, {"depth": depth, "shape": shape}, fn, (jax.ShapeDtypeStruct(shape, dtype),))


def matmul(shape: tuple[int, int, int], dtype: str) -> Case:
    m, k, n = shape
    return Case(f"matmul-{m}x{k}x{n}-{dtype}", "matmul", dtype, {"m": m, "k": k, "n": n}, lambda x, w: x @ w, (jax.ShapeDtypeStruct((m, k), dtype), jax.ShapeDtypeStruct((k, n), dtype)))


def residual(depth: int, shape: tuple[int, ...], dtype: str) -> Case:
    def fn(x):
        values = []
        y = x
        for _ in range(depth):
            y = jnp.tanh(y)
            values.append(y)
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result + x

    return Case(f"residual-d{depth}-{shape[0]}-{dtype}", "residual", dtype, {"depth": depth, "shape": shape}, fn, (jax.ShapeDtypeStruct(shape, dtype),))


def mlp(batch: int, width: int, depth: int, dtype: str) -> Case:
    def fn(x, *weights):
        y = x
        for weight in weights[:-1]:
            y = jnp.tanh(y @ weight)
        return y @ weights[-1]

    args: list[Any] = [jax.ShapeDtypeStruct((batch, width), dtype)]
    args.extend(jax.ShapeDtypeStruct((width, width), dtype) for _ in range(depth - 1))
    args.append(jax.ShapeDtypeStruct((width, width // 2), dtype))
    return Case(f"mlp-b{batch}-w{width}-d{depth}-{dtype}", "mlp", dtype, {"batch": batch, "width": width, "depth": depth}, fn, tuple(args))


def attention(batch: int, heads: int, sequence: int, head_dim: int, dtype: str) -> Case:
    width = heads * head_dim

    def fn(x):
        q = x.reshape(batch, sequence, heads, head_dim)
        k = q
        v = q
        scores = jnp.einsum("bshd,bthd->bhst", q, k) / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype))
        weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum("bhst,bthd->bshd", weights, v)
        return output.reshape(batch, sequence, width)

    return Case(f"attention-b{batch}-h{heads}-s{sequence}-d{head_dim}-{dtype}", "attention", dtype, {"batch": batch, "heads": heads, "sequence": sequence, "head_dim": head_dim}, fn, (jax.ShapeDtypeStruct((batch, sequence, width), dtype),))


def transformer(sequence: int, width: int, heads: int, dtype: str) -> Case:
    head_dim = width // heads

    def fn(x, w1, w2):
        q = x.reshape(1, sequence, heads, head_dim)
        scores = jnp.einsum("bshd,bthd->bhst", q, q) / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype))
        attended = jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(scores, axis=-1), q).reshape(1, sequence, width)
        residual = x + attended
        hidden = jnp.tanh(residual @ w1)
        return residual + hidden @ w2

    return Case(f"transformer-s{sequence}-w{width}-{dtype}", "transformer", dtype, {"sequence": sequence, "width": width, "heads": heads}, fn, (jax.ShapeDtypeStruct((1, sequence, width), dtype), jax.ShapeDtypeStruct((width, 4 * width), dtype), jax.ShapeDtypeStruct((4 * width, width), dtype)))


def training_step(batch: int, width: int, dtype: str) -> Case:
    def fn(x, target, w1, b1, w2, b2):
        def loss(w1, b1, w2, b2):
            hidden = jnp.tanh(x @ w1 + b1)
            prediction = hidden @ w2 + b2
            return jnp.mean((prediction - target) ** 2)

        value, grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3))(w1, b1, w2, b2)
        return value, w1 - 0.001 * grads[0], b1 - 0.001 * grads[1], w2 - 0.001 * grads[2], b2 - 0.001 * grads[3]

    return Case(f"training-b{batch}-w{width}-{dtype}", "training", dtype, {"batch": batch, "width": width}, fn, (jax.ShapeDtypeStruct((batch, width), dtype), jax.ShapeDtypeStruct((batch, width // 2), dtype), jax.ShapeDtypeStruct((width, width), dtype), jax.ShapeDtypeStruct((width,), dtype), jax.ShapeDtypeStruct((width, width // 2), dtype), jax.ShapeDtypeStruct((width // 2,), dtype)))


def cases() -> list[Case]:
    result: list[Case] = []
    for dtype in ("float32", "float16"):
        for depth in (1, 8, 16):
            for shape in ((2048, 2048), (4096, 4096)):
                result.append(elementwise(depth, shape, dtype))
        for shape in ((1024, 4096, 1024), (2048, 2048, 2048)):
            result.append(matmul(shape, dtype))
        for depth in (4, 16):
            result.append(residual(depth, (2048, 2048), dtype))
        for batch, width, depth in ((256, 2048, 2), (512, 2048, 4), (1024, 4096, 4)):
            result.append(mlp(batch, width, depth, dtype))
        for sequence in (512, 1024, 2048, 4096):
            result.append(attention(1, 8, sequence, 64, dtype))
        result.append(transformer(512, 512, 8, dtype))
        result.append(training_step(512, 1024, dtype))
    return result


def environment() -> dict[str, Any]:
    devices = jax.devices()
    device = devices[0] if devices else None
    stats = device.memory_stats() if device is not None and hasattr(device, "memory_stats") else None
    return {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "jax_version": jax.__version__,
        "jaxlib_version": _jaxlib_version(),
        "backend": jax.default_backend(),
        "platform": getattr(device, "platform", None),
        "device_kind": getattr(device, "device_kind", None),
        "device_id": getattr(device, "id", None),
        "device": str(device) if device is not None else None,
        "physical_device_memory_bytes": _physical_memory_bytes(),
        "jax_device_memory_limit_bytes": stats.get("bytes_limit") if stats else None,
        "allocator_environment": {key: os.environ.get(key) for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")},
    }


def run_case(case: Case, env: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"name": case.name, "family": case.family, "dtype": case.dtype, "configuration": json.dumps(case.config, sort_keys=True), **env, "status": "ok", "error_message": ""}
    try:
        static = jaxoom.estimate(case.fn, *case.args)
        compiler = jaxoom.compile_analyze(case.fn, *case.args)
        comparison = jaxoom.compare_memory(static, compiler)
        primitive_counts = primitive_composition(case.fn, case.args)
        max_bytes = max(static.estimated_peak_bytes, compiler.compiler_accounted_bytes or 0)
        target = env["jax_device_memory_limit_bytes"] or env["physical_device_memory_bytes"]
        row.update(static_peak_bytes=static.estimated_peak_bytes, equations=static.equations_analyzed, confidence=static.confidence, nested_constructs=";".join(static.unsupported_constructs), primitive_counts=json.dumps(primitive_counts, sort_keys=True), compiler_argument_bytes=compiler.argument_bytes, compiler_output_bytes=compiler.output_bytes, compiler_temp_bytes=compiler.temporary_bytes, compiler_alias_bytes=compiler.alias_bytes, compiler_accounted_bytes=compiler.compiler_accounted_bytes, signed_error_bytes=comparison.signed_difference_bytes, absolute_error_bytes=comparison.absolute_difference_bytes, signed_relative_error=comparison.relative_difference, absolute_relative_error=abs(comparison.relative_difference) if comparison.relative_difference is not None else None, static_compiler_ratio=static.estimated_peak_bytes / compiler.compiler_accounted_bytes if compiler.compiler_accounted_bytes else None, static_shortfall_bytes=max(0, (compiler.compiler_accounted_bytes or 0) - static.estimated_peak_bytes), target_budget_bytes=target, error_fraction_of_target=(comparison.absolute_difference_bytes / target) if comparison.absolute_difference_bytes is not None and target else None, size_bucket=size_bucket(max_bytes), static_overpredicts=comparison.static_overpredicts, static_underpredicts=comparison.static_underpredicts)
        if not compiler.available:
            row["status"] = "compiler_unavailable"
            row["error_message"] = "; ".join(compiler.limitations)
    except Exception as exc:
        row.update(status="failed", error_message=f"{type(exc).__name__}: {exc}")
    return row


def primitive_composition(fn: Callable[..., Any], args: tuple[Any, ...]) -> dict[str, int]:
    closed = jax.make_jaxpr(fn)(*args)
    counts: dict[str, int] = {}
    for eqn in closed.jaxpr.eqns:
        primitive = str(eqn.primitive)
        counts[primitive] = counts.get(primitive, 0) + 1
    return counts


def size_bucket(value: int) -> str:
    mib = value / 1024**2
    if mib < 10:
        return "<10 MiB"
    if mib < 100:
        return "10-100 MiB"
    if mib < 500:
        return "100-500 MiB"
    if mib < 1024:
        return "500 MiB-1 GiB"
    return ">1 GiB"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "ok" and row.get("signed_relative_error") is not None]
    return {"n": len(valid), "overall": summarize(valid), "by_family": grouped(valid, "family"), "by_dtype": grouped(valid, "dtype"), "by_size_bucket": grouped(valid, "size_bucket"), "environment": environment()}


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {value: summarize([row for row in rows if row.get(key) == value]) for value in sorted({row.get(key) for row in rows})}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rel = [float(row["signed_relative_error"]) for row in rows]
    absolute = [int(row["absolute_error_bytes"]) for row in rows]
    signed_bytes = [int(row["signed_error_bytes"]) for row in rows]
    over = [x for x in rel if x > 0]
    under = [x for x in rel if x < 0]
    return {"n": len(rows), "median_absolute_error_bytes": statistics.median(absolute) if absolute else None, "p90_absolute_error_bytes": percentile(absolute, 0.9), "median_absolute_relative_error": statistics.median([abs(x) for x in rel]) if rel else None, "p90_absolute_relative_error": percentile([abs(x) for x in rel], 0.9), "mean_signed_relative_error": statistics.mean(rel) if rel else None, "overprediction_fraction": len(over) / len(rel) if rel else None, "underprediction_fraction": len(under) / len(rel) if rel else None, "worst_absolute_overprediction_bytes": max(signed_bytes) if signed_bytes and max(signed_bytes) > 0 else None, "worst_absolute_underprediction_bytes": min(signed_bytes) if signed_bytes and min(signed_bytes) < 0 else None, "worst_relative_overprediction": max(over) if over else None, "worst_relative_underprediction": min(under) if under else None, "mean_temporary_bytes": statistics.mean([int(row["compiler_temp_bytes"]) for row in rows]) if rows else None}


def percentile(values: list[Any], probability: float) -> Any:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))]


def write_outputs(prefix: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    csv_path, json_path = prefix.with_suffix(".csv"), prefix.with_suffix(".json")
    for path in (csv_path, json_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing result: {path}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jaxlib_version() -> str:
    import jaxlib
    return jaxlib.__version__


def _physical_memory_bytes() -> int | None:
    if jax.default_backend() != "gpu" or shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True, timeout=10)
        mib = int(output.strip().splitlines()[0])
        return mib * 1024**2
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    env = environment()
    rows = [run_case(case, env) for case in cases()]
    summary = aggregate(rows)
    write_outputs(args.output_prefix, rows, summary)
    print(json.dumps({"rows": len(rows), "summary": summary["overall"], "environment": env}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
