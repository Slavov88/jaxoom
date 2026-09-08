"""Experimental runtime-memory validation on isolated GPU subprocesses.

This module is not part of the public API. Each trial runs in a fresh process so
expected failures do not terminate the driver. Allocator high-water marks are
reported as allocator counters, not as universal execution-interval peaks.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

import jaxoom


@dataclass(frozen=True)
class Workload:
    family: str
    name: str
    config: dict[str, Any]
    dtype: str
    fn: Callable[..., Any]
    abstract_args: tuple[Any, ...]


def attention_workload(sequence: int, dtype: str, heads: int = 8, head_dim: int = 64) -> Workload:
    width = heads * head_dim

    def fn(x):
        q = x.reshape(1, sequence, heads, head_dim)
        scores = jnp.einsum("bshd,bthd->bhst", q, q) / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype))
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bhst,bthd->bshd", weights, q).reshape(1, sequence, width)

    return Workload(
        "attention", f"attention-s{sequence}-{dtype}", {"sequence": sequence, "heads": heads, "head_dim": head_dim}, dtype,
        fn, (jax.ShapeDtypeStruct((1, sequence, width), dtype),),
    )


def mlp_workload(batch: int, width: int, depth: int, dtype: str) -> Workload:
    def fn(x, *weights):
        y = x
        for weight in weights[:-1]:
            y = jnp.tanh(y @ weight)
        return y @ weights[-1]

    args: list[Any] = [jax.ShapeDtypeStruct((batch, width), dtype)]
    args.extend(jax.ShapeDtypeStruct((width, width), dtype) for _ in range(depth - 1))
    args.append(jax.ShapeDtypeStruct((width, width // 2), dtype))
    return Workload(
        "mlp", f"mlp-b{batch}-w{width}-d{depth}-{dtype}", {"batch": batch, "width": width, "depth": depth}, dtype,
        fn, tuple(args),
    )


def transformer_workload(sequence: int, width: int, heads: int, dtype: str) -> Workload:
    head_dim = width // heads

    def fn(x, w1, w2):
        q = x.reshape(1, sequence, heads, head_dim)
        scores = jnp.einsum("bshd,bthd->bhst", q, q) / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype))
        attended = jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(scores, axis=-1), q).reshape(1, sequence, width)
        residual = x + attended
        hidden = jnp.tanh(residual @ w1)
        return residual + hidden @ w2

    return Workload(
        "transformer", f"transformer-s{sequence}-w{width}-{dtype}", {"sequence": sequence, "width": width, "heads": heads}, dtype,
        fn, (jax.ShapeDtypeStruct((1, sequence, width), dtype), jax.ShapeDtypeStruct((width, 4 * width), dtype), jax.ShapeDtypeStruct((4 * width, width), dtype)),
    )


def matmul_workload(batch: int, m: int, n: int, k: int, dtype: str) -> Workload:
    def fn(x, weight):
        return jnp.matmul(x, weight)

    return Workload(
        "matmul", f"matmul-b{batch}-m{m}-n{n}-k{k}-{dtype}", {"batch": batch, "m": m, "n": n, "k": k}, dtype,
        fn, (jax.ShapeDtypeStruct((batch, m, k), dtype), jax.ShapeDtypeStruct((k, n), dtype)),
    )


def convolution_workload(batch: int, height: int, width: int, channels: int, out_channels: int, dtype: str) -> Workload:
    def fn(x, kernel):
        return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))

    return Workload(
        "convolution", f"conv-b{batch}-h{height}-w{width}-c{channels}-o{out_channels}-{dtype}", {"batch": batch, "height": height, "width": width, "channels": channels, "out_channels": out_channels}, dtype,
        fn, (jax.ShapeDtypeStruct((batch, height, width, channels), dtype), jax.ShapeDtypeStruct((3, 3, channels, out_channels), dtype)),
    )


def autodiff_workload(batch: int, width: int, layers: int, dtype: str) -> Workload:
    def fn(x, *weights):
        def loss(params):
            y = x
            for weight in params:
                y = jnp.tanh(y @ weight)
            return jnp.mean(y)
        value, grads = jax.value_and_grad(loss)(weights)
        return value, grads

    args: list[Any] = [jax.ShapeDtypeStruct((batch, width), dtype)]
    args.extend(jax.ShapeDtypeStruct((width, width), dtype) for _ in range(layers))
    return Workload(
        "autodiff", f"autodiff-b{batch}-w{width}-l{layers}-{dtype}", {"batch": batch, "width": width, "layers": layers}, dtype,
        fn, tuple(args),
    )


def training_workload(batch: int, width: int, dtype: str) -> Workload:
    def fn(x, target, w1, b1, w2, b2):
        def loss(w1, b1, w2, b2):
            hidden = jnp.tanh(x @ w1 + b1)
            prediction = hidden @ w2 + b2
            return jnp.mean((prediction - target) ** 2)

        value, grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3))(w1, b1, w2, b2)
        return value, w1 - 0.001 * grads[0], b1 - 0.001 * grads[1], w2 - 0.001 * grads[2], b2 - 0.001 * grads[3]

    return Workload(
        "training", f"training-b{batch}-w{width}-{dtype}", {"batch": batch, "width": width}, dtype,
        fn, (jax.ShapeDtypeStruct((batch, width), dtype), jax.ShapeDtypeStruct((batch, width // 2), dtype), jax.ShapeDtypeStruct((width, width), dtype), jax.ShapeDtypeStruct((width,), dtype), jax.ShapeDtypeStruct((width, width // 2), dtype), jax.ShapeDtypeStruct((width // 2,), dtype)),
    )


def workload_from_config(family: str, config: dict[str, Any], dtype: str) -> Workload:
    if family == "attention":
        return attention_workload(config["sequence"], dtype, config.get("heads", 8), config.get("head_dim", 64))
    if family == "mlp":
        return mlp_workload(config["batch"], config["width"], config["depth"], dtype)
    if family == "transformer":
        return transformer_workload(config["sequence"], config["width"], config.get("heads", 8), dtype)
    if family == "training":
        return training_workload(config["batch"], config["width"], dtype)
    if family == "matmul":
        return matmul_workload(config["batch"], config["m"], config["n"], config["k"], dtype)
    if family == "convolution":
        return convolution_workload(config["batch"], config["height"], config["width"], config["channels"], config["out_channels"], dtype)
    if family == "autodiff":
        return autodiff_workload(config["batch"], config["width"], config["layers"], dtype)
    raise ValueError(f"unknown workload family: {family}")


def default_trials() -> list[Workload]:
    trials: list[Workload] = []
    for dtype in ("float32", "float16"):
        for sequence in (512, 1024, 2048, 4096):
            trials.append(attention_workload(sequence, dtype))
        for batch, width, depth in ((256, 2048, 2), (512, 2048, 4), (1024, 4096, 4)):
            trials.append(mlp_workload(batch, width, depth, dtype))
        for sequence, width in ((256, 256), (512, 512)):
            trials.append(transformer_workload(sequence, width, 8, dtype))
        for batch, width in ((256, 1024), (512, 1024)):
            trials.append(training_workload(batch, width, dtype))
    trials.append(attention_workload(8192, "float32"))
    return trials


def memory_stats(device: Any) -> dict[str, int] | None:
    if not hasattr(device, "memory_stats"):
        return None
    stats = device.memory_stats()
    return {key: int(value) for key, value in stats.items()}


def base_metadata(device: Any) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "jax_version": jax.__version__,
        "jaxlib_version": _jaxlib_version(),
        "backend": jax.default_backend(),
        "platform": getattr(device, "platform", None),
        "device": str(device),
        "device_kind": getattr(device, "device_kind", None),
        "device_id": getattr(device, "id", None),
        "physical_vram_bytes": physical_vram_bytes(),
        "allocator_environment": {key: os.environ.get(key) for key in allocator_keys()},
    }


def child_trial(workload: Workload) -> dict[str, Any]:
    device = jax.devices()[0]
    static = jaxoom.estimate(workload.fn, *workload.abstract_args)
    interval = jaxoom.calibrate(static, backend=jax.default_backend())
    row: dict[str, Any] = {
        **base_metadata(device),
        "family": workload.family,
        "name": workload.name,
        "configuration": workload.config,
        "dtype": workload.dtype,
        "structural_peak_bytes": static.estimated_peak_bytes,
        "calibrated_lower_bytes": interval.lower_bytes,
        "calibrated_central_bytes": interval.central_bytes,
        "calibrated_upper_bytes": interval.upper_bytes,
        "calibration_dataset": interval.dataset_version,
        "calibration_applicability": interval.applicability,
        "compiler_argument_bytes": None,
        "compiler_output_bytes": None,
        "compiler_temp_bytes": None,
        "compiler_alias_bytes": None,
        "compiler_accounted_bytes": None,
        "baseline_memory_stats": memory_stats(device),
        "after_input_memory_stats": None,
        "after_lower_memory_stats": None,
        "after_compile_memory_stats": None,
        "after_execution_memory_stats": None,
        "final_memory_stats": None,
        "allocator_peak_bytes_in_use": None,
        "allocator_pool_bytes": None,
        "allocator_bytes_limit": None,
        "profile_bytes": None,
        "status": None,
        "failure_stage": None,
        "failure_message": None,
    }
    inputs: tuple[Any, ...] | None = None
    try:
        inputs = concrete_inputs(workload.abstract_args, workload.dtype)
        block_tree(inputs)
        row["after_input_memory_stats"] = memory_stats(device)
    except Exception as exc:
        return failure_row(row, "input", exc)

    try:
        lowered = jax.jit(workload.fn).lower(*workload.abstract_args)
        row["after_lower_memory_stats"] = memory_stats(device)
        compiled = lowered.compile()
        row["after_compile_memory_stats"] = memory_stats(device)
        stats = compiled.memory_analysis()
        if stats is not None:
            row.update(
                compiler_argument_bytes=_field(stats, "argument_size_in_bytes"),
                compiler_output_bytes=_field(stats, "output_size_in_bytes"),
                compiler_temp_bytes=_field(stats, "temp_size_in_bytes"),
                compiler_alias_bytes=_field(stats, "alias_size_in_bytes"),
            )
            values = [row[key] for key in ("compiler_argument_bytes", "compiler_output_bytes", "compiler_temp_bytes", "compiler_alias_bytes")]
            if all(value is not None for value in values):
                row["compiler_accounted_bytes"] = values[0] + values[1] + values[2] - values[3]
    except Exception as exc:
        return failure_row(row, "compile", exc)

    try:
        output = compiled(*inputs)
        block_tree(output)
        row["after_execution_memory_stats"] = memory_stats(device)
        row["status"] = "FIT"
        profile = jax.profiler.device_memory_profile()
        row["profile_bytes"] = len(profile)
        del output, compiled, lowered, inputs
        gc.collect()
        row["final_memory_stats"] = memory_stats(device)
        peak = row["after_execution_memory_stats"] or {}
        row["allocator_peak_bytes_in_use"] = peak.get("peak_bytes_in_use")
        row["allocator_pool_bytes"] = peak.get("pool_bytes")
        row["allocator_bytes_limit"] = peak.get("bytes_limit")
        add_derived_metrics(row)
        return row
    except Exception as exc:
        return failure_row(row, "execute", exc)


def failure_row(row: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "out of memory" in lowered or "resource_exhausted" in lowered or "cuda_error_out_of_memory" in lowered:
        status = "COMPILE_OOM" if stage == "compile" else "EXECUTION_OOM" if stage == "execute" else "OTHER_FAILURE"
    else:
        status = "OTHER_FAILURE"
    row.update(status=status, failure_stage=stage, failure_message=text[:4000])
    stats = row.get("after_compile_memory_stats") or row.get("after_execution_memory_stats") or row.get("after_input_memory_stats") or {}
    row["allocator_peak_bytes_in_use"] = stats.get("peak_bytes_in_use")
    row["allocator_pool_bytes"] = stats.get("pool_bytes")
    row["allocator_bytes_limit"] = stats.get("bytes_limit")
    add_derived_metrics(row)
    return row


def add_derived_metrics(row: dict[str, Any]) -> None:
    compiler = row.get("compiler_accounted_bytes")
    peak = row.get("allocator_peak_bytes_in_use")
    structural = row["structural_peak_bytes"]
    upper = row["calibrated_upper_bytes"]
    row["runtime_overhead_bytes"] = peak - compiler if peak is not None and compiler is not None else None
    row["allocator_peak_over_compiler"] = peak / compiler if peak is not None and compiler else None
    row["allocator_peak_over_structural"] = peak / structural if peak is not None and structural else None
    row["allocator_peak_over_calibrated_upper"] = peak / upper if peak is not None and upper else None


def concrete_inputs(abstract_args: tuple[Any, ...], dtype: str) -> tuple[Any, ...]:
    return tuple(jnp.ones(tuple(arg.shape), dtype=dtype) for arg in abstract_args)


def block_tree(value: Any) -> None:
    jax.tree_util.tree_map(lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item, value)


def _field(stats: Any, name: str) -> int | None:
    value = getattr(stats, name, None)
    return int(value) if value is not None else None


def allocator_keys() -> tuple[str, ...]:
    return ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")


def physical_vram_bytes() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True, timeout=10)
        return int(output.strip().splitlines()[0]) * 1024**2
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _jaxlib_version() -> str:
    import jaxlib
    return jaxlib.__version__


def classify_subprocess_output(completed: subprocess.CompletedProcess[str], fallback: Workload) -> dict[str, Any]:
    lines = completed.stdout.strip().splitlines()
    parsed: dict[str, Any] = {}
    if lines:
        try:
            parsed = json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    if parsed.get("status") in {"FIT", "COMPILE_OOM", "EXECUTION_OOM", "OTHER_FAILURE"}:
        return parsed
    return {
        "family": fallback.family,
        "name": fallback.name,
        "configuration": fallback.config,
        "dtype": fallback.dtype,
        "status": "OTHER_FAILURE",
        "failure_message": (completed.stderr or completed.stdout)[-4000:],
    }


def run_parent(output: Path, trials: list[Workload], timeout: int) -> None:
    rows: list[dict[str, Any]] = []
    for workload in trials:
        command = [sys.executable, str(Path(__file__).resolve()), "--trial", workload.family, "--config", json.dumps(workload.config), "--dtype", workload.dtype]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            row = classify_subprocess_output(completed, workload)
            row["subprocess_returncode"] = completed.returncode
            if completed.returncode != 0:
                row["subprocess_stderr_tail"] = completed.stderr[-4000:]
        except subprocess.TimeoutExpired as exc:
            row = {"family": workload.family, "name": workload.name, "configuration": workload.config, "dtype": workload.dtype, "status": "OTHER_FAILURE", "failure_stage": "driver_timeout", "failure_message": str(exc)}
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    payload = {
        "environment": {"python_version": platform.python_version(), "jax_version": jax.__version__, "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()]},
        "trials": rows,
        "measurement_semantics": "memory_stats fields are allocator counters; peak_bytes_in_use is a process high-water mark that includes compilation.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", choices=("attention", "mlp", "transformer", "training", "matmul", "convolution", "autodiff"))
    parser.add_argument("--config")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()
    if args.trial:
        config = json.loads(args.config or "{}")
        workload = workload_from_config(args.trial, config, args.dtype)
        print(json.dumps(child_trial(workload), sort_keys=True), flush=True)
        return
    if args.output is None:
        parser.error("--output is required in parent mode")
    run_parent(args.output, default_trials(), args.timeout)


if __name__ == "__main__":
    main()
