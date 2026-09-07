"""Experimental execution-window memory observations on GPU.

The harness compiles and warms each workload before starting a bounded polling
window. It samples the backend allocator's bytes-in-use counter while already
compiled calls execute. The result is a sampled allocator observation, not an
exact device-memory trace. XProf trace capture is evaluated separately because
its memory viewer is not available for every JAX trace.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import jaxoom
from runtime_validation import (
    Workload,
    base_metadata,
    block_tree,
    concrete_inputs,
    memory_stats,
    mlp_workload,
    attention_workload,
    training_workload,
    transformer_workload,
)


class MemoryPoller:
    def __init__(self, device: Any, interval_seconds: float) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemoryPoller":
        self._thread = threading.Thread(target=self._run, name="jaxoom-memory-poller", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = memory_stats(self.device)
            if sample is not None:
                sample["monotonic_seconds"] = time.monotonic()
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)


def convolution_workload(batch: int, height: int, width: int, channels: int, features: int, dtype: str) -> Workload:
    def fn(x, kernel):
        return jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    config = {"batch": batch, "height": height, "width": width, "channels": channels, "features": features}
    return Workload(
        "convolution",
        f"conv-b{batch}-h{height}-w{width}-c{channels}-f{features}-{dtype}",
        config,
        dtype,
        fn,
        (jax.ShapeDtypeStruct((batch, height, width, channels), dtype), jax.ShapeDtypeStruct((3, 3, channels, features), dtype)),
    )


def fft_workload(batch: int, height: int, width: int, dtype: str) -> Workload:
    def fn(x):
        return jnp.fft.rfftn(x, axes=(-2, -1))

    config = {"batch": batch, "height": height, "width": width}
    return Workload(
        "fft",
        f"fft-b{batch}-h{height}-w{width}-{dtype}",
        config,
        dtype,
        fn,
        (jax.ShapeDtypeStruct((batch, height, width), dtype),),
    )


def workload_from_config(family: str, config: dict[str, Any], dtype: str) -> Workload:
    if family == "convolution":
        return convolution_workload(config["batch"], config["height"], config["width"], config["channels"], config["features"], dtype)
    if family == "fft":
        return fft_workload(config["batch"], config["height"], config["width"], dtype)
    if family == "attention":
        return attention_workload(config["sequence"], dtype, config.get("heads", 8), config.get("head_dim", 64))
    if family == "mlp":
        return mlp_workload(config["batch"], config["width"], config["depth"], dtype)
    if family == "transformer":
        return transformer_workload(config["sequence"], config["width"], config.get("heads", 8), dtype)
    if family == "training":
        return training_workload(config["batch"], config["width"], dtype)
    raise ValueError(f"unknown workload family: {family}")


def default_trials() -> list[Workload]:
    trials: list[Workload] = []
    for dtype in ("float32", "float16"):
        for sequence in (1024, 2048, 4096):
            trials.append(attention_workload(sequence, dtype))
        for batch, width, depth in ((256, 2048, 2), (512, 2048, 4), (1024, 4096, 4)):
            trials.append(mlp_workload(batch, width, depth, dtype))
        for sequence, width in ((256, 256), (512, 512)):
            trials.append(transformer_workload(sequence, width, 8, dtype))
        for batch, width in ((256, 1024), (512, 1024)):
            trials.append(training_workload(batch, width, dtype))
    trials.extend(
        [
            convolution_workload(1, 256, 256, 32, 64, "float32"),
            convolution_workload(1, 512, 512, 32, 64, "float16"),
            fft_workload(4, 512, 512, "float32"),
            fft_workload(4, 1024, 1024, "float32"),
        ]
    )
    return trials


def xprof_version() -> str | None:
    try:
        return importlib.metadata.version("xprof")
    except importlib.metadata.PackageNotFoundError:
        return None


def child_trial(workload: Workload, repetitions: int, poll_interval: float, contention_bytes: int | None = None) -> dict[str, Any]:
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
        "after_warmup_memory_stats": None,
        "pre_execution_snapshot": None,
        "post_execution_snapshot": None,
        "final_memory_stats": None,
        "allocator_peak_bytes_in_use": None,
        "allocator_pool_bytes": None,
        "allocator_bytes_limit": None,
        "execution_peak_bytes": None,
        "execution_peak_source": "memory_stats_bytes_in_use_polling",
        "execution_window_samples": 0,
        "execution_window_seconds": None,
        "execution_repetitions": repetitions,
        "experiment_type": "contention" if contention_bytes is not None else "intrinsic",
        "contention_bytes": contention_bytes,
        "after_contention_memory_stats": None,
        "profile_method": "allocator_bytes_in_use_polling_during_compiled_execution",
        "profile_window": "warmup_complete_to_end_of_repeated_compiled_calls",
        "warmup_completed": False,
        "compile_completed": False,
        "xprof_version": xprof_version(),
        "status": None,
        "failure_stage": None,
        "failure_message": None,
    }

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
        row["compile_completed"] = True
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

    contention = None
    try:
        if contention_bytes is not None:
            if contention_bytes <= 0 or contention_bytes % 4:
                raise ValueError("contention_bytes must be a positive multiple of four")
            contention = jnp.ones((contention_bytes // 4,), dtype=jnp.float32)
            contention.block_until_ready()
            row["after_contention_memory_stats"] = memory_stats(device)
        warmup = compiled(*inputs)
        block_tree(warmup)
        del warmup
        row["after_warmup_memory_stats"] = memory_stats(device)
        row["warmup_completed"] = True
        row["pre_execution_snapshot"] = memory_stats(device)
        started = time.monotonic()
        output = None
        with MemoryPoller(device, poll_interval) as poller:
            for _ in range(repetitions):
                if output is not None:
                    del output
                output = compiled(*inputs)
                block_tree(output)
        row["execution_window_seconds"] = time.monotonic() - started
        row["execution_window_samples"] = len(poller.samples)
        row["post_execution_snapshot"] = memory_stats(device)
        row["status"] = "FIT"
        samples = poller.samples + [row["post_execution_snapshot"]]
        observed = [sample["bytes_in_use"] for sample in samples if sample and "bytes_in_use" in sample]
        row["execution_peak_bytes"] = max(observed) if observed else None
        del output, contention, compiled, lowered, inputs
        gc.collect()
        row["final_memory_stats"] = memory_stats(device)
        peak = row["post_execution_snapshot"] or {}
        row["allocator_peak_bytes_in_use"] = peak.get("peak_bytes_in_use")
        row["allocator_pool_bytes"] = peak.get("pool_bytes")
        row["allocator_bytes_limit"] = peak.get("bytes_limit")
        add_execution_metrics(row)
        return row
    except Exception as exc:
        return failure_row(row, "execute", exc)


def failure_row(row: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    oom = "out of memory" in lowered or "resource_exhausted" in lowered or "cuda_error_out_of_memory" in lowered
    if stage == "compile" and oom:
        status = "COMPILE_OOM"
    elif stage in {"warmup", "execute"} and oom:
        status = "CONTENTION_EXECUTION_OOM" if row.get("contention_bytes") is not None else "EXECUTION_OOM"
    else:
        status = "OTHER_FAILURE"
    row.update(status=status, failure_stage=stage, failure_message=text[:4000])
    stats = row.get("post_execution_snapshot") or row.get("after_contention_memory_stats") or row.get("after_warmup_memory_stats") or row.get("after_compile_memory_stats") or row.get("after_input_memory_stats") or {}
    row["allocator_peak_bytes_in_use"] = stats.get("peak_bytes_in_use")
    row["allocator_pool_bytes"] = stats.get("pool_bytes")
    row["allocator_bytes_limit"] = stats.get("bytes_limit")
    add_execution_metrics(row)
    return row


def add_execution_metrics(row: dict[str, Any]) -> None:
    execution = row.get("execution_peak_bytes")
    compiler = row.get("compiler_accounted_bytes")
    structural = row.get("structural_peak_bytes")
    upper = row.get("calibrated_upper_bytes")
    row["execution_over_compiler_bytes"] = execution - compiler if execution is not None and compiler is not None else None
    row["execution_over_compiler_ratio"] = execution / compiler if execution is not None and compiler else None
    row["execution_over_structural_ratio"] = execution / structural if execution is not None and structural else None
    row["execution_over_calibrated_upper_ratio"] = execution / upper if execution is not None and upper else None
    row["execution_upper_covers_observation"] = execution <= upper if execution is not None and upper is not None else None


def _field(stats: Any, name: str) -> int | None:
    value = getattr(stats, name, None)
    return int(value) if value is not None else None


def classify_subprocess_output(completed: subprocess.CompletedProcess[str], fallback: Workload) -> dict[str, Any]:
    lines = completed.stdout.strip().splitlines()
    if lines:
        try:
            parsed = json.loads(lines[-1])
            if parsed.get("status") in {"FIT", "COMPILE_OOM", "EXECUTION_OOM", "CONTENTION_EXECUTION_OOM", "OTHER_FAILURE"}:
                return parsed
        except json.JSONDecodeError:
            pass
    return {
        "family": fallback.family,
        "name": fallback.name,
        "configuration": fallback.config,
        "dtype": fallback.dtype,
        "status": "OTHER_FAILURE",
        "failure_stage": "subprocess",
        "failure_message": (completed.stderr or completed.stdout)[-4000:],
    }


def run_parent(output: Path, trials: list[Workload], timeout: int, repetitions: int, poll_interval: float) -> None:
    rows: list[dict[str, Any]] = []
    for workload in trials:
        command = [sys.executable, str(Path(__file__).resolve()), "--trial", workload.family, "--config", json.dumps(workload.config), "--dtype", workload.dtype, "--repetitions", str(repetitions), "--poll-interval", str(poll_interval)]
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
        "environment": {
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "xprof_version": xprof_version(),
            "allocator_environment": {key: os.environ.get(key) for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")},
        },
        "trials": rows,
        "measurement_semantics": "execution_peak_bytes is the maximum sampled bytes_in_use during repeated calls to an already compiled executable. It is a sampled allocator observation and can miss shorter-lived allocations.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", choices=("attention", "mlp", "transformer", "training", "convolution", "fft"))
    parser.add_argument("--config")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--contention-bytes", type=int)
    args = parser.parse_args()
    if args.trial:
        workload = workload_from_config(args.trial, json.loads(args.config or "{}"), args.dtype)
        print(json.dumps(child_trial(workload, args.repetitions, args.poll_interval, args.contention_bytes), sort_keys=True), flush=True)
        return
    if args.output is None:
        parser.error("--output is required in parent mode")
    run_parent(args.output, default_trials(), args.timeout, args.repetitions, args.poll_interval)


if __name__ == "__main__":
    main()
