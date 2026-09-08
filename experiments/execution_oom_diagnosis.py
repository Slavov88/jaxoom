"""Diagnostic probes for execution-time GPU OOMs.

This is experimental infrastructure. Target trials and contention holders run
in separate processes. It does not alter the public predictor.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def cuda_mem_info() -> dict[str, Any]:
    names = []
    found = ctypes.util.find_library("cudart")
    if found:
        names.append(found)
    names.extend(("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"))
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            fn = lib.cudaMemGetInfo
            fn.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            fn.restype = ctypes.c_int
            free = ctypes.c_size_t()
            total = ctypes.c_size_t()
            code = int(fn(ctypes.byref(free), ctypes.byref(total)))
            if code == 0:
                return {"available": True, "free_bytes": int(free.value), "total_bytes": int(total.value), "library": name}
            return {"available": False, "error_code": code, "library": name}
        except (OSError, AttributeError):
            continue
    return {"available": False, "reason": "CUDA runtime library unavailable"}


def stats(device: Any) -> dict[str, int] | None:
    try:
        values = device.memory_stats() if hasattr(device, "memory_stats") else None
        return {str(key): int(value) for key, value in (values or {}).items()}
    except Exception:
        return None


def snapshot(label: str, device: Any) -> dict[str, Any]:
    import jaxoom
    value = jaxoom.device_memory()
    return {
        "label": label,
        "timestamp": value.timestamp,
        "driver_free_bytes": value.driver_free_bytes,
        "driver_used_bytes": value.driver_used_bytes,
        "physical_total_bytes": value.physical_total_bytes,
        "jax_bytes_in_use": value.jax_bytes_in_use,
        "jax_peak_bytes_in_use": value.jax_peak_bytes_in_use,
        "jax_pool_bytes": value.jax_pool_bytes,
        "external_used_bytes": value.external_used_bytes,
        "effective_available_bytes": value.effective_available_bytes,
        "cuda_mem_get_info": cuda_mem_info(),
        "jax_memory_stats": stats(device),
    }


def block(value: Any) -> None:
    import jax
    jax.tree_util.tree_map(lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item, value)


def parse_oom(message: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    patterns = (
        ("requested_bytes", r"(?:allocate|allocation of)\s+([0-9.]+)\s*(GiB|MiB|KiB|bytes)"),
        ("requested_bytes", r"Tried to allocate\s+([0-9.]+)\s*(GiB|MiB|KiB|bytes)"),
    )
    for key, pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if not match:
            continue
        amount, unit = match.groups()
        multiplier = {"gib": 1024**3, "mib": 1024**2, "kib": 1024, "bytes": 1}[unit.lower()]
        result[key] = int(float(amount) * multiplier)
        break
    for key, pattern in (("allocator_free_bytes", r"free\s*[:=]\s*([0-9]+)"), ("largest_free_block_bytes", r"largest[^0-9]*([0-9]+)")):
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            result[key] = int(match.group(1))
    return result


def workload(family: str, config: dict[str, Any], dtype: str):
    from runtime_validation import workload_from_config
    return workload_from_config(family, config, dtype)


def child_trial(family: str, config: dict[str, Any], dtype: str, pressure: int, repetitions: int) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import jaxoom
    from jaxoom.device import device_budget

    case = workload(family, config, dtype)
    device = jax.devices()[0]
    static = jaxoom.estimate(case.fn, *case.abstract_args)
    interval = jaxoom.calibrate(static, backend=jax.default_backend())
    before_inputs = snapshot("before_inputs", device)
    inputs = tuple(jnp.ones(tuple(arg.shape), dtype=dtype) for arg in case.abstract_args)
    block(inputs)
    after_inputs = snapshot("after_inputs", device)
    budget = device_budget(jaxoom.device_memory())
    assessment = jaxoom.assess(static, memory_limit=budget.assessment_budget_bytes, device_budget=budget, backend=jax.default_backend())
    precompile = snapshot("immediately_before_compile", device)
    row: dict[str, Any] = {
        "family": family,
        "configuration": config,
        "dtype": dtype,
        "external_pressure_bytes": pressure,
        "jax_version": jax.__version__,
        "device": str(device),
        "snapshots": [before_inputs, after_inputs, precompile],
        "structural_bytes": static.estimated_peak_bytes,
        "calibrated_lower_bytes": interval.lower_bytes,
        "calibrated_central_bytes": interval.central_bytes,
        "calibrated_upper_bytes": interval.upper_bytes,
        "calibration_applicability": interval.applicability,
        "assessment_budget_bytes": budget.assessment_budget_bytes,
        "safety_reserve_bytes": budget.safety_reserve_bytes,
        "risk": assessment.risk.value if assessment.risk else None,
        "upper_headroom_bytes": assessment.headroom_to_upper_bytes,
        "compiler_memory": None,
        "hlo": None,
        "compile_status": None,
        "execution_statuses": [],
        "timings_ms": {},
        "diagnostics": [],
    }
    compile_start = time.perf_counter_ns()
    try:
        lowered = jax.jit(case.fn).lower(*inputs)
        hlo_text = lowered.as_text()
        custom_calls = sorted(set(re.findall(r"custom_call_target=\"([^\"]+)\"", hlo_text)))
        row["hlo"] = {"text_sha256": hashlib.sha256(hlo_text.encode()).hexdigest(), "text_bytes": len(hlo_text), "custom_call_targets": custom_calls}
        compiled = lowered.compile()
        row["compile_status"] = "SUCCESS"
        row["timings_ms"]["compile"] = (time.perf_counter_ns() - compile_start) / 1e6
        memory = compiled.memory_analysis()
        if memory is not None:
            values = {"argument_bytes": getattr(memory, "argument_size_in_bytes", None), "output_bytes": getattr(memory, "output_size_in_bytes", None), "temporary_bytes": getattr(memory, "temp_size_in_bytes", None), "alias_bytes": getattr(memory, "alias_size_in_bytes", None)}
            values = {key: int(value) if value is not None else None for key, value in values.items()}
            if all(values[key] is not None for key in values):
                values["compiler_accounted_bytes"] = values["argument_bytes"] + values["output_bytes"] + values["temporary_bytes"] - values["alias_bytes"]
            row["compiler_memory"] = values
        row["snapshots"].append(snapshot("after_compile", device))
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        row["compile_status"] = "COMPILE_OOM" if "out of memory" in message.lower() or "resource_exhausted" in message.lower() else "OTHER_FAILURE"
        row["diagnostics"].append({"stage": "compile", "message": message[:2000], "parsed_oom": parse_oom(message)})
        return row
    for index in range(repetitions):
        start = time.perf_counter_ns()
        try:
            output = compiled(*inputs)
            block(output)
            row["execution_statuses"].append("FIT")
            row["timings_ms"][f"execution_{index + 1}"] = (time.perf_counter_ns() - start) / 1e6
            row["snapshots"].append(snapshot(f"after_execution_{index + 1}", device))
            del output
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            status = "EXECUTION_OOM" if "out of memory" in message.lower() or "resource_exhausted" in message.lower() else "OTHER_FAILURE"
            row["execution_statuses"].append(status)
            row["diagnostics"].append({"stage": "execute", "iteration": index + 1, "message": message[:2000], "parsed_oom": parse_oom(message)})
            row["snapshots"].append(snapshot(f"after_execution_{index + 1}_failure", device))
            break
    return row


def capacity_probe(output: Path, chunk_bytes: int = 256 * 1024**2, max_chunks: int = 16) -> None:
    import numpy as np
    import jax
    import jaxoom

    held = []
    rows = []
    device = jax.devices()[0]
    for index in range(1, max_chunks + 1):
        try:
            value = jax.device_put(np.zeros((chunk_bytes // 4,), dtype=np.float32))
            value.block_until_ready()
            held.append(value)
            rows.append({"chunks": index, "allocated_bytes": index * chunk_bytes, "status": "FIT", "snapshot": snapshot(f"after_{index}_chunks", device)})
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            rows.append({"chunks": index, "allocated_bytes": index * chunk_bytes, "status": "OOM" if "out of memory" in message.lower() or "resource_exhausted" in message.lower() else "OTHER_FAILURE", "message": message[:2000], "parsed_oom": parse_oom(message), "snapshot": snapshot(f"after_{index}_failure", device)})
            break
    output.write_text(json.dumps({"status": "OBSERVED", "chunk_bytes": chunk_bytes, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def holder(bytes_to_hold: int, ready: Path) -> None:
    import jax.numpy as jnp
    value = jnp.zeros((max(1, bytes_to_hold // 4),), dtype=jnp.float32)
    value.block_until_ready()
    ready.write_text("ready", encoding="utf-8")
    sys.stdin.readline()


def run_one(family: str, config: dict[str, Any], dtype: str, pressure: int, repetitions: int, timeout: int) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--trial", family, "--config", json.dumps(config), "--dtype", dtype, "--pressure", str(pressure), "--repetitions", str(repetitions)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    return {"status": "OTHER_FAILURE", "message": (completed.stderr or completed.stdout)[-2000:], "returncode": completed.returncode}


def run_threshold(output: Path, timeout: int) -> None:
    cases = [
        ("attention", {"sequence": 1024, "heads": 8, "head_dim": 64}, "float32"),
        ("attention", {"sequence": 2048, "heads": 8, "head_dim": 64}, "float32"),
        ("attention", {"sequence": 3072, "heads": 8, "head_dim": 64}, "float32"),
        ("attention", {"sequence": 4096, "heads": 8, "head_dim": 64}, "float32"),
        ("attention", {"sequence": 2048, "heads": 8, "head_dim": 64}, "float16"),
        ("mlp", {"batch": 256, "width": 8192, "depth": 4}, "float32"),
        ("training", {"batch": 256, "width": 8192}, "float32"),
    ]
    pressures = (0, 768 * 1024**2, 1280 * 1024**2)
    rows = []
    with tempfile.TemporaryDirectory(prefix="jaxoom-execution-diagnosis-") as temporary:
        for index, (family, config, dtype) in enumerate(cases):
            for pressure in pressures:
                holder_process = None
                try:
                    if pressure:
                        ready = Path(temporary) / f"ready-{index}-{pressure}"
                        env = os.environ.copy()
                        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                        holder_process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--holder", str(pressure), str(ready)], stdin=subprocess.PIPE, text=True, env=env)
                        deadline = time.monotonic() + 45
                        while not ready.exists() and time.monotonic() < deadline:
                            time.sleep(0.1)
                    row = run_one(family, config, dtype, pressure, 3, timeout)
                    row["pressure_requested_bytes"] = pressure
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                finally:
                    if holder_process is not None:
                        if holder_process.stdin:
                            holder_process.stdin.write("release\n")
                            holder_process.stdin.flush()
                        holder_process.wait(timeout=30)
    output.write_text(json.dumps({"status": "OBSERVED", "trials": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial")
    parser.add_argument("--config")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--pressure", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--holder", nargs=2, metavar=("BYTES", "READY"))
    parser.add_argument("--threshold-output", type=Path)
    parser.add_argument("--capacity-output", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.holder:
        holder(int(args.holder[0]), Path(args.holder[1]))
    elif args.trial:
        print(json.dumps(child_trial(args.trial, json.loads(args.config), args.dtype, args.pressure, args.repetitions), sort_keys=True), flush=True)
    elif args.threshold_output:
        run_threshold(args.threshold_output, args.timeout)
    elif args.capacity_output:
        capacity_probe(args.capacity_output)
    else:
        parser.error("one of --trial, --holder, or --threshold-output is required")


if __name__ == "__main__":
    main()
