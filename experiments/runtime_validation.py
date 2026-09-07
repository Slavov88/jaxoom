"""Experimental GPU allocator/fit-boundary validation.

This is intentionally not part of the public API. The parent process launches
one fresh child process per trial so an expected OOM cannot kill the suite.
Reported memory_stats values are allocator counters/snapshots, not a universal
runtime-peak oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import jaxoom


def attention_fn(batch: int, heads: int, sequence: int, head_dim: int, dtype: str):
    width = heads * head_dim

    def fn(x):
        q = x.reshape(batch, sequence, heads, head_dim)
        scores = jnp.einsum("bshd,bthd->bhst", q, q) / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype))
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bhst,bthd->bshd", weights, q).reshape(batch, sequence, width)

    return fn


def child_trial(sequence: int, heads: int, head_dim: int, dtype: str) -> dict[str, Any]:
    fn = attention_fn(1, heads, sequence, head_dim, dtype)
    abstract = jax.ShapeDtypeStruct((1, sequence, heads * head_dim), dtype)
    static = jaxoom.estimate(fn, abstract)
    compiler = jaxoom.compile_analyze(fn, abstract)
    device = jax.devices()[0]
    before = device.memory_stats() if hasattr(device, "memory_stats") else None
    started = time.monotonic()
    try:
        compiled = jax.jit(fn).lower(abstract).compile()
        x = jnp.ones((1, sequence, heads * head_dim), dtype=dtype)
        output = compiled(x)
        output.block_until_ready()
    except Exception as exc:
        return {
            "status": "runtime_failure",
            "sequence": sequence,
            "heads": heads,
            "head_dim": head_dim,
            "dtype": dtype,
            "backend": jax.default_backend(),
            "device": str(device),
            "jax_version": jax.__version__,
            "jaxlib_version": _jaxlib_version(),
            "static_peak_bytes": static.estimated_peak_bytes,
            "compiler_accounted_bytes": compiler.compiler_accounted_bytes,
            "compiler_available": compiler.available,
            "compiler_limitations": compiler.limitations,
            "runtime_error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
        }
    elapsed = time.monotonic() - started
    after = device.memory_stats() if hasattr(device, "memory_stats") else None
    profile_path = f"/tmp/jaxoom-device-memory-{os.getpid()}.prof"
    profile_error = None
    try:
        jax.profiler.save_device_memory_profile(profile_path)
    except Exception as exc:
        profile_error = f"{type(exc).__name__}: {exc}"
    return {
        "status": "success",
        "sequence": sequence,
        "heads": heads,
        "head_dim": head_dim,
        "dtype": dtype,
        "backend": jax.default_backend(),
        "device": str(device),
        "jax_version": jax.__version__,
        "jaxlib_version": _jaxlib_version(),
        "allocator_environment": {key: os.environ.get(key) for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")},
        "static_peak_bytes": static.estimated_peak_bytes,
        "compiler_accounted_bytes": compiler.compiler_accounted_bytes,
        "compiler_temp_bytes": compiler.temporary_bytes,
        "compiler_alias_bytes": compiler.alias_bytes,
        "before_memory_stats": before,
        "after_memory_stats": after,
        "allocator_peak_bytes_in_use": after.get("peak_bytes_in_use") if after else None,
        "elapsed_seconds": elapsed,
        "profile_path": profile_path,
        "profile_error": profile_error,
        "measurement_note": "peak_bytes_in_use is an allocator counter for this process, not claimed as an exact interval peak.",
    }


def run_parent(output: Path, sequences: list[int], heads: int, head_dim: int, dtype: str) -> None:
    trials: list[dict[str, Any]] = []
    for sequence in sequences:
        command = [sys.executable, str(Path(__file__).resolve()), "--trial", str(sequence), "--heads", str(heads), "--head-dim", str(head_dim), "--dtype", dtype]
        started = time.monotonic()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=360)
        elapsed = time.monotonic() - started
        parsed: dict[str, Any]
        if completed.stdout.strip():
            try:
                parsed = json.loads(completed.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}
        if completed.returncode == 0 and parsed.get("status") == "success":
            row = parsed
        else:
            combined = (completed.stdout + "\n" + completed.stderr).strip()
            lower = combined.lower()
            classification = "oom" if completed.returncode < 0 or any(token in lower for token in ("out of memory", "resource_exhausted", "oom", "cuda_error_out_of_memory")) else "failed"
            row = parsed if parsed else {"sequence": sequence, "heads": heads, "head_dim": head_dim, "dtype": dtype}
            row.update(status=classification, returncode=completed.returncode, elapsed_seconds=elapsed, stdout_tail=completed.stdout[-2000:], stderr_tail=completed.stderr[-4000:])
        trials.append(row)
        print(json.dumps(row, sort_keys=True))
        if row["status"] in {"oom", "failed"}:
            break
    payload = {"environment": {"jax_version": jax.__version__, "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()]}, "trials": trials, "search_note": "bounded increasing-sequence search; each trial ran in a fresh subprocess; search stopped at first failure."}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jaxlib_version() -> str:
    import jaxlib
    return jaxlib.__version__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trial is not None:
        try:
            print(json.dumps(child_trial(args.trial, args.heads, args.head_dim, args.dtype), sort_keys=True), flush=True)
        except Exception as exc:
            print(json.dumps({"status": "exception", "type": type(exc).__name__, "message": str(exc)}), flush=True)
            raise
    else:
        if args.output is None:
            parser.error("--output is required for parent mode")
        run_parent(args.output, [1024, 2048, 4096, 8192], args.heads, args.head_dim, args.dtype)


if __name__ == "__main__":
    main()
