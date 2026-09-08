"""Fresh-process validation of device-aware OOM boundary predictions.

The parent only orchestrates trials. Each target compilation and execution runs
in a new child process. The optional holder is another process used for
controlled GPU contention.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


OOM_WORDS = ("out of memory", "resource_exhausted", "cuda_error_out_of_memory", "cudnn_status_alloc_failed")


def classify_prediction(structural: int | None, upper: int | None, budget: int | None) -> str:
    if budget is None or structural is None or upper is None:
        return "UNAVAILABLE"
    if structural > budget:
        return "PREDICTED EXCEEDS"
    if upper <= budget:
        return "PREDICTED FIT"
    return "UNCERTAIN"


def classify_failure(stage: str, message: str) -> str:
    lowered = message.lower()
    if any(word in lowered for word in OOM_WORDS):
        if stage == "compile":
            return "COMPILE_OOM"
        if stage == "execute":
            return "EXECUTION_OOM"
    return "OTHER_FAILURE"


def empty_outcome_table() -> dict[str, dict[str, int]]:
    return {risk: {outcome: 0 for outcome in ("FIT", "COMPILE_OOM", "EXECUTION_OOM", "OTHER_FAILURE")} for risk in ("LOW", "MODERATE", "HIGH", "LIKELY EXCEEDS BUDGET", "UNAVAILABLE")}


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(row.get("status", "OTHER_FAILURE") for row in trials)
    table = empty_outcome_table()
    false_fit: list[str] = []
    false_oom: list[str] = []
    for row in trials:
        risk = row.get("risk") or "UNAVAILABLE"
        status = row.get("status", "OTHER_FAILURE")
        table.setdefault(risk, {outcome: 0 for outcome in ("FIT", "COMPILE_OOM", "EXECUTION_OOM", "OTHER_FAILURE")})
        table[risk][status] = table[risk].get(status, 0) + 1
        if row.get("prediction") == "PREDICTED FIT" and status in {"COMPILE_OOM", "EXECUTION_OOM"}:
            false_fit.append(row.get("trial_id", "unknown"))
        if row.get("prediction") == "PREDICTED EXCEEDS" and status == "FIT":
            false_oom.append(row.get("trial_id", "unknown"))
    return {
        "trial_count": len(trials),
        "outcomes": dict(outcomes),
        "risk_x_outcome": table,
        "false_fit_ids": false_fit,
        "false_fit_count": len(false_fit),
        "false_oom_ids": false_oom,
        "false_oom_count": len(false_oom),
        "low_trials": table["LOW"],
        "likely_exceeds_trials": table["LIKELY EXCEEDS BUDGET"],
    }


def _workload_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for sequence in (4096, 5120, 6144, 7168, 8192):
        specs.append({"family": "attention", "config": {"sequence": sequence, "heads": 8, "head_dim": 64}, "dtype": "float32"})
    for batch in (256, 512, 1024):
        specs.append({"family": "mlp", "config": {"batch": batch, "width": 8192, "depth": 4}, "dtype": "float32"})
    for batch in (256, 512):
        specs.append({"family": "training", "config": {"batch": batch, "width": 8192}, "dtype": "float32"})
    for sequence in (1024, 2048):
        specs.append({"family": "transformer", "config": {"sequence": sequence, "width": 2048, "heads": 8}, "dtype": "float32"})
    return specs


def _fixed_contention_specs() -> list[dict[str, Any]]:
    return [
        {"family": "attention", "config": {"sequence": 4096, "heads": 8, "head_dim": 64}, "dtype": "float32"},
        {"family": "attention", "config": {"sequence": 6144, "heads": 8, "head_dim": 64}, "dtype": "float32"},
    ]


def _start_holder(bytes_to_hold: int, ready: Path):
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--holder", str(bytes_to_hold), str(ready)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    deadline = time.monotonic() + 45
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not ready.exists():
        process.kill()
        raise RuntimeError(f"holder did not become ready: {(process.stderr.read() if process.stderr else '')[-1000:]}")
    return process


def _run_child(spec: dict[str, Any], track: str, pressure: int, trial_id: str, timeout: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--trial",
        spec["family"],
        "--config",
        json.dumps(spec["config"], sort_keys=True),
        "--dtype",
        spec["dtype"],
        "--track",
        track,
        "--pressure",
        str(pressure),
        "--trial-id",
        trial_id,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"trial_id": trial_id, "track": track, "family": spec["family"], "configuration": spec["config"], "external_pressure_bytes": pressure, "status": "OTHER_FAILURE", "failure_stage": "driver_timeout", "failure_message": str(exc)}
    for line in reversed(completed.stdout.splitlines()):
        try:
            row = json.loads(line)
            if row.get("trial_id") == trial_id:
                row["subprocess_returncode"] = completed.returncode
                if completed.returncode:
                    row["subprocess_stderr_tail"] = completed.stderr[-2000:]
                return row
        except json.JSONDecodeError:
            continue
    return {
        "trial_id": trial_id,
        "track": track,
        "family": spec["family"],
        "configuration": spec["config"],
        "external_pressure_bytes": pressure,
        "status": "OTHER_FAILURE",
        "failure_stage": "child_protocol",
        "failure_message": (completed.stderr or completed.stdout)[-2000:],
        "subprocess_returncode": completed.returncode,
    }


def run_parent(output: Path, timeout: int, mode: str = "all") -> None:
    if os.environ.get("JAX_PLATFORMS") not in (None, "cuda"):
        raise RuntimeError("run this experiment with a CUDA JAX backend")
    rows: list[dict[str, Any]] = []
    if mode in {"all", "intrinsic"}:
        for index, spec in enumerate(_workload_specs()):
            rows.append(_run_child(spec, "intrinsic", 0, f"intrinsic-{index:03d}", timeout))
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    pressure_levels = (0, 768 * 1024**2, 1280 * 1024**2) if mode in {"all", "contention"} else ()
    with tempfile.TemporaryDirectory(prefix="jaxoom-boundary-holder-") as temporary:
        for pressure in pressure_levels:
            for offset, spec in enumerate(_fixed_contention_specs()):
                holder = None
                try:
                    if pressure:
                        holder = _start_holder(pressure, Path(temporary) / f"ready-{pressure}-{offset}")
                    trial_id = f"contention-{pressure // 1024**2:04d}-{offset:03d}"
                    row = _run_child(spec, "contention", pressure, trial_id, timeout)
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                finally:
                    if holder is not None:
                        if holder.stdin:
                            holder.stdin.write("release\n")
                            holder.stdin.flush()
                        try:
                            holder.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            holder.kill()
    metadata = {
        "python_version": platform.python_version(),
        "host_platform": platform.platform(),
        "jax_backend": "recorded per trial",
        "allocator_environment": {key: os.environ.get(key) for key in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR")},
    }
    payload = {"status": "OBSERVED", "metadata": metadata, "trials": rows, "summary": summarize_trials(rows)}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _field(stats: Any, name: str) -> int | None:
    value = getattr(stats, name, None)
    return int(value) if value is not None else None


def _memory_analysis(compiled: Any) -> dict[str, int | None]:
    values = {"argument_bytes": None, "output_bytes": None, "temporary_bytes": None, "alias_bytes": None, "compiler_accounted_bytes": None}
    stats = compiled.memory_analysis()
    if stats is None:
        return values
    values.update(argument_bytes=_field(stats, "argument_size_in_bytes"), output_bytes=_field(stats, "output_size_in_bytes"), temporary_bytes=_field(stats, "temp_size_in_bytes"), alias_bytes=_field(stats, "alias_size_in_bytes"))
    fields = [values[key] for key in ("argument_bytes", "output_bytes", "temporary_bytes", "alias_bytes")]
    if all(value is not None for value in fields):
        values["compiler_accounted_bytes"] = fields[0] + fields[1] + fields[2] - fields[3]
    return values


def _block(value: Any) -> None:
    import jax
    jax.tree_util.tree_map(lambda item: item.block_until_ready() if hasattr(item, "block_until_ready") else item, value)


def child_trial(family: str, config: dict[str, Any], dtype: str, track: str, pressure: int, trial_id: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import jaxoom
    from runtime_validation import workload_from_config
    from jaxoom.device import device_budget

    workload = workload_from_config(family, config, dtype)
    device = jax.devices()[0]
    static = jaxoom.estimate(workload.fn, *workload.abstract_args)
    interval = jaxoom.calibrate(static, backend=jax.default_backend())
    inputs = tuple(jnp.ones(tuple(arg.shape), dtype=dtype) for arg in workload.abstract_args)
    _block(inputs)
    snapshot = jaxoom.device_memory()
    budget = device_budget(snapshot)
    assessment = jaxoom.assess(static, memory_limit=budget.assessment_budget_bytes, device_budget=budget, backend=jax.default_backend())
    prediction = classify_prediction(static.estimated_peak_bytes, interval.upper_bytes, budget.assessment_budget_bytes)
    snapshot_time = time.perf_counter_ns()
    row: dict[str, Any] = {
        "trial_id": trial_id,
        "track": track,
        "family": family,
        "configuration": config,
        "dtype": dtype,
        "external_pressure_bytes": pressure,
        "device": str(device),
        "device_kind": getattr(device, "device_kind", None),
        "jax_version": jax.__version__,
        "jaxlib_version": __import__("jaxlib").__version__,
        "snapshot": {
            "timestamp": snapshot.timestamp,
            "backend": snapshot.backend,
            "device_id": snapshot.device_id,
            "device_uuid": snapshot.device_uuid,
            "physical_total_bytes": snapshot.physical_total_bytes,
            "driver_used_bytes": snapshot.driver_used_bytes,
            "driver_free_bytes": snapshot.driver_free_bytes,
            "jax_bytes_in_use": snapshot.jax_bytes_in_use,
            "jax_peak_bytes_in_use": snapshot.jax_peak_bytes_in_use,
            "jax_pool_bytes": snapshot.jax_pool_bytes,
            "external_used_bytes": snapshot.external_used_bytes,
            "allocator_mode": snapshot.allocator_mode,
            "allocator_preallocate": snapshot.allocator_preallocate,
            "allocator_memory_fraction": snapshot.allocator_memory_fraction,
            "effective_available_bytes": snapshot.effective_available_bytes,
            "measurement_sources": snapshot.measurement_sources,
            "limitations": snapshot.limitations,
        },
        "assessment_budget_bytes": budget.assessment_budget_bytes,
        "safety_reserve_bytes": budget.safety_reserve_bytes,
        "structural_bytes": assessment.structural_peak_bytes,
        "calibrated_lower_bytes": interval.lower_bytes,
        "calibrated_central_bytes": interval.central_bytes,
        "calibrated_upper_bytes": interval.upper_bytes,
        "calibration_applicability": interval.applicability,
        "structural_headroom_bytes": assessment.headroom_to_structural_bytes,
        "central_headroom_bytes": assessment.headroom_to_central_bytes,
        "upper_headroom_bytes": assessment.headroom_to_upper_bytes,
        "upper_headroom_fraction": assessment.headroom_fraction,
        "risk": assessment.risk.value if assessment.risk else None,
        "prediction": prediction,
        "compile_status": None,
        "execution_status": None,
        "status": None,
        "compiler_memory": None,
        "timings_ms": {},
        "snapshot_to_compile_ms": None,
        "failure_stage": None,
        "failure_message": None,
    }
    compile_start = time.perf_counter_ns()
    row["snapshot_to_compile_ms"] = (compile_start - snapshot_time) / 1e6
    row["timings_ms"]["snapshot_and_assessment"] = (snapshot_time - compile_start) / -1e6
    compile_begin = time.perf_counter_ns()
    try:
        lowered = jax.jit(workload.fn).lower(*inputs)
        compiled = lowered.compile()
        row["timings_ms"]["compile"] = (time.perf_counter_ns() - compile_begin) / 1e6
        row["compile_status"] = "SUCCESS"
        row["compiler_memory"] = _memory_analysis(compiled)
    except Exception as exc:
        row["timings_ms"]["compile"] = (time.perf_counter_ns() - compile_begin) / 1e6
        row["compile_status"] = "FAILED"
        row["status"] = classify_failure("compile", f"{type(exc).__name__}: {exc}")
        row["failure_stage"] = "compile"
        row["failure_message"] = f"{type(exc).__name__}: {exc}"[:800]
        return row
    execute_start = time.perf_counter_ns()
    try:
        output = compiled(*inputs)
        _block(output)
        row["execution_status"] = "SUCCESS"
        row["status"] = "FIT"
        row["timings_ms"]["execution"] = (time.perf_counter_ns() - execute_start) / 1e6
    except Exception as exc:
        row["execution_status"] = "FAILED"
        row["status"] = classify_failure("execute", f"{type(exc).__name__}: {exc}")
        row["failure_stage"] = "execute"
        row["failure_message"] = f"{type(exc).__name__}: {exc}"[:800]
    return row


def holder(bytes_to_hold: int, ready: Path) -> None:
    import jax.numpy as jnp
    count = max(1, bytes_to_hold // 4)
    value = jnp.zeros((count,), dtype=jnp.float32)
    value.block_until_ready()
    ready.write_text("ready", encoding="utf-8")
    sys.stdin.readline()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial")
    parser.add_argument("--config")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--track")
    parser.add_argument("--pressure", type=int, default=0)
    parser.add_argument("--trial-id")
    parser.add_argument("--holder", nargs=2, metavar=("BYTES", "READY"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--mode", choices=("all", "intrinsic", "contention"), default="all")
    args = parser.parse_args()
    if args.holder:
        holder(int(args.holder[0]), Path(args.holder[1]))
    elif args.trial:
        row = child_trial(args.trial, json.loads(args.config or "{}"), args.dtype, args.track or "intrinsic", args.pressure, args.trial_id or "trial")
        print(json.dumps(row, sort_keys=True), flush=True)
    else:
        if args.output is None:
            parser.error("--output is required in parent mode")
        run_parent(args.output, args.timeout, args.mode)


if __name__ == "__main__":
    main()
