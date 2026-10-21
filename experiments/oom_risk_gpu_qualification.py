"""Fresh-process CUDA qualification for the frozen restricted WorkloadSpec.

Run parent on CPU; children explicitly select CUDA. No public API is enabled.
Each plan/result/log is exclusively created. Existing evidence is never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from experiments import oom_risk_applicability as policy
from experiments import oom_risk_workload_replay as replay
from experiments import oom_risk_workload_spec as adapter

ROOT = Path(__file__).resolve().parents[1]


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def primary_rows():
    rows, _ = replay.load_prediction_rows()
    return [r for r in rows if r["replay_panel"] in {n for n, _ in replay.PANELS[:2]}]


def make_plan():
    """Select pilots by smallest structural demand, without runtime outcomes."""
    rows = primary_rows()
    candidates = []
    for r in rows:
        spec = adapter.WorkloadSpec(r["family"], r["configuration"], r["dtype"])
        candidates.append({"spec": spec.to_dict(), "spec_hash": spec.spec_hash,
                           "panel": r["replay_panel"], "candidate_id": r["candidate_id"],
                           "expected_features": {n: float(r["static_features"].get(n, 0.0) if n == "largest_over_budget" else r["static_features"][n])
                                                 for n in adapter.FEATURE_NAMES},
                           "expected_probability": r["v2_probability"],
                           "structural_bytes": r["static_features"]["structural_peak_bytes"]})
    pilots = {}
    for c in sorted(candidates, key=lambda c: (c["structural_bytes"], c["spec_hash"])):
        pilots.setdefault((c["spec"]["family"], c["spec"]["dtype"]), c["spec_hash"])
    pilot_ids = list(pilots.values())
    contract = policy.policy_contract()
    result = {"status": "FROZEN_BEFORE_RUNTIME", "policy_contract": contract,
              "policy_hash": adapter.digest(contract), "pilot_spec_hashes": pilot_ids,
              "candidates": candidates, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "threshold": adapter.FIT_THRESHOLD, "initialization": "zeros", "seed": None,
              "timeout_seconds": 180, "public_api_enabled": False,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    result["plan_hash"] = adapter.digest(result)
    return result


def gpu_snapshot():
    fields = "name,driver_version,memory.total,memory.used,memory.free,utilization.gpu"
    gpu = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         check=True, capture_output=True, text=True, timeout=10)
    procs = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           check=True, capture_output=True, text=True, timeout=10)
    lines = [line for line in gpu.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("qualification requires exactly one NVIDIA device")
    name, driver, total, used, free, util = (x.strip() for x in lines[0].split(","))
    return {"device_kind": name, "driver_version": driver, "physical_memory_bytes": int(total) * 1024**2,
            "used_memory_bytes": int(used) * 1024**2, "free_memory_bytes": int(free) * 1024**2,
            "utilization_percent": int(util),
            "compute_process_count": len([p for p in procs.stdout.splitlines() if p.strip()]),
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_gpu": gpu.stdout, "raw_processes": procs.stdout}


def classify_error(error, phase):
    text = str(error).lower()
    if any(token in text for token in ("resource_exhausted", "out of memory", "cuda_error_out_of_memory", "cublas_status_alloc_failed")):
        return {"initialization": "INITIALIZATION_OOM", "compile": "COMPILE_OOM", "execute": "EXECUTION_OOM"}.get(phase, "OTHER_FAILURE")
    return "OTHER_FAILURE"


def emit_phase(phase):
    print(json.dumps({"event": "phase", "phase": phase}), flush=True)


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    saved_hash = plan.pop("plan_hash")
    if adapter.digest(plan) != saved_hash:
        raise RuntimeError("qualification plan hash mismatch")
    plan["plan_hash"] = saved_hash
    if plan["policy_hash"] != adapter.digest(policy.policy_contract()):
        raise RuntimeError("qualification policy/source changed after plan freeze")
    if plan["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("qualification runner changed after plan freeze")
    return plan


def run_one(plan, spec_hash, evidence):
    import jax
    import jax.numpy as jnp
    import jaxlib

    c = next(c for c in plan["candidates"] if c["spec_hash"] == spec_hash)
    spec = adapter.WorkloadSpec.from_dict(c["spec"])
    phase = "preflight"
    started = time.monotonic()
    row = {"spec_hash": spec_hash, "spec": spec.to_dict(), "plan_hash": plan["plan_hash"],
           "policy_hash": plan["policy_hash"], "expected_probability": c["expected_probability"],
           "likely_fit": c["expected_probability"] <= adapter.FIT_THRESHOLD,
           "outcome": "OTHER_FAILURE", "pid": os.getpid(), "gpu_runtime_executed": False}
    try:
        # Observe idle state BEFORE CUDA context creation, as in the historical campaign.
        before = gpu_snapshot()
        fractions = [os.environ.get(n) for n in ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION")]
        # Gate the fresh idle observation immediately before CUDA initialization.
        # Configured backend identity must subsequently be confirmed by the
        # actual device/client; it cannot by itself certify a runtime outcome.
        runtime = dict(before, backend="gpu" if os.environ.get("JAX_PLATFORMS") == "cuda" else "UNKNOWN",
                       gpu_vendor="NVIDIA" if "NVIDIA" in before["device_kind"] else "UNKNOWN",
                       device_count=1, jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
                       allocator="bfc" if os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR") in (None, "default") else os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"],
                       preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                       memory_fraction=next((v for v in fractions if v is not None), None),
                       tf_gpu_allocator=os.environ.get("TF_GPU_ALLOCATOR"), xla_flags=os.environ.get("XLA_FLAGS"),
                       jax_enable_x64=bool(jax.config.x64_enabled),
                       default_matmul_precision=jax.config.jax_default_matmul_precision,
                       fresh_process=True, donation=False, sharding=False)
        eligibility = policy.evaluate_applicability(spec, runtime, evidence)
        row.update(runtime=runtime, applicability=eligibility, backend_verified=False,
                   packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                             if d.metadata.get("Name", "").lower().startswith(("jax", "nvidia", "numpy", "scipy"))})
        if eligibility["status"] != "ELIGIBLE_FOR_QUALIFICATION":
            row["outcome"] = "UNSUPPORTED"
            return row
        phase = "initialization"; emit_phase(phase)
        devices = jax.devices("gpu")
        device = devices[0]
        row["platform_version"] = device.client.platform_version
        if (len(devices) != 1 or device.platform != runtime["backend"]
                or device.device_kind != runtime["device_kind"]
                or "cuda" not in device.client.platform_version.lower()):
            row["outcome"] = "UNSUPPORTED_BACKEND_IDENTITY"
            return row
        row["backend_verified"] = True
        fresh = adapter.extract_features(spec)
        score = adapter._score_features(fresh, adapter.load_frozen_model())
        row.update(fresh_features=fresh, fresh_probability=score)
        if any(abs(fresh[n] - c["expected_features"][n]) > 1e-12 for n in adapter.FEATURE_NAMES) or abs(score-c["expected_probability"]) > 1e-12:
            row["outcome"] = "REPLAY_MISMATCH"
            return row
        fn, abstract = adapter.build_workload(spec)
        phase = "initialization"; emit_phase(phase)
        args = tuple(jnp.zeros(a.shape, dtype=a.dtype) for a in abstract)
        jax.tree_util.tree_map(lambda a: a.block_until_ready(), args)
        phase = "compile"; emit_phase(phase)
        executable = jax.jit(fn).lower(*args).compile()
        phase = "execute"; emit_phase(phase)
        row["gpu_runtime_executed"] = True
        outputs = executable(*args)
        leaves = jax.tree_util.tree_leaves(outputs)
        for output in leaves:
            output.block_until_ready()
            if output.devices() != {device}:
                raise RuntimeError("output was not resident on the qualified CUDA device")
        row["outputs"] = [{"shape": list(a.shape), "dtype": str(a.dtype)} for a in leaves]
        row["outcome"] = "FIT"
        phase = "complete"
    except Exception as exc:
        row.update(outcome=classify_error(exc, phase), error=f"{type(exc).__name__}: {exc}")
    finally:
        row.update(phase=phase, wall_seconds=time.monotonic()-started)
        try:
            row["nvidia_after"] = gpu_snapshot()
        except Exception as exc:
            row["nvidia_after_error"] = f"{type(exc).__name__}: {exc}"
    return row


def timeout_result(stdout):
    phase = "preflight"
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict) and event.get("event") == "phase" and event.get("phase") in ("preflight", "initialization", "compile", "execute"):
                phase = event["phase"]
        except (ValueError, TypeError):
            pass
    return {"outcome": phase.upper() + "_TIMEOUT", "phase": phase,
            "gpu_runtime_executed": phase == "execute"}


def trial_passes(row):
    if not isinstance(row, dict) or not isinstance(row.get("applicability"), dict) or row["applicability"].get("status") != "ELIGIBLE_FOR_QUALIFICATION":
        return False
    if row.get("outcome") == "FIT":
        return row.get("gpu_runtime_executed") is True and row.get("phase") == "complete"
    return (row.get("likely_fit") is False and (
        (row.get("outcome") == "EXECUTION_OOM" and row.get("phase") == "execute" and row.get("gpu_runtime_executed") is True)
        or (row.get("outcome") == "COMPILE_OOM" and row.get("phase") == "compile" and row.get("gpu_runtime_executed") is False)))


def validate_child_result(row, candidate, plan):
    """Validate identity and fresh inference before trusting an individual result."""
    try:
        if any(row[k] != expected for k, expected in (
            ("spec_hash", candidate["spec_hash"]), ("spec", candidate["spec"]),
            ("plan_hash", plan["plan_hash"]), ("policy_hash", plan["policy_hash"]),
        )):
            return False
        if row["applicability"]["status"] != "ELIGIBLE_FOR_QUALIFICATION":
            return False
        if row["backend_verified"] is not True or "cuda" not in row["platform_version"].lower():
            return False
        score = row["fresh_probability"]
        if not math.isfinite(score) or abs(score-candidate["expected_probability"]) > 1e-12:
            return False
        fresh = row["fresh_features"]
        return set(fresh) == set(adapter.FEATURE_NAMES) and all(
            math.isfinite(fresh[n]) and abs(fresh[n]-candidate["expected_features"][n]) <= 1e-12
            for n in adapter.FEATURE_NAMES)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def parse_child_result(stdout):
    rows = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and value.get("event") == "result":
                rows.append(value)
        except (ValueError, TypeError):
            pass
    return rows[0] if len(rows) == 1 else {"outcome": "MISSING_OR_DUPLICATE_RESULT"}


def wait_for_idle():
    # The just-finished process may leave a one-second utilization sample. Wait
    # only for that sample to decay; an actual compute process fails immediately.
    for _ in range(12):
        snapshot = gpu_snapshot()
        if snapshot["compute_process_count"] != 0:
            raise RuntimeError("another GPU compute process is active")
        if snapshot["used_memory_bytes"] <= policy.MAX_USED_BYTES and snapshot["free_memory_bytes"] >= policy.MIN_FREE_BYTES and snapshot["utilization_percent"] <= policy.MAX_UTILIZATION:
            return snapshot
        time.sleep(1)
    raise RuntimeError("GPU did not return to the frozen idle state")


def child_env():
    env = dict(os.environ, JAX_PLATFORMS="cuda", CUDA_VISIBLE_DEVICES="0",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONDONTWRITEBYTECODE="1")
    for name in ("JAX_CUDA_VISIBLE_DEVICES", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                 "XLA_PYTHON_CLIENT_ALLOCATOR", "TF_GPU_ALLOCATOR", "XLA_FLAGS"):
        env.pop(name, None)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--child")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.child:
        row = run_one(load_plan(args.plan), args.child, json.loads(args.replay.read_text()))
        print(json.dumps({"event": "result", **row}), flush=True)
        return
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("qualification parent must select JAX_PLATFORMS=cpu")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = make_plan()
    plan_path = args.output_dir / "plan.json"
    write_new(plan_path, plan)
    gpu_replay = args.output_dir / "gpu_replay.json"
    try:
        idle = wait_for_idle()
    except Exception as exc:
        write_new(args.output_dir / "summary.json", {"status": "IDLE_GATE_FAILED", "all_checks_pass": False,
                  "public_api_enabled": False, "runtime_trials": 0, "error": f"{type(exc).__name__}: {exc}"})
        raise SystemExit(1)
    write_new(args.output_dir / "initial_idle_snapshot.json", idle)
    command = [sys.executable, "-m", "experiments.oom_risk_workload_replay", "--output", str(gpu_replay)]
    try:
        replay_proc = subprocess.run(command, capture_output=True, text=True, env=child_env(), timeout=90, cwd=ROOT)
    except subprocess.TimeoutExpired as exc:
        (args.output_dir / "gpu_replay.stdout.txt").write_bytes(exc.stdout or b"")
        (args.output_dir / "gpu_replay.stderr.txt").write_bytes(exc.stderr or b"")
        write_new(args.output_dir / "summary.json", {"status": "GPU_REPLAY_TIMEOUT", "all_checks_pass": False,
                  "public_api_enabled": False, "runtime_trials": 0})
        raise SystemExit(1)
    (args.output_dir / "gpu_replay.stdout.txt").write_text(replay_proc.stdout)
    (args.output_dir / "gpu_replay.stderr.txt").write_text(replay_proc.stderr)
    try:
        replay_evidence = json.loads(gpu_replay.read_text()) if gpu_replay.exists() else {}
        if not isinstance(replay_evidence, dict):
            replay_evidence = {}
    except (ValueError, OSError):
        replay_evidence = {}
    if replay_proc.returncode != 0 or replay_evidence.get("provenance", {}).get("backend") != "gpu" or not replay_evidence.get("fresh_tracing_and_replay_pass"):
        write_new(args.output_dir / "summary.json", {"status": "GPU_REPLAY_FAILED", "all_checks_pass": False, "public_api_enabled": False,
                  "returncode": replay_proc.returncode, "runtime_trials": 0})
        raise SystemExit(1)
    by_id = {c["spec_hash"]: c for c in plan["candidates"]}
    order = plan["pilot_spec_hashes"] + [c["spec_hash"] for c in plan["candidates"]
                                          if c["spec_hash"] not in plan["pilot_spec_hashes"]]
    results = []
    stop = None
    for index, spec_hash in enumerate(order):
        candidate = by_id[spec_hash]
        try:
            wait_for_idle()
        except Exception as exc:
            stop = f"IDLE_GATE: {type(exc).__name__}: {exc}"
            break
        command = [sys.executable, "-m", "experiments.oom_risk_gpu_qualification", "--child", spec_hash,
                   "--plan", str(plan_path), "--replay", str(gpu_replay)]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, env=child_env(),
                                  timeout=plan["timeout_seconds"], cwd=ROOT)
            stdout, stderr = proc.stdout, proc.stderr
            row = parse_child_result(stdout)
            row["returncode"] = proc.returncode
            if proc.returncode != 0:
                row["outcome"] = "CHILD_PROCESS_FAILURE"
            elif row.get("outcome") not in ("MISSING_OR_DUPLICATE_RESULT", "UNSUPPORTED", "REPLAY_MISMATCH") and not validate_child_result(row, candidate, plan):
                row["outcome"] = "INVALID_CHILD_PROVENANCE_OR_REPLAY"
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
            stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
            row = dict(timeout_result(stdout), returncode=None)
        row.update(spec_hash=spec_hash, likely_fit=candidate["expected_probability"] <= adapter.FIT_THRESHOLD,
                   expected_probability=candidate["expected_probability"], command=command,
                   is_pilot=spec_hash in plan["pilot_spec_hashes"], index=index)
        write_new(args.output_dir / f"trial_{index:03d}.json", row)
        (args.output_dir / f"trial_{index:03d}.stdout.txt").write_text(stdout)
        (args.output_dir / f"trial_{index:03d}.stderr.txt").write_text(stderr)
        results.append(row)
        print(index+1, candidate["candidate_id"], candidate["spec"]["family"], candidate["spec"]["dtype"],
              round(candidate["expected_probability"], 6), row["outcome"], flush=True)
        if not trial_passes(row) or (row["is_pilot"] and row["outcome"] != "FIT"):
            stop = row["outcome"]
            break
    passed = len(results) == 200 and stop is None
    write_new(args.output_dir / "summary.json", {
        "status": "GPU_QUALIFICATION_PASS" if passed else "GPU_QUALIFICATION_STOPPED",
        "all_checks_pass": passed, "public_api_enabled": False, "runtime_trials": len(results),
        "pilot_count": len(plan["pilot_spec_hashes"]), "outcomes": dict(Counter(r["outcome"] for r in results)),
        "false_safe_execution_ooms": sum(r["likely_fit"] and r["outcome"] == "EXECUTION_OOM" for r in results),
        "likely_fit_failures": sum(r["likely_fit"] and r["outcome"] != "FIT" for r in results),
        "stop_reason": stop, "plan_hash": plan["plan_hash"], "policy_hash": plan["policy_hash"],
    })
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
