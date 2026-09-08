"""Targeted, resumable allocator-residency campaign.

Screening uses abstract tracing and calibration only. Threshold probes run in
fresh subprocesses and are never part of normal ``jaxoom.assess`` behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jaxoom
try:
    from .runtime_validation import workload_from_config
except ImportError:
    from runtime_validation import workload_from_config

CAPACITY = 4 * 1024**3

CANDIDATES = [
    (family, config, dtype)
    for dtype in ("float32", "float16")
    for family, configs in {
        "attention": [{"sequence": s, "heads": 8, "head_dim": 64} for s in (256, 512, 1024, 1536, 2048, 3072, 3584, 3840, 4096)],
        "mlp": ([{"batch": b, "width": 2048, "depth": d} for b, d in ((256, 2), (512, 2), (1024, 2), (512, 4), (1024, 4), (2048, 4))]
                + [{"batch": b, "width": 8192, "depth": d} for b, d in ((64, 2), (128, 2), (256, 2), (64, 4), (128, 4))]),
        "training": ([{"batch": b, "width": 2048} for b in (256, 512, 1024, 2048)]
                      + [{"batch": b, "width": 8192} for b in (64, 128, 256, 512)]
                      + [{"batch": b, "width": 4096} for b in (256, 512, 1024)]),
        "transformer": ([{"sequence": s, "width": 1024, "heads": 8} for s in (512, 1024, 1536, 2048, 3072)]
                        + [{"sequence": s, "width": 2048, "heads": 8} for s in (1024, 1536, 2048, 3072)]),
        "matmul": [{"batch": 1, "m": n, "n": n, "k": n} for n in (1024, 1536, 2048, 2560, 3072, 4096, 5120, 8192, 10240)],
        "convolution": ([{"batch": b, "height": s, "width": s, "channels": 32, "out_channels": 64} for b, s in ((1, 256), (2, 256), (4, 256), (1, 384), (2, 384))]
                        + [{"batch": b, "height": s, "width": s, "channels": 64, "out_channels": 128} for b, s in ((1, 512), (2, 512), (4, 512), (1, 768), (2, 768))]),
        "autodiff": ([{"batch": b, "width": 1024, "layers": l} for b, l in ((64, 2), (128, 2), (256, 2), (128, 4), (256, 4))]
                     + [{"batch": b, "width": 4096, "layers": l} for b, l in ((32, 2), (64, 2), (32, 4), (64, 4), (128, 4))]),
    }.items()
    for config in configs
]


def config_id(family: str, config: dict[str, Any], dtype: str) -> str:
    payload = json.dumps({"family": family, "config": config, "dtype": dtype}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def screen_one(family: str, config: dict[str, Any], dtype: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        workload = workload_from_config(family, config, dtype)
        report = jaxoom.estimate(workload.fn, *workload.abstract_args)
        interval = jaxoom.calibrate(report, backend="gpu", jax_version="0.11.0")
        jaxpr = jax.make_jaxpr(workload.fn)(*workload.abstract_args).jaxpr
        primitive_counts: dict[str, int] = {}
        for equation in jaxpr.eqns:
            name = str(equation.primitive)
            primitive_counts[name] = primitive_counts.get(name, 0) + 1
        upper_fraction = interval.upper_bytes / CAPACITY
        status = "THRESHOLD_CANDIDATE" if 0.08 <= upper_fraction <= 0.90 else "SCREENED_OUT_TOO_SMALL" if upper_fraction < 0.08 else "SCREENED_OUT_TOO_LARGE"
        return {
            "configuration_id": config_id(family, config, dtype), "family": family, "configuration": config, "dtype": dtype,
            "status": status, "structural_peak_bytes": report.estimated_peak_bytes,
            "largest_buffer_bytes": max((getattr(item, "size_bytes", getattr(item, "bytes", 0)) for item in report.largest_buffers), default=0),
            "input_bytes": sum(int(__import__("math").prod(arg.shape)) * __import__("numpy").dtype(arg.dtype).itemsize for arg in workload.abstract_args),
            "output_bytes": None, "equation_count": report.equations_analyzed,
            "primitive_counts": primitive_counts, "calibrated_lower_bytes": interval.lower_bytes,
            "calibrated_upper_bytes": interval.upper_bytes, "screen_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {"configuration_id": config_id(family, config, dtype), "family": family, "configuration": config, "dtype": dtype, "status": "OTHER_FAILURE", "failure": f"{type(exc).__name__}: {exc}", "screen_seconds": time.perf_counter() - started}


def screen(output: Path) -> list[dict[str, Any]]:
    existing: dict[str, Any] = {}
    if output.exists():
        existing = {row["configuration_id"]: row for row in json.loads(output.read_text()).get("rows", [])}
    rows = list(existing.values())
    for family, config, dtype in CANDIDATES:
        identifier = config_id(family, config, dtype)
        if identifier in existing:
            continue
        row = screen_one(family, config, dtype)
        rows.append(row)
        output.write_text(json.dumps({"status": "OBSERVED", "phase": "screening", "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def run_thresholds(screening: list[dict[str, Any]], output: Path, timeout: int, families: set[str] | None = None, dtypes: set[str] | None = None, fractions_override: tuple[float, ...] | None = None) -> None:
    selected = [row for row in screening if row["status"] == "THRESHOLD_CANDIDATE" and (families is None or row["family"] in families) and (dtypes is None or row["dtype"] in dtypes)]
    rows: list[dict[str, Any]] = []
    existing = {}
    if output.exists():
        existing = {(row.get("configuration_id"), row.get("configured_fraction")): row for row in json.loads(output.read_text()).get("rows", [])}
        rows = list(existing.values())
    for candidate in selected:
        upper = candidate["calibrated_upper_bytes"]
        fractions = fractions_override or ((0.35, 0.45, 0.55, 0.65, 0.75) if upper < 2.0 * 1024**3 else (0.55, 0.65, 0.75, 0.85))
        for fraction in fractions:
            key = (candidate["configuration_id"], fraction)
            if key in existing:
                continue
            env = os.environ.copy()
            env["XLA_CLIENT_MEM_FRACTION"] = str(fraction)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
            command = [sys.executable, str(Path(__file__).with_name("execution_oom_diagnosis.py")), "--trial", candidate["family"], "--config", json.dumps(candidate["configuration"]), "--dtype", candidate["dtype"], "--repetitions", "1"]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
                parsed = next((json.loads(line) for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), {"status": "OTHER_FAILURE", "message": completed.stderr[-2000:]})
            except subprocess.TimeoutExpired as exc:
                parsed = {"status": "COMPILE_TIMEOUT", "message": str(exc)}
            parsed.update({"configuration_id": candidate["configuration_id"], "configured_fraction": fraction, "family": candidate["family"], "configuration": candidate["configuration"], "dtype": candidate["dtype"]})
            rows.append(parsed)
            existing[key] = parsed
            output.write_text(json.dumps({"status": "OBSERVED", "phase": "thresholds", "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            execution = parsed.get("execution_statuses") or []
            print(json.dumps({"id": candidate["configuration_id"], "fraction": fraction, "status": execution[0] if execution else parsed.get("status", parsed.get("compile_status"))}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=75)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--families", help="comma-separated threshold families")
    parser.add_argument("--fractions", help="comma-separated allocator fractions")
    parser.add_argument("--dtypes", help="comma-separated threshold dtypes")
    args = parser.parse_args()
    rows = screen(args.screening)
    if not args.screen_only:
        families = set(args.families.split(",")) if args.families else None
        dtypes = set(args.dtypes.split(",")) if args.dtypes else None
        fractions = tuple(float(item) for item in args.fractions.split(",")) if args.fractions else None
        run_thresholds(rows, args.thresholds, args.timeout, families, dtypes, fractions)


if __name__ == "__main__":
    main()
