"""Fresh-process probes for allocator-capacity requirements.

This harness varies the JAX allocator fraction and delegates phase
instrumentation to ``execution_oom_diagnosis.py``. It is diagnostic only and
never changes the public predictor.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CONFIGS = [
    ("attention", {"sequence": 4096, "heads": 8, "head_dim": 64}, "float32", (0.75, 0.76, 0.77)),
    ("attention", {"sequence": 3072, "heads": 8, "head_dim": 64}, "float32", (0.50, 0.60, 0.70)),
    ("attention", {"sequence": 2048, "heads": 8, "head_dim": 64}, "float32", (0.35, 0.45, 0.55)),
    ("attention", {"sequence": 2048, "heads": 8, "head_dim": 64}, "float16", (0.35, 0.45, 0.55)),
    ("mlp", {"batch": 256, "width": 8192, "depth": 4}, "float32", (0.35, 0.45, 0.55)),
    ("training", {"batch": 256, "width": 8192}, "float32", (0.35, 0.45, 0.55)),
    ("transformer", {"sequence": 2048, "width": 2048, "heads": 8}, "float32", (0.35, 0.45, 0.55)),
    ("mlp", {"batch": 256, "width": 12288, "depth": 4}, "float32", (0.45, 0.55, 0.65, 0.75)),
    ("training", {"batch": 256, "width": 12288}, "float32", (0.45, 0.55, 0.65, 0.75)),
    ("transformer", {"sequence": 4096, "width": 2048, "heads": 8}, "float32", (0.55, 0.65, 0.75, 0.85)),
]


def run_probe(family: str, config: dict[str, Any], dtype: str, fraction: float, timeout: int) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(root / "execution_oom_diagnosis.py"),
        "--trial",
        family,
        "--config",
        json.dumps(config, sort_keys=True),
        "--dtype",
        dtype,
        "--repetitions",
        "1",
    ]
    env = os.environ.copy()
    env["XLA_CLIENT_MEM_FRACTION"] = str(fraction)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        return {"family": family, "configuration": config, "dtype": dtype, "configured_fraction": fraction, "status": "OTHER_FAILURE", "failure_stage": "timeout", "message": str(exc)}
    for line in reversed(completed.stdout.splitlines()):
        try:
            row = json.loads(line)
            row["configured_fraction"] = fraction
            row["subprocess_returncode"] = completed.returncode
            return row
        except json.JSONDecodeError:
            continue
    return {
        "family": family,
        "configuration": config,
        "dtype": dtype,
        "configured_fraction": fraction,
        "status": "OTHER_FAILURE",
        "message": (completed.stderr or completed.stdout)[-2000:],
        "subprocess_returncode": completed.returncode,
    }


def threshold_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row.get("family", "unknown"), json.dumps(row.get("configuration", {}), sort_keys=True), row.get("dtype", "unknown"))
        groups[key].append(row)
    output = []
    for (family, config, dtype), values in sorted(groups.items()):
        values.sort(key=lambda item: item.get("configured_fraction", 0))
        fit_limits = [item["snapshots"][0].get("allocator_limit_bytes") for item in values if item.get("execution_statuses", [None])[0] == "FIT"]
        oom_limits = [item["snapshots"][0].get("allocator_limit_bytes") for item in values if item.get("execution_statuses", [None])[0] in {"EXECUTION_OOM", "COMPILE_OOM"}]
        output.append({
            "family": family,
            "configuration": json.loads(config),
            "dtype": dtype,
            "tested_limits": [item["snapshots"][0].get("allocator_limit_bytes") for item in values],
            "outcomes": [item.get("execution_statuses", [item.get("compile_status")])[0] for item in values],
            "required_capacity_lower_bytes": max(oom_limits) if oom_limits else None,
            "required_capacity_upper_bytes": min(fit_limits) if fit_limits else None,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    rows = []
    for family, config, dtype, fractions in CONFIGS:
        for fraction in fractions:
            row = run_probe(family, config, dtype, fraction, args.timeout)
            rows.append(row)
            print(json.dumps({"family": family, "configuration": config, "dtype": dtype, "fraction": fraction, "limit": (row.get("snapshots") or [{}])[0].get("allocator_limit_bytes"), "compile": row.get("compile_status"), "execution": row.get("execution_statuses")}, sort_keys=True), flush=True)
    payload = {"status": "OBSERVED", "allocator": "BFC-style", "preallocate": False, "trials": rows, "thresholds": threshold_summary(rows)}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
