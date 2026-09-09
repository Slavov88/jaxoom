"""Portable T4 allocator-residency campaign runner.

The parent process only orchestrates fresh subprocesses. Results are appended to
a checkpoint after every probe so Colab interruptions are resumable.
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


def run_json(command: list[str], env: dict[str, str], timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"status": "EXECUTION_TIMEOUT", "message": str(exc)}
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    return {"status": "OTHER_FAILURE", "returncode": completed.returncode, "stderr": completed.stderr[-4000:]}


def base_env(fraction: float) -> dict[str, str]:
    env = os.environ.copy()
    env["XLA_CLIENT_MEM_FRACTION"] = str(fraction)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    return env


def calibrate_fractions(output: Path, fractions: list[float], timeout: int) -> None:
    rows = []
    probe = (
        "import json, jax; d=jax.devices()[0]; "
        "s=d.memory_stats() if hasattr(d, 'memory_stats') else {}; "
        "print(json.dumps({'device':str(d),'device_kind':getattr(d,'device_kind',None),"
        "'backend':jax.default_backend(),'jax_version':jax.__version__,'bytes_limit':s.get('bytes_limit'),"
        "'bytes_in_use':s.get('bytes_in_use')}))"
    )
    for fraction in fractions:
        row = run_json([sys.executable, "-c", probe], base_env(fraction), timeout)
        row["requested_fraction"] = fraction
        rows.append(row)
        output.write_text(json.dumps({"status": "OBSERVED", "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def outcome(row: dict[str, Any]) -> str:
    statuses = row.get("execution_statuses") or []
    return statuses[0] if statuses else row.get("status") or row.get("compile_status") or "OTHER_FAILURE"


def probe(candidate: dict[str, Any], fraction: float, timeout: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("execution_oom_diagnosis.py")),
        "--trial", candidate["family"], "--config", json.dumps(candidate["configuration"], sort_keys=True),
        "--dtype", candidate["dtype"], "--repetitions", "1",
    ]
    row = run_json(command, base_env(fraction), timeout)
    row.update({"configuration_id": candidate["configuration_id"], "family": candidate["family"], "configuration": candidate["configuration"], "dtype": candidate["dtype"], "requested_fraction": fraction, "outcome": outcome(row)})
    return row


def run_thresholds(manifest: Path, fraction_map: Path, output: Path, timeout: int, max_probes: int) -> None:
    candidates = json.loads(manifest.read_text(encoding="utf-8"))["candidates"]
    mapping = json.loads(fraction_map.read_text(encoding="utf-8"))["rows"]
    usable = sorted((row for row in mapping if row.get("bytes_limit") is not None), key=lambda row: row["bytes_limit"])
    if len(usable) < 3:
        raise RuntimeError("fraction calibration did not produce three usable bytes_limit values")
    checkpoint = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"status": "OBSERVED", "rows": []}
    rows = checkpoint.get("rows", [])
    seen = {(row.get("configuration_id"), row.get("requested_fraction")) for row in rows}
    probes = 0
    for candidate in candidates:
        candidate_rows = [row for row in rows if row.get("configuration_id") == candidate["configuration_id"]]
        fractions = [row["requested_fraction"] for row in usable]
        # Start at both ends, then bisect the measured-capacity interval.
        indices = [0, len(usable) - 1]
        while indices and probes < max_probes:
            index = indices.pop(0)
            fraction = fractions[index]
            if (candidate["configuration_id"], fraction) in seen:
                continue
            row = probe(candidate, fraction, timeout)
            rows.append(row)
            seen.add((candidate["configuration_id"], fraction))
            probes += 1
            output.write_text(json.dumps({"status": "OBSERVED", "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            candidate_rows.append(row)
            outcomes = {item["outcome"]: item for item in candidate_rows}
            fit = [item for item in candidate_rows if item["outcome"] == "FIT" and item.get("snapshots")]
            oom = [item for item in candidate_rows if item["outcome"] in {"COMPILE_OOM", "EXECUTION_OOM"} and item.get("snapshots")]
            if fit and oom:
                fit_cap = min(item["snapshots"][0].get("allocator_limit_bytes", 0) for item in fit)
                oom_cap = max(item["snapshots"][0].get("allocator_limit_bytes", 0) for item in oom)
                if fit_cap - oom_cap <= 64 * 1024**2:
                    break
            if len(candidate_rows) >= 2:
                measured = sorted((item.get("snapshots", [{}])[0].get("allocator_limit_bytes"), item["requested_fraction"], item["outcome"]) for item in candidate_rows if item.get("snapshots"))
                for left, right in zip(measured, measured[1:]):
                    if left[2] in {"COMPILE_OOM", "EXECUTION_OOM"} and right[2] == "FIT":
                        middle = [i for i, item in enumerate(usable) if left[0] < item["bytes_limit"] < right[0]]
                        if middle:
                            indices.insert(0, middle[len(middle) // 2])
                            break
    print(json.dumps({"probes_added": probes, "total_rows": len(rows)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction-map", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fractions", default="0.10,0.13,0.17,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-probes", type=int, default=80)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if args.calibrate:
        calibrate_fractions(args.output, [float(item) for item in args.fractions.split(",")], args.timeout)
    else:
        run_thresholds(args.manifest, args.fraction_map, args.output, args.timeout, args.max_probes)


if __name__ == "__main__":
    main()
