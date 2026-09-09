"""Family-directed T4 workload-scale search for allocator thresholds.

Stage A searches workload scale at one low allocator capacity. Stage B is
performed by the existing fresh-process allocator threshold runner using the
selected fixed configurations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def configuration_id(family: str, config: dict[str, Any], dtype: str) -> str:
    payload = json.dumps({"family": family, "config": config, "dtype": dtype}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def outcome(row: dict[str, Any]) -> str:
    return (row.get("execution_statuses") or [row.get("status") or row.get("compile_status") or "OTHER_FAILURE"])[0]


def run_probe(candidate: dict[str, Any], fraction: float, timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["XLA_CLIENT_MEM_FRACTION"] = str(fraction)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    command = [sys.executable, str(Path(__file__).with_name("execution_oom_diagnosis.py")), "--trial", candidate["family"], "--config", json.dumps(candidate["configuration"], sort_keys=True), "--dtype", candidate["dtype"], "--repetitions", "1"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {**candidate, "requested_fraction": fraction, "outcome": "EXECUTION_TIMEOUT", "message": str(exc)}
    for line in reversed(completed.stdout.splitlines()):
        try:
            row = json.loads(line)
            row.update({"configuration_id": candidate["configuration_id"], "requested_fraction": fraction, "scale_stage": True, "outcome": outcome(row)})
            return row
        except json.JSONDecodeError:
            continue
    return {**candidate, "requested_fraction": fraction, "outcome": "OTHER_FAILURE", "stderr": completed.stderr[-4000:]}


def default_candidates() -> list[dict[str, Any]]:
    candidates = []
    grids = {
        "mlp": [
            ({"batch": b, "width": 4096, "depth": 4}, d) for d in ("float32", "float16") for b in (256, 512, 1024, 2048, 4096)
        ] + [
            ({"batch": b, "width": 6144, "depth": 4}, d) for d in ("float32", "float16") for b in (256, 512, 1024)
        ],
        "training": [
            ({"batch": b, "width": 4096}, d) for d in ("float32", "float16") for b in (256, 512, 1024, 2048, 4096)
        ] + [
            ({"batch": b, "width": 6144}, d) for d in ("float32", "float16") for b in (256, 512, 1024)
        ],
        "autodiff": [
            ({"batch": b, "width": 4096, "layers": 4}, d) for d in ("float32", "float16") for b in (64, 128, 256, 512, 1024)
        ] + [
            ({"batch": b, "width": 6144, "layers": 4}, d) for d in ("float32", "float16") for b in (64, 128, 256)
        ],
        "transformer": [
            ({"sequence": s, "width": 2048, "heads": 8}, d) for d in ("float32", "float16") for s in (1024, 1536, 2048, 3072)
        ],
        "convolution": [
            ({"batch": b, "height": s, "width": s, "channels": c, "out_channels": 2 * c}, d)
            for d in ("float32", "float16") for c, s in ((64, 512), (64, 768), (96, 512)) for b in (1, 2, 4)
        ],
    }
    for family, values in grids.items():
        for config, dtype in values:
            candidates.append({"configuration_id": configuration_id(family, config, dtype), "family": family, "configuration": config, "dtype": dtype})
    return candidates


def run_scale_search(candidates: list[dict[str, Any]], output: Path, fraction: float, timeout: int, max_probes: int) -> None:
    checkpoint = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"status": "OBSERVED", "rows": []}
    rows = checkpoint.get("rows", [])
    seen = {row.get("configuration_id") for row in rows}
    added = 0
    for candidate in candidates:
        if added >= max_probes or candidate["configuration_id"] in seen:
            continue
        row = run_probe(candidate, fraction, timeout)
        rows.append(row); seen.add(candidate["configuration_id"]); added += 1
        output.write_text(json.dumps({"status": "OBSERVED", "stage": "scale_search", "allocator_fraction": fraction, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"family": candidate["family"], "configuration": candidate["configuration"], "dtype": candidate["dtype"], "outcome": row["outcome"]}, sort_keys=True), flush=True)


def select_boundary_candidates(scale_rows: list[dict[str, Any]], output: Path, max_per_family: int) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in scale_rows:
        groups.setdefault(row.get("family", "unknown"), []).append(row)
    selected = []
    for family, rows in groups.items():
        useful = [row for row in rows if row.get("outcome") in {"COMPILE_OOM", "EXECUTION_OOM"}]
        useful.sort(key=lambda row: (row.get("dtype") != "float16", json.dumps(row.get("configuration", {}), sort_keys=True)))
        selected.extend(useful[:max_per_family])
    output.write_text(json.dumps({"status": "OBSERVED", "selected": selected, "selected_family_counts": {family: sum(row.get("family") == family for row in selected) for family in groups}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.06)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-probes", type=int, default=100)
    parser.add_argument("--max-per-family", type=int, default=3)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    if args.select_only:
        rows = json.loads(args.scale_output.read_text(encoding="utf-8")).get("rows", [])
    else:
        run_scale_search(default_candidates(), args.scale_output, args.fraction, args.timeout, args.max_probes)
        rows = json.loads(args.scale_output.read_text(encoding="utf-8")).get("rows", [])
    select_boundary_candidates(rows, args.selection_output, args.max_per_family)


if __name__ == "__main__":
    main()
