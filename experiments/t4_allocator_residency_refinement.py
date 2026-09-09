"""Resumable T4 threshold refinement and targeted candidate search.

All target probes execute in fresh subprocesses. This module is experimental and
never runs from the public assessment API.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def classify(row: dict[str, Any]) -> str:
    statuses = row.get("execution_statuses") or []
    return statuses[0] if statuses else row.get("status") or row.get("compile_status") or "OTHER_FAILURE"


def capacity(row: dict[str, Any]) -> int | None:
    snapshots = row.get("snapshots") or []
    if snapshots:
        return snapshots[0].get("allocator_limit_bytes")
    return None


def bracket(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    fits = [row for row in rows if classify(row) == "FIT" and capacity(row) is not None]
    ooms = [row for row in rows if classify(row) in {"COMPILE_OOM", "EXECUTION_OOM"} and capacity(row) is not None]
    if not fits or not ooms:
        return None
    low = max(capacity(row) for row in ooms)
    high = min(capacity(row) for row in fits)
    return {"known_oom_capacity": low, "known_fit_capacity": high, "bracket_width_bytes": high - low, "usable": high > low}


def run_json(command: list[str], env: dict[str, str], timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"status": "EXECUTION_TIMEOUT", "message": str(exc)}
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"status": "OTHER_FAILURE", "returncode": completed.returncode, "stderr": completed.stderr[-4000:]}


def env_for(fraction: float) -> dict[str, str]:
    env = os.environ.copy()
    env["XLA_CLIENT_MEM_FRACTION"] = str(fraction)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    return env


def probe(candidate: dict[str, Any], fraction: float, timeout: int) -> dict[str, Any]:
    command = [
        sys.executable, str(Path(__file__).with_name("execution_oom_diagnosis.py")),
        "--trial", candidate["family"], "--config", json.dumps(candidate["configuration"], sort_keys=True),
        "--dtype", candidate["dtype"], "--repetitions", "1",
    ]
    row = run_json(command, env_for(fraction), timeout)
    row.update({"configuration_id": candidate["configuration_id"], "family": candidate["family"], "configuration": candidate["configuration"], "dtype": candidate["dtype"], "requested_fraction": fraction, "outcome": classify(row)})
    return row


def candidate_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("configuration_id", "family", "configuration", "dtype")}


def seed_candidates(thresholds: Path, raw: Path) -> list[dict[str, Any]]:
    threshold_data = json.loads(thresholds.read_text(encoding="utf-8"))
    raw_rows = json.loads(raw.read_text(encoding="utf-8"))["rows"]
    seeds = []
    for row in threshold_data.get("rows", threshold_data.get("useful_thresholds", [])):
        cid = row["configuration_id"]
        related = [item for item in raw_rows if item.get("configuration_id") == cid]
        useful = bracket(related)
        if useful:
            candidate = candidate_from_row(row)
            candidate["seed_rows"] = related
            candidate["seed_bracket"] = useful
            seeds.append(candidate)
    return seeds


def search(candidates: list[dict[str, Any]], fraction_map: Path, output: Path, timeout: int, max_probes: int, target_width: int, max_candidates: int | None) -> None:
    map_rows = json.loads(fraction_map.read_text(encoding="utf-8")).get("rows", [])
    usable = sorted((row for row in map_rows if row.get("bytes_limit") is not None), key=lambda row: row["bytes_limit"])
    if len(usable) < 3:
        raise RuntimeError("fraction map has fewer than three usable allocator limits")
    candidates = candidates[:max_candidates] if max_candidates else candidates
    checkpoint = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"status": "OBSERVED", "rows": []}
    rows = checkpoint.get("rows", [])
    seen = {(row.get("configuration_id"), row.get("requested_fraction")) for row in rows}
    probes = 0
    for candidate in candidates:
        candidate_rows = [row for row in rows if row.get("configuration_id") == candidate["configuration_id"]]
        seed_rows = candidate.get("seed_rows", [])
        seed = bracket(seed_rows) if seed_rows else None
        if seed:
            seed_oom = max((row for row in seed_rows if classify(row) in {"COMPILE_OOM", "EXECUTION_OOM"}), key=lambda row: capacity(row))
            seed_fit = min((row for row in seed_rows if classify(row) == "FIT"), key=lambda row: capacity(row))
            low_fraction = seed_oom["requested_fraction"]
            high_fraction = seed_fit["requested_fraction"]
        else:
            low_fraction = usable[0].get("requested_fraction")
            high_fraction = usable[-1].get("requested_fraction")
        if low_fraction is None or high_fraction is None:
            continue
        pending = [float(low_fraction), float(high_fraction)]
        while probes < max_probes:
            if candidate_rows:
                current = bracket(candidate_rows)
                if current and current["bracket_width_bytes"] <= target_width:
                    break
            fraction = pending.pop(0) if pending else (float(low_fraction) + float(high_fraction)) / 2
            key = (candidate["configuration_id"], fraction)
            if key in seen:
                # Find the next midpoint from measured outcomes instead of looping.
                measured = sorted((capacity(row), row["requested_fraction"], classify(row)) for row in candidate_rows if capacity(row) is not None)
                if len(measured) >= 2:
                    gaps = [(right[0] - left[0], left, right) for left, right in zip(measured, measured[1:]) if left[2] in {"COMPILE_OOM", "EXECUTION_OOM"} and right[2] == "FIT"]
                    if gaps:
                        fraction = (gaps[0][1][1] + gaps[0][2][1]) / 2
                key = (candidate["configuration_id"], fraction)
                if key in seen:
                    break
            row = probe(candidate, fraction, timeout)
            rows.append(row)
            seen.add((candidate["configuration_id"], fraction))
            probes += 1
            output.write_text(json.dumps({"status": "OBSERVED", "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            candidate_rows.append(row)
            current = bracket(candidate_rows)
            if current and current["bracket_width_bytes"] <= target_width:
                break
            measured = sorted((capacity(item), item["requested_fraction"], classify(item)) for item in candidate_rows if capacity(item) is not None)
            gaps = [(right, left) for left, right in zip(measured, measured[1:]) if left[2] in {"COMPILE_OOM", "EXECUTION_OOM"} and right[2] == "FIT"]
            if gaps:
                right, left = gaps[0]
                middle = (left[1] + right[1]) / 2
                pending.insert(0, middle)
    print(json.dumps({"probes_added": probes, "rows": len(rows)}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--fraction-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-probes", type=int, default=35)
    parser.add_argument("--target-width", type=int, default=64 * 1024**2)
    parser.add_argument("--max-candidates", type=int)
    args = parser.parse_args()
    search(seed_candidates(args.thresholds, args.raw), args.fraction_map, args.output, args.timeout, args.max_probes, args.target_width, args.max_candidates)


if __name__ == "__main__":
    main()
