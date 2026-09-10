"""Run isolated ATT-2/ATT-1 probes without initializing JAX in the parent."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def hold(bytes_to_hold: int, ready: Path) -> None:
    import jax.numpy as jnp

    value = jnp.zeros((max(1, bytes_to_hold // 4),), dtype=jnp.float32)
    value.block_until_ready()
    ready.write_text("ready", encoding="utf-8")
    sys.stdin.readline()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=420)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hold-bytes", type=int, default=0)
    parser.add_argument("--holder", nargs=2, metavar=("BYTES", "READY"))
    args = parser.parse_args()
    if args.holder:
        hold(int(args.holder[0]), Path(args.holder[1]))
    if args.output is None:
        parser.error("--output is required in parent mode")
    investigate = Path(__file__).with_name("investigate.py")
    rows = []
    holder = None
    temporary = None
    if args.hold_bytes:
        temporary = tempfile.TemporaryDirectory(prefix="att2-holder-")
        ready = Path(temporary.name) / "ready"
        holder = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--holder", str(args.hold_bytes), str(ready)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
        )
        deadline = time.monotonic() + 90
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not ready.exists():
            holder.kill()
            raise RuntimeError("holder did not become ready")
    for rep in range(1, args.repetitions + 1):
        started = time.perf_counter()
        command = [
            sys.executable,
            str(investigate),
            "--probe",
            "--workload-id",
            args.workload_id,
            "--rep",
            str(rep),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=os.environ.copy(),
            )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if lines:
                try:
                    row = json.loads(lines[-1])
                except json.JSONDecodeError:
                    row = {"outcome": "OTHER_FAILURE", "failure_phase": "UNKNOWN", "error": lines[-1]}
            else:
                text = (completed.stderr or completed.stdout)[-8000:]
                row = {
                    "outcome": "OOM" if "out of memory" in text.lower() or "resource_exhausted" in text.lower() else "OTHER_FAILURE",
                    "failure_phase": "UNKNOWN",
                    "error": text,
                }
            row["parent_returncode"] = completed.returncode
        except subprocess.TimeoutExpired as exc:
            row = {"outcome": "TIMEOUT", "failure_phase": "UNKNOWN", "error": str(exc)}
        row["rep"] = rep
        row["parent_wall_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(rep, row.get("outcome"), row.get("failure_phase"), flush=True)
    if holder is not None:
        if holder.stdin:
            holder.stdin.write("release\n")
            holder.stdin.flush()
        try:
            holder.wait(timeout=60)
        except subprocess.TimeoutExpired:
            holder.kill()
        if temporary is not None:
            temporary.cleanup()
    payload = {
        "status": "OBSERVED",
        "workload_id": args.workload_id,
        "hold_bytes": args.hold_bytes,
        "repetitions": args.repetitions,
        "allocator_environment": {key: os.environ.get(key) for key in (
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "XLA_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "TF_GPU_ALLOCATOR",
        )},
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
