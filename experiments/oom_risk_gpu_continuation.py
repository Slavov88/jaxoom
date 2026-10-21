"""Append-only completion of unrun recipes; unknown outcomes never qualify."""
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import oom_risk_gpu_qualification as q
from experiments import oom_risk_workload_spec as adapter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_previous(directory):
    directory = Path(directory)
    audit = json.loads((directory / "independent_audit.json").read_text())
    for name, expected in audit["artifact_sha256"].items():
        path = directory / name
        if path.parent.resolve() != directory.resolve() or sha(path) != expected:
            raise ValueError("previous evidence hash mismatch: " + name)
    plan = q.load_plan(directory / "plan.json")
    candidates = {c["spec_hash"]: c for c in plan["candidates"]}
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob("trial_*.json"))]
    if len({r["spec_hash"] for r in rows}) != len(rows):
        raise ValueError("duplicate prior recipes")
    for row in rows:
        c = candidates[row["spec_hash"]]
        if row["likely_fit"] != (c["expected_probability"] <= adapter.FIT_THRESHOLD):
            raise ValueError("prior decision mismatch")
        if q.trial_passes(row):
            if not q.validate_child_result(row, c, plan):
                raise ValueError("invalid prior child provenance")
        elif row["outcome"] != "EXECUTE_TIMEOUT" or row["likely_fit"] is not False:
            raise ValueError("unrecoverable prior stop")
    if len(rows) != 31 or sum(r["outcome"] == "EXECUTE_TIMEOUT" for r in rows) != 1:
        raise ValueError("unexpected prior run")
    return plan, rows


def pending_candidates(plan, previous):
    attempted = {r["spec_hash"] for r in previous}
    return [c for c in plan["candidates"] if c["spec_hash"] not in attempted]


def can_continue(row):
    return q.trial_passes(row) or (row.get("outcome") == "EXECUTE_TIMEOUT"
                                  and row.get("likely_fit") is False)


def summarize(rows, total=200):
    unique = len({r["spec_hash"] for r in rows}) == len(rows)
    complete = unique and len(rows) == total
    passed = complete and all(q.trial_passes(r) for r in rows)
    return dict(status="GPU_QUALIFICATION_PASS" if passed else "GPU_QUALIFICATION_INCOMPLETE",
                all_checks_pass=passed, public_api_enabled=False,
                attempted=len(rows), not_run=total-len(rows), coverage_complete=complete,
                outcomes=dict(Counter(r["outcome"] for r in rows)),
                unresolved=sum(not q.trial_passes(r) for r in rows),
                likely_fit_count=sum(r["likely_fit"] for r in rows),
                likely_fit_failures=sum(r["likely_fit"] and r["outcome"] != "FIT" for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("parent must select CPU")
    plan, previous = audit_previous(args.previous)
    pending = pending_candidates(plan, previous)
    args.output_dir.mkdir(exist_ok=False)
    contract = dict(protocol="restricted-v2-continuation-1", previous_plan_hash=plan["plan_hash"],
                    previous_audit_sha256=sha(args.previous / "independent_audit.json"),
                    source_sha256=sha(__file__), pending_spec_hashes=[c["spec_hash"] for c in pending],
                    timeout_seconds=180, max_campaign_seconds=3600, max_new_timeouts=12,
                    retry_previous_unknown=False, public_api_enabled=False)
    q.write_new(args.output_dir / "continuation_plan.json", contract)
    replay_path = args.output_dir / "gpu_replay.json"
    results = []
    stop = None
    started = time.monotonic()
    try:
        q.write_new(args.output_dir / "initial_idle_snapshot.json", q.wait_for_idle())
        command = [sys.executable, "-m", "experiments.oom_risk_workload_replay", "--output", str(replay_path)]
        try:
            proc = subprocess.run(command, capture_output=True, env=q.child_env(), timeout=90, cwd=q.ROOT)
        except subprocess.TimeoutExpired as exc:
            (args.output_dir / "gpu_replay.stdout.txt").write_bytes(exc.stdout or b"")
            (args.output_dir / "gpu_replay.stderr.txt").write_bytes(exc.stderr or b"")
            raise RuntimeError("fresh GPU replay timeout") from exc
        (args.output_dir / "gpu_replay.stdout.txt").write_bytes(proc.stdout)
        (args.output_dir / "gpu_replay.stderr.txt").write_bytes(proc.stderr)
        evidence = json.loads(replay_path.read_text())
        if proc.returncode != 0 or evidence["provenance"]["backend"] != "gpu" or not evidence["fresh_tracing_and_replay_pass"]:
            raise RuntimeError("fresh GPU replay failed")
        for index, c in enumerate(pending):
            if time.monotonic()-started > 3600-180:
                stop = "CAMPAIGN_TIME_BUDGET"
                break
            q.wait_for_idle()
            command = [sys.executable, "-m", "experiments.oom_risk_gpu_qualification", "--child", c["spec_hash"],
                       "--plan", str(args.previous / "plan.json"), "--replay", str(replay_path)]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, env=q.child_env(), timeout=180, cwd=q.ROOT)
                stdout, stderr = proc.stdout, proc.stderr
                row = q.parse_child_result(stdout)
                if proc.returncode != 0:
                    row["outcome"] = "CHILD_PROCESS_FAILURE"
                elif not q.validate_child_result(row, c, plan):
                    row["outcome"] = "INVALID_CHILD_PROVENANCE_OR_REPLAY"
                row["returncode"] = proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or b""
                stderr = exc.stderr or b""
                stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
                stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
                row = dict(q.timeout_result(stdout), returncode=None)
            row.update(spec_hash=c["spec_hash"], likely_fit=c["expected_probability"] <= adapter.FIT_THRESHOLD,
                       expected_probability=c["expected_probability"], command=command, index=index)
            q.write_new(args.output_dir / f"trial_{index:03d}.json", row)
            (args.output_dir / f"trial_{index:03d}.stdout.txt").write_text(stdout)
            (args.output_dir / f"trial_{index:03d}.stderr.txt").write_text(stderr)
            results.append(row)
            print(index+1, c["candidate_id"], row["outcome"], flush=True)
            if not can_continue(row):
                stop = row["outcome"]
                break
            # subprocess.run has killed and waited for a timed-out child. A new
            # trial is forbidden until telemetry confirms complete recovery.
            if row["outcome"] == "EXECUTE_TIMEOUT":
                q.write_new(args.output_dir / f"recovery_{index:03d}.json", q.wait_for_idle())
                if sum(r["outcome"] == "EXECUTE_TIMEOUT" for r in results) >= 12:
                    stop = "TIMEOUT_BUDGET"
                    break
    except Exception as exc:
        stop = f"{type(exc).__name__}: {exc}"
    summary = summarize(previous+results)
    summary.update(new_trials=len(results), stop_reason=stop, previous_plan_hash=plan["plan_hash"])
    try:
        q.write_new(args.output_dir / "final_idle_snapshot.json", q.wait_for_idle())
    except Exception as exc:
        summary.update(all_checks_pass=False, recovery_error=str(exc))
    q.write_new(args.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not summary["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
