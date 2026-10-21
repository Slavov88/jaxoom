"""Finish unattempted recipes after audited, recovered uncertain timeouts.

This is evidence acquisition only. All unknowns still block qualification.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import oom_risk_gpu_continuation as c
from experiments import oom_risk_gpu_qualification as q


def recoverable(row):
    return q.trial_passes(row) or (row.get("likely_fit") is False and (
        (row.get("outcome") == "COMPILE_TIMEOUT" and row.get("phase") == "compile")
        or (row.get("outcome") == "EXECUTE_TIMEOUT" and row.get("phase") == "execute")))


def audit_chain(original, continuation):
    plan, prior = c.audit_previous(original)
    audit = json.loads((continuation / "independent_persisted_audit.json").read_text())
    if audit.get("verified") is not True:
        raise ValueError("continuation audit missing")
    for name, expected in audit["artifact_sha256"].items():
        path = continuation / name
        if path.parent.resolve() != continuation.resolve() or c.sha(path) != expected:
            raise ValueError("continuation artifact hash mismatch")
    contract = json.loads((continuation / "continuation_plan.json").read_text())
    if contract["source_sha256"] != c.sha(Path(c.__file__)):
        raise ValueError("continuation source changed")
    pending = c.pending_candidates(plan, prior)
    new = [json.loads(p.read_text()) for p in sorted(continuation.glob("trial_*.json"))]
    for index, row in enumerate(new):
        candidate = pending[index]
        if row["spec_hash"] != candidate["spec_hash"] or not recoverable(row):
            raise ValueError("unrecoverable continuation result")
        if row["likely_fit"] != (candidate["expected_probability"] <= .2):
            raise ValueError("decision mismatch")
        if q.trial_passes(row) and not q.validate_child_result(row, candidate, plan):
            raise ValueError("child provenance mismatch")
    combined = prior + new
    if c.summarize(combined) != audit["aggregate"]:
        raise ValueError("aggregate mismatch")
    return plan, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("parent must select CPU")
    plan, previous = audit_chain(args.original, args.continuation)
    pending = c.pending_candidates(plan, previous)
    args.output_dir.mkdir(exist_ok=False)
    contract = dict(protocol="restricted-v2-completion-1", source_sha256=c.sha(__file__),
                    original_plan_hash=plan["plan_hash"],
                    previous_audit_sha256=c.sha(args.continuation / "independent_persisted_audit.json"),
                    pending_spec_hashes=[p["spec_hash"] for p in pending],
                    timeout_seconds=180, max_new_timeouts=12, campaign_seconds=3600,
                    retry_unknown=False, public_api_enabled=False)
    q.write_new(args.output_dir / "completion_plan.json", contract)
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
            raise RuntimeError("fresh replay timeout") from exc
        (args.output_dir / "gpu_replay.stdout.txt").write_bytes(proc.stdout)
        (args.output_dir / "gpu_replay.stderr.txt").write_bytes(proc.stderr)
        evidence = json.loads(replay_path.read_text())
        if proc.returncode != 0 or evidence["provenance"]["backend"] != "gpu" or not evidence["fresh_tracing_and_replay_pass"]:
            raise RuntimeError("fresh GPU replay failed")
        for index, candidate in enumerate(pending):
            idle = q.wait_for_idle()
            # Check AFTER idle gating to reserve the complete child budget.
            if time.monotonic()-started > 3420:
                stop = "CAMPAIGN_TIME_BUDGET"
                break
            command = [sys.executable, "-m", "experiments.oom_risk_gpu_qualification", "--child", candidate["spec_hash"],
                       "--plan", str(args.original / "plan.json"), "--replay", str(replay_path)]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, env=q.child_env(), timeout=180, cwd=q.ROOT)
                stdout, stderr = proc.stdout, proc.stderr
                row = q.parse_child_result(stdout)
                if proc.returncode != 0:
                    row["outcome"] = "CHILD_PROCESS_FAILURE"
                elif not q.validate_child_result(row, candidate, plan):
                    row["outcome"] = "INVALID_CHILD_PROVENANCE_OR_REPLAY"
                row["returncode"] = proc.returncode
            except subprocess.TimeoutExpired as exc:
                stdout, stderr = exc.stdout or b"", exc.stderr or b""
                stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
                stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
                row = dict(q.timeout_result(stdout), returncode=None)
            row.update(spec_hash=candidate["spec_hash"], likely_fit=candidate["expected_probability"] <= .2,
                       expected_probability=candidate["expected_probability"], command=command, index=index)
            q.write_new(args.output_dir / f"trial_{index:03d}.json", row)
            (args.output_dir / f"trial_{index:03d}.stdout.txt").write_text(stdout)
            (args.output_dir / f"trial_{index:03d}.stderr.txt").write_text(stderr)
            results.append(row)
            print(index+1, candidate["candidate_id"], row["outcome"], flush=True)
            if not recoverable(row):
                stop = row["outcome"]
                break
            if row["outcome"].endswith("_TIMEOUT"):
                q.write_new(args.output_dir / f"recovery_{index:03d}.json", q.wait_for_idle())
                if sum(r["outcome"].endswith("_TIMEOUT") for r in results) >= 12:
                    stop = "TIMEOUT_BUDGET"
                    break
    except Exception as exc:
        stop = f"{type(exc).__name__}: {exc}"
    summary = c.summarize(previous+results)
    summary.update(new_trials=len(results), stop_reason=stop)
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
