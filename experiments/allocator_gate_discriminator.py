"""Adversarial, frozen discriminator study for the experimental allocator gate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxoom

BUDGET = 3 * 1024**3
DTYPES = ("float16", "float32")

def attention(q, k, v):
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def args_for(case, concrete=False):
    dtype = getattr(jnp, case["dtype"])
    shape = (case["B"], case["H"], case["S"], case["D"])
    return tuple(jnp.ones(shape, dtype) if concrete else jax.ShapeDtypeStruct(shape, dtype) for _ in range(3))


def static_row(case):
    report = jaxoom.estimate(attention, *args_for(case))
    interval = jaxoom.calibrate(report, backend="gpu", jax_version=jax.__version__)
    live = sorted((b.nbytes for b in report.peak.live_buffers), reverse=True)
    largest = live[0] if live else 0
    top_two = sum(live[:2])
    peak = report.peak.live_bytes
    upper = interval.upper_bytes
    return {**case, "budget_bytes": BUDGET, "structural_peak_bytes": report.estimated_peak_bytes, "calibrated_upper_bytes": upper, "largest_buffer_bytes": largest, "top_two_peak_live_bytes": top_two, "peak_live_bytes": peak, "aggregate_score": upper / BUDGET, "top_two_score": (upper + top_two) / BUDGET, "peak_live_score": (upper + peak) / BUDGET, "largest_score": (upper + largest) / BUDGET, "aggregate_pass": upper <= BUDGET, "top_two_pass": upper + top_two <= BUDGET, "peak_live_pass": upper + peak <= BUDGET, "largest_pass": upper + largest <= BUDGET, "multiplier_predictions": {"1.10": upper * 1.10 <= BUDGET, "1.25": upper * 1.25 <= BUDGET, "1.50": upper * 1.50 <= BUDGET}}


def candidate_pool():
    rows = []
    for B in (1, 2, 4):
        for H in (4, 8, 12, 16):
            for D in (32, 64, 96, 128):
                for S in (1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5632, 6144):
                    for dtype in DTYPES:
                        case = {"configuration_id": f"DISC-B{B}-H{H}-S{S}-D{D}-{dtype}", "B": B, "H": H, "S": S, "D": D, "dtype": dtype, "family": "attention"}
                        rows.append(static_row(case))
    return rows


def diverse_take(pool, count, used):
    selected = []
    remaining = list(pool)
    while remaining and len(selected) < count:
        remaining.sort(key=lambda r: (len({(r["B"], r["H"], r["D"], r["dtype"])} & used), abs(r["top_two_score"] - 1.05), r["configuration_id"]))
        best = remaining.pop(0)
        selected.append(best)
        used.add((best["B"], best["H"], best["D"], best["dtype"]))
    return selected


def select(rows):
    for r in rows: r["selection_categories"] = []
    A = [r for r in rows if r["aggregate_pass"] and not r["top_two_pass"]]
    A.sort(key=lambda r: (abs(r["aggregate_score"] - .85), abs(r["top_two_score"] - 1.15)))
    selected = diverse_take(A, min(14, len(A)), set())
    for r in selected: r["selection_categories"].append("AGG_PASS_TOP2_REJECT")
    near = [r for r in rows if r["aggregate_pass"] and r["top_two_pass"] and .85 <= r["top_two_score"] <= 1.02 and r not in selected]
    for r in diverse_take(sorted(near, key=lambda x: abs(x["top_two_score"] - .95)), min(6, len(near)), set()):
        r["selection_categories"].append("AGG_PASS_TOP2_NEAR_PASS")
        selected.append(r)
    tp = [r for r in rows if r["top_two_pass"] != r["peak_live_pass"] and r not in selected]
    for r in diverse_take(sorted(tp, key=lambda x: abs(x["top_two_score"] - 1)), min(5, len(tp)), set()):
        r["selection_categories"].append("TOP2_VS_PEAKLIVE_DISAGREE")
        selected.append(r)
    ld = [r for r in rows if r["top_two_pass"] != r["largest_pass"] and r not in selected]
    for r in diverse_take(sorted(ld, key=lambda x: abs(x["top_two_score"] - 1)), min(3, len(ld)), set()):
        r["selection_categories"].append("TOP2_VS_LARGEST_DISAGREE")
        selected.append(r)
    # Static matched groups are labels, not outcome-based tuning.
    for i, r in enumerate(selected):
        if .78 <= r["aggregate_score"] <= .90: r["selection_categories"].append("MATCHED_AGGREGATE_GROUP")
        if any(abs(r["top_two_peak_live_bytes"] - q["top_two_peak_live_bytes"]) <= 32 * 1024**2 and (r["B"], r["H"], r["S"], r["D"], r["dtype"]) != (q["B"], q["H"], q["S"], q["D"], q["dtype"]) for q in selected): r["selection_categories"].append("MATCHED_TOP2_GROUP")
    return selected


def child(case):
    started = time.perf_counter()
    try:
        args = args_for(case, True)
        compiled = jax.jit(attention).lower(*args).compile()
        compile_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        out = compiled(*args)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
        print(json.dumps({"outcome": "FIT", "compile_latency_ms": compile_ms, "execution_latency_ms": (time.perf_counter() - started) * 1000}), flush=True)
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        lower = text.lower()
        phase = "COMPILE" if "compiled" not in locals() else "EXECUTION"
        oom = "out of memory" in lower or "resource_exhausted" in lower
        print(json.dumps({"outcome": f"{phase}_OOM" if oom else "OTHER_FAILURE", "error": text}), flush=True)


def probe(row, timeout=60):
    env = os.environ.copy(); env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    try:
        p = subprocess.run([sys.executable, str(Path(__file__)), "--child-json", json.dumps({k: row[k] for k in ("B", "H", "S", "D", "dtype")})], capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"outcome": "EXECUTION_TIMEOUT"}
    for line in reversed(p.stdout.splitlines()):
        try: return json.loads(line)
        except json.JSONDecodeError: pass
    return {"outcome": "OTHER_FAILURE", "stderr": p.stderr[-2000:]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--plan-output", type=Path); ap.add_argument("--results-output", type=Path); ap.add_argument("--run", action="store_true"); ap.add_argument("--child-json"); ap.add_argument("--timeout", type=int, default=60)
    a = ap.parse_args()
    if a.child_json:
        child(json.loads(a.child_json)); return
    started = time.perf_counter(); all_rows = candidate_pool(); screening_s = time.perf_counter() - started
    selected = select(all_rows)
    plan = {"status": "FROZEN_PLAN", "environment": {"device": str(jax.devices()[0]), "backend": jax.default_backend(), "jax_version": jax.__version__}, "budget_bytes": BUDGET, "static_candidates_screened": len(all_rows), "static_screening_seconds": screening_s, "selected_rows": selected, "selection_counts": dict(Counter(c for r in selected for c in r["selection_categories"]))}
    if a.plan_output: a.plan_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print("screened", len(all_rows), "selected", len(selected), "seconds", round(screening_s, 2), "counts", plan["selection_counts"])
    if a.run:
        results=[]
        for row in selected:
            repeats = 3 if .85 <= row["top_two_score"] <= 1.15 else 2
            outcomes=[probe(row, a.timeout) for _ in range(repeats)]
            stable=outcomes[0]["outcome"] if all(x.get("outcome")==outcomes[0].get("outcome") for x in outcomes) else "UNSTABLE"
            results.append({"configuration_id": row["configuration_id"], "prediction_snapshot": {"aggregate": row["aggregate_pass"], "largest": row["largest_pass"], "top_two": row["top_two_pass"], "peak_live": row["peak_live_pass"], "multipliers": row["multiplier_predictions"]}, "outcomes": outcomes, "stable_outcome": stable})
        payload={"status":"COMPUTATIONALLY_VERIFIED", "plan":plan, "results":results}
        if a.results_output: a.results_output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")

if __name__ == "__main__": main()
