"""Fresh-trace replay gate for the restricted V2 adapter; no public API changes.

Run from the repository with PYTHONPATH=src:
python -m experiments.oom_risk_workload_replay --output NEW_PATH.json
Existing evidence is never overwritten. A failed replay exits nonzero.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

from experiments import oom_risk_workload_spec as adapter

ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    ("oom_risk_heldout_predictions_2026-09-13.json", 90),
    ("oom_risk_heldout2_predictions_2026-09-13.json", 110),
    ("oom_risk_heldout2_conv_stress_predictions_2026-09-13.json", 24),
)
PANEL_SHA256 = {
    "oom_risk_heldout_predictions_2026-09-13.json": "116ff4a1a75831e4867fc7c7a7f203459c9ea0f37174d0854ec5b263f2de0e63",
    "oom_risk_heldout2_predictions_2026-09-13.json": "f265eb9697b341cfc99cb7bb33504ae2e4d0b3aa68e6688f4c7f7c9620f3b0bb",
    "oom_risk_heldout2_conv_stress_predictions_2026-09-13.json": "2ed9d119cd91b290867251d4017d9805168ab6e5486b6664f41e17e4d494ce09",
}
TOLERANCE = 1e-12


@contextmanager
def static_only_guard():
    """Reject public compilation paths and any concrete call to the target."""
    import jax
    import jaxoom

    original = adapter.build_workload
    observations = {"target_calls_with_tracers": 0, "concrete_target_calls": 0,
                    "blocked_compile_calls": 0}

    def forbidden(*args, **kwargs):
        observations["blocked_compile_calls"] += 1
        raise RuntimeError("target compilation is forbidden in the replay")

    def guarded_build(spec):
        fn, abstract = original(spec)

        def traced_only(*args):
            if not all(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(args)):
                observations["concrete_target_calls"] += 1
                raise RuntimeError("concrete target execution is forbidden in the replay")
            observations["target_calls_with_tracers"] += 1
            return fn(*args)

        return traced_only, abstract

    with patch.object(adapter, "build_workload", guarded_build), \
         patch.object(jax, "jit", forbidden), \
         patch.object(jaxoom, "compile_analyze", forbidden), \
         patch.object(jaxoom, "analyze_donation", forbidden):
        yield observations


def load_prediction_rows():
    rows = []
    sources = {}
    for name, expected_count in PANELS:
        path = ROOT / "experiments" / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != PANEL_SHA256[name]:
            raise ValueError(f"frozen prediction artifact bytes have changed: {name}")
        payload = json.loads(raw)
        if payload.get("v2_model_hash") != adapter.MODEL_HASH:
            raise ValueError(f"unexpected V2 model provenance: {name}")
        expected_hash = payload.pop("prediction_hash")
        if adapter.digest(payload) != expected_hash:
            raise ValueError(f"prediction payload hash mismatch: {name}")
        if payload["status"] != "FROZEN_BEFORE_RUNTIME" or not payload["outcomes_excluded"]:
            raise ValueError(f"prediction panel is not outcome-free: {name}")
        if len(payload["predictions"]) != expected_count:
            raise ValueError(f"incomplete prediction panel: {name}")
        sources[name] = hashlib.sha256(raw).hexdigest()
        rows.extend(dict(r, replay_panel=name) for r in payload["predictions"])
    # The supplemental panel reuses H2 IDs; identity includes the source panel.
    if len({(r["replay_panel"], r["candidate_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate replay candidate IDs")
    return rows, sources


def run_replay():
    import jax
    import jaxlib
    import jaxoom
    import numpy

    model = adapter.load_frozen_model()
    contract = adapter.pipeline_contract()
    rows, sources = load_prediction_rows()
    mismatches = Counter({n: 0 for n in adapter.FEATURE_NAMES})
    score_mismatches = decisions = completed = 0
    max_feature_delta = max_score_delta = 0.0
    details = []
    errors = []
    with static_only_guard() as observations:
        for row in rows:
            try:
                spec = adapter.WorkloadSpec(row["family"], row["configuration"], row["dtype"])
                fresh = adapter.extract_features(spec)
                expected = {}
                for name in adapter.FEATURE_NAMES:
                    # Only this single, historically absent field may default.
                    value = row["static_features"].get(name, 0.0) if name == "largest_over_budget" else row["static_features"][name]
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"invalid saved feature: {name}")
                    expected[name] = float(value)
                deltas = {n: abs(fresh[n] - expected[n]) for n in adapter.FEATURE_NAMES}
                for n, delta in deltas.items():
                    mismatches[n] += delta > TOLERANCE
                max_feature_delta = max(max_feature_delta, *deltas.values())
                score = adapter._score_features(fresh, model)
                saved_score = row["v2_probability"]
                if not isinstance(saved_score, (int, float)) or not math.isfinite(saved_score) or not 0 <= saved_score <= 1:
                    raise ValueError("invalid saved probability")
                delta = abs(score - saved_score)
                score_mismatches += delta > TOLERANCE
                decisions += (score <= adapter.FIT_THRESHOLD) != (saved_score <= adapter.FIT_THRESHOLD)
                max_score_delta = max(max_score_delta, delta)
                completed += 1
                details.append({"panel": row["replay_panel"], "candidate_id": row["candidate_id"], "spec_hash": spec.spec_hash,
                                "features": fresh, "saved_probability": saved_score,
                                "fresh_probability": score, "absolute_probability_delta": delta})
            except Exception as exc:
                errors.append({"panel": row["replay_panel"], "candidate_id": row["candidate_id"],
                               "error": f"{type(exc).__name__}: {exc}"})
    purity_pass = (observations["target_calls_with_tracers"] == len(rows)
                   and observations["concrete_target_calls"] == 0
                   and observations["blocked_compile_calls"] == 0)
    public_absent = (not hasattr(jaxoom, "assess_oom_risk")
                     and not hasattr(jaxoom, "WorkloadSpec") and not adapter.PUBLIC_API_ENABLED)
    passed = (completed == len(rows) == 224 and not errors and not any(mismatches.values())
              and score_mismatches == 0 and decisions == 0 and purity_pass)
    source_files = sorted((ROOT / "src" / "jaxoom").rglob("*.py")) + [
        Path(adapter.__file__), Path(__file__), ROOT / "experiments" / "oom_risk_expansion_v2.py",
    ]
    return {
        "status": "COMPUTATIONALLY_VERIFIED" if passed else "FAILED",
        "fresh_tracing_and_replay_pass": passed, "public_api_enabled": False,
        "public_api_absent": public_absent,
        "runtime_gpu_validation": "NOT_RUN", "arbitrary_callable_support": "DISABLED",
        "reference_is_runtime_quota": False,
        "rows": len(rows), "completed": completed, "feature_mismatches": sum(mismatches.values()),
        "feature_mismatches_by_name": dict(mismatches), "score_mismatches": score_mismatches,
        "decision_mismatches": decisions, "max_feature_delta": max_feature_delta,
        "max_probability_delta": max_score_delta, "tolerance": TOLERANCE,
        "errors": errors, "purity": observations, "purity_pass": purity_pass,
        "contract": contract, "contract_hash": adapter.digest(contract),
        "prediction_file_sha256": sources,
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "provenance": {
            "utc": datetime.now(timezone.utc).isoformat(), "seed": None,
            "command": [sys.executable, *sys.argv], "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "jax": jax.__version__, "jaxlib": jaxlib.__version__, "numpy": numpy.__version__,
            "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        },
        "replayed_rows": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve the destination before doing work; never replace old evidence.
    with args.output.open("x", encoding="utf-8") as output:
        try:
            result = run_replay()
        except Exception as exc:
            result = {"status": "FAILED", "fresh_tracing_and_replay_pass": False,
                      "public_api_enabled": False, "public_api_absent": None,
                      "command": [sys.executable, *sys.argv],
                      "utc": datetime.now(timezone.utc).isoformat(),
                      "error": f"{type(exc).__name__}: {exc}"}
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in (
        "replayed_rows", "source_sha256", "contract", "provenance")}, indent=2))
    if not result["fresh_tracing_and_replay_pass"] or not result["public_api_absent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
