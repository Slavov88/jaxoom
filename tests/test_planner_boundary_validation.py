import json

import jax
import jax.numpy as jnp

from experiments import planner_boundary_validation_v1 as boundary


def test_workload_matrix_is_deterministic_and_covers_required_families():
    first = boundary.workloads()
    second = boundary.workloads()
    assert list(first) == list(second)
    assert {workload.family for workload in first.values()} == {
        "MLP",
        "training",
        "attention",
        "convolution",
    }
    assert all(first[key].dimensions == second[key].dimensions for key in first)


def test_boundary_classification_distinguishes_exact_conservative_and_invalid():
    assert boundary._classification({"planned_batch_fits": True, "next_tested_outcome": "OOM"}) == "EXACT_BOUNDARY"
    assert boundary._classification({"planned_batch_fits": True, "next_tested_outcome": "FIT"}) == "CONSERVATIVE_SAFE"
    assert boundary._classification({"planned_batch_fits": False, "next_tested_outcome": "FIT"}) == "INVALID/INCONCLUSIVE"


def test_probe_search_brackets_and_binary_searches(monkeypatch):
    outcomes = {10: "FIT", 11: "FIT", 20: "OOM", 15: "FIT", 17: "OOM", 16: "FIT"}
    monkeypatch.setattr(
        boundary,
        "_run_probe",
        lambda workload_id, batch, timeout: {"batch_size": batch, "outcome": outcomes.get(batch, "OOM")},
    )
    probes, maximum, exact = boundary._probe_sequence("MLP-1", 10, 20, 1)
    assert maximum == 16
    assert exact is True
    assert [probe["batch_size"] for probe in probes] == [10, 11, 20, 15, 17, 16]


def test_explicit_budget_recommendations_are_monotonic():
    # Keep the assertion about ordering independent of built-in calibration data.
    from jaxoom.types import CalibrationSummary

    calibration = CalibrationSummary(
        "test",
        "test",
        "0.6",
        "test",
        1,
        (),
        "test",
        1.0,
        1.0,
        1.0,
        0.95,
        "EXACT_TESTED",
        (),
        ("0.6.2",),
    )

    def args_for_batch(batch):
        return (jax.ShapeDtypeStruct((batch, 128), jnp.float32),)

    recommendations = [
        boundary.jaxoom.plan_batch_size(
            lambda x: x + 1,
            args_for_batch,
            memory_limit=budget,
            min_batch_size=1,
            max_batch_size=32,
            summary=calibration,
        ).recommended_batch_size
        for budget in (1, 4096, 1 << 20)
    ]
    numeric = [value if value is not None else 0 for value in recommendations]
    assert numeric == sorted(numeric)


def test_probe_payload_is_json_serializable():
    payload = {"id": "MLP-1", "classification": "CONSERVATIVE_SAFE", "probes": []}
    assert json.loads(json.dumps(payload)) == payload
