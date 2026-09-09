import jax
import jax.numpy as jnp

import jaxoom
from jaxoom import batch_planner
from jaxoom.types import DeviceBudget, DeviceMemorySnapshot


def args_for_batch(batch):
    return (jax.ShapeDtypeStruct((batch, 8), jnp.float32),)


def test_planner_finds_largest_discrete_batch_without_compiling(monkeypatch):
    calls = []
    original = batch_planner.estimate
    monkeypatch.setattr(batch_planner, "estimate", lambda fn, *args: (calls.append(args[0].shape[0]) or original(fn, *args)))
    monkeypatch.setattr(jax, "jit", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("compiled")))
    plan = jaxoom.plan_batch_size(lambda x: x * 2, args_for_batch, memory_limit="1 KiB", max_batch_size=32)
    assert plan.recommended_batch_size is not None
    assert plan.next_failing_or_riskier.batch_size == plan.recommended_batch_size + 1
    assert plan.evaluations == len(calls)
    assert plan.status == "COMPLETE"


def test_planner_nothing_and_everything_fit():
    fn = lambda x: x + 1
    none = jaxoom.plan_batch_size(fn, args_for_batch, memory_limit=1, max_batch_size=8)
    assert none.recommended_batch_size is None
    assert none.status == "NOTHING_FITS"
    all_fit = jaxoom.plan_batch_size(fn, args_for_batch, memory_limit="1 GiB", max_batch_size=8)
    assert all_fit.recommended_batch_size == 8
    assert all_fit.upper_bound_reached


def test_planner_uses_one_frozen_auto_budget(monkeypatch):
    snapshot = DeviceMemorySnapshot(
        backend="cpu", device_kind="test", device_id=0, device_uuid=None,
        physical_total_bytes=4 * 1024**3, driver_used_bytes=0, driver_free_bytes=4 * 1024**3,
        jax_bytes_in_use=0, jax_peak_bytes_in_use=0, jax_pool_bytes=0, external_used_bytes=0,
        allocator_mode="test", allocator_preallocate=False, allocator_memory_fraction=None,
        effective_available_bytes=4 * 1024**3, measurement_sources=(), limitations=(), timestamp="test",
    )
    budget = DeviceBudget(snapshot, 4 * 1024**3, 64 * 1024**2, 4 * 1024**3 - 64 * 1024**2, "test", ())
    calls = []
    monkeypatch.setattr(batch_planner, "device_budget", lambda: (calls.append(1) or budget))
    plan = jaxoom.plan_batch_size(lambda x: x + 1, args_for_batch, memory_limit="auto", max_batch_size=4)
    assert calls == [1]
    assert plan.device_budget is budget
    assert plan.memory_limit_source == "AUTO_DEVICE_SNAPSHOT"


def test_planner_rejects_uncalibrated_explicitly(monkeypatch):
    monkeypatch.setattr(batch_planner, "estimate", lambda fn, *args: jaxoom.estimate(fn, *args))
    plan = jaxoom.plan_batch_size(lambda x: x + 1, args_for_batch, memory_limit="1 GiB", max_batch_size=4)
    assert plan.recommended_batch_size == 4


def test_planner_respects_evaluation_cap():
    plan = jaxoom.plan_batch_size(lambda x: x + 1, args_for_batch, memory_limit="1 KiB", max_batch_size=1024, max_evaluations=3)
    assert plan.status == "SEARCH_LIMIT_REACHED"
    assert plan.evaluations == 3


def test_planner_reports_invalid_factory():
    plan = jaxoom.plan_batch_size(lambda x: x, lambda batch: jax.ShapeDtypeStruct((batch,), jnp.float32), memory_limit="1 GiB")
    assert plan.recommended_batch_size is None
    assert plan.trials[0].error and "tuple or list" in plan.trials[0].error


def test_planner_reports_uncalibrated_basis(monkeypatch):
    monkeypatch.setattr("jaxoom.calibration._select_summary", lambda backend, version: None)
    plan = jaxoom.plan_batch_size(lambda x: x + 1, args_for_batch, memory_limit="1 GiB", max_batch_size=4)
    assert plan.recommended_batch_size is None
    assert plan.status == "NOTHING_FITS"
    assert all(not trial.assessment.calibrated for trial in plan.trials if trial.assessment)


def test_planner_reports_non_monotonic_estimates(monkeypatch):
    original = batch_planner.estimate
    reports = []
    def fake_estimate(fn, *args):
        report = original(fn, *args)
        if args[0].shape[0] == 2:
            from dataclasses import replace
            report = replace(report, estimated_peak_bytes=1)
        reports.append(report)
        return report
    monkeypatch.setattr(batch_planner, "estimate", fake_estimate)
    def larger_args(batch):
        return (jax.ShapeDtypeStruct((batch, 1024), jnp.float32),)
    plan = jaxoom.plan_batch_size(lambda x: x + 1, larger_args, memory_limit="1 MiB", max_batch_size=8)
    assert plan.status == "NON_MONOTONIC"
    assert not plan.monotonic
