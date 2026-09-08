from types import SimpleNamespace

import jax
import jax.numpy as jnp

import jaxoom
from jaxoom import donation as donation_module
from jaxoom.types import CompilerMemoryReport


S = jax.ShapeDtypeStruct


def test_simple_update_has_compiler_confirmed_candidate():
    report = jaxoom.analyze_donation(lambda x: x + 1, S((8,), "float32"))
    assert report.best is not None
    assert report.best.argnums == (0,)
    assert report.best.status == "COMPILER_CONFIRMED"
    assert report.best.compiler_saving_bytes > 0
    assert report.best.alias_gain_bytes > 0
    assert report.compiler_evaluations == 2
    assert any("must not be used" in warning for warning in report.best.warnings)


def test_shape_mismatch_is_not_compiled_or_recommended():
    report = jaxoom.analyze_donation(lambda x: jnp.ones((4,), x.dtype), S((8,), "float32"))
    assert report.best is None
    assert report.candidates[0].status == "NOT_COMPATIBLE"
    assert report.compiler_evaluations == 1


def test_pytree_argument_is_one_donation_argument():
    report = jaxoom.analyze_donation(
        lambda tree: {"weight": tree["weight"] + 1},
        {"weight": S((8,), "float32")},
    )
    assert report.best is not None
    assert report.best.argnums == (0,)
    assert report.best.input_leaves[0].shape == (8,)
    assert "pytree" in " ".join(report.limitations)


def test_multiple_inputs_and_combination_statuses():
    report = jaxoom.analyze_donation(
        lambda left, right: (left + right, left - right),
        S((8,), "float32"),
        S((8,), "float32"),
    )
    assert report.best is not None
    assert report.best.argnums in ((0,), (1,), (0, 1))
    assert {(0,), (1,), (0, 1)}.issubset({candidate.argnums for candidate in report.candidates})
    combination = next(candidate for candidate in report.candidates if candidate.argnums == (0, 1))
    assert combination.status in {"COMPILER_CONFIRMED", "NOT_BENEFICIAL"}


def test_budget_limits_compiler_evaluations():
    report = jaxoom.analyze_donation(
        lambda a, b, c: (a + b, b + c),
        S((8,), "float32"),
        S((8,), "float32"),
        S((8,), "float32"),
        max_compilations=2,
    )
    assert report.compiler_evaluations <= 2
    assert any(candidate.status == "COMPILATION_FAILED" for candidate in report.candidates)


def test_negative_saving_is_not_confirmed():
    report = jaxoom.analyze_donation(
        lambda left, right: (left + right, left - right),
        S((8,), "float32"),
        S((8,), "float32"),
    )
    combination = next(candidate for candidate in report.candidates if candidate.argnums == (0, 1))
    if combination.compiler_saving_bytes is not None and combination.compiler_saving_bytes > 0:
        assert combination.status == "COMPILER_CONFIRMED"
        assert combination.compiler_confirmed
    else:
        assert combination.status == "NOT_BENEFICIAL"
        assert not combination.compiler_confirmed


def test_compilation_failure_is_reported_per_candidate(monkeypatch):
    def unavailable(fn, args, kwargs=None, donate_argnums=()):
        return CompilerMemoryReport(
            argument_bytes=None,
            output_bytes=None,
            temporary_bytes=None,
            alias_bytes=None,
            compiler_accounted_bytes=None,
            backend="cpu",
            platform="cpu",
            jax_version="test",
            jaxlib_version="test",
            available=False,
            limitations=("synthetic compiler failure",),
        )

    monkeypatch.setattr(donation_module, "_compile_analyze", unavailable)
    report = jaxoom.analyze_donation(lambda x: x + 1, S((8,), "float32"))
    assert report.best is None
    assert report.candidates[0].status == "COMPILATION_FAILED"
    assert report.candidates[0].error == "synthetic compiler failure"


def test_console_rendering_handles_no_best_candidate():
    report = jaxoom.analyze_donation(lambda x: jnp.ones((4,), x.dtype), S((8,), "float32"))
    rendered = report.render()
    assert "JAXOOM DONATION ANALYSIS" in rendered
    assert "No compiler-confirmed beneficial donation" in rendered
