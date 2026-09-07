from types import SimpleNamespace

import jax
import jax.numpy as jnp

import jaxoom
from jaxoom.compiler import memory_analysis as compiler_module
from jaxoom.compiler.memory_analysis import compile_analyze
from jaxoom.types import CompilerMemoryReport


def test_compiler_report_and_formula():
    report = jaxoom.compile_analyze(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"))
    assert report.available
    assert report.compiler_accounted_bytes == (
        report.argument_bytes + report.output_bytes + report.temporary_bytes - report.alias_bytes
    )
    assert report.backend == jax.default_backend()
    assert report.jax_version == jax.__version__


def test_comparison_sign_convention():
    static = jaxoom.estimate(lambda x: x + 1, jax.ShapeDtypeStruct((8,), "float32"))
    compiler = CompilerMemoryReport(10, 20, 3, 2, 31, "cpu", "cpu", "x", "y", True, ())
    comparison = jaxoom.compare_memory(static, compiler)
    assert comparison.signed_difference_bytes == static.estimated_peak_bytes - 31
    assert comparison.absolute_difference_bytes == abs(comparison.signed_difference_bytes)
    assert comparison.static_overpredicts == (comparison.signed_difference_bytes > 0)


def test_unavailable_compiler_comparison():
    static = jaxoom.estimate(lambda x: x, jax.ShapeDtypeStruct((1,), jnp.float32))
    compiler = CompilerMemoryReport(None, None, None, None, None, "cpu", "cpu", "x", "y", False, ("missing",))
    comparison = jaxoom.compare_memory(static, compiler)
    assert comparison.compiler_accounted_bytes is None
    assert comparison.relative_difference is None
    assert comparison.static_overpredicts is None


def test_none_memory_analysis_is_graceful(monkeypatch):
    class FakeCompiled:
        def memory_analysis(self):
            return None

    class FakeLowered:
        def compile(self):
            return FakeCompiled()

    class FakeJitted:
        def lower(self, *args, **kwargs):
            return FakeLowered()

    monkeypatch.setattr(compiler_module.jax, "jit", lambda fn: FakeJitted())
    report = compile_analyze(lambda x: x, (jax.ShapeDtypeStruct((1,), "float32"),))
    assert not report.available
    assert report.compiler_accounted_bytes is None
    assert "returned None" in report.limitations[0]


def test_missing_compiler_fields_is_graceful(monkeypatch):
    class FakeCompiled:
        def memory_analysis(self):
            return SimpleNamespace(argument_size_in_bytes=4)

    class FakeLowered:
        def compile(self):
            return FakeCompiled()

    class FakeJitted:
        def lower(self, *args, **kwargs):
            return FakeLowered()

    monkeypatch.setattr(compiler_module.jax, "jit", lambda fn: FakeJitted())
    report = compile_analyze(lambda x: x, (jax.ShapeDtypeStruct((1,), "float32"),))
    assert not report.available
    assert report.argument_bytes == 4
    assert report.compiler_accounted_bytes is None
    assert "missing memory_analysis fields" in report.limitations[-1]
