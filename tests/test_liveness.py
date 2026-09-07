import jax
import jax.numpy as jnp

from jaxoom import estimate


def test_linear_chain_keeps_input_and_output_during_equation():
    def f(x):
        return (x + 1) * 2

    report = estimate(f, jax.ShapeDtypeStruct((2, 2), "float32"))
    assert report.estimated_peak_bytes == 2 * 4 * 4
    assert report.equations_analyzed == 2


def test_reused_value_is_live_until_last_consumer():
    def f(x):
        a = x + 1
        b = a * 2
        return a + b

    report = estimate(f, jax.ShapeDtypeStruct((2, 2), "float32"))
    assert report.estimated_peak_bytes == 3 * 4 * 4
    assert len(report.peak.live_buffers) == 3


def test_multiple_outputs_remain_live():
    def f(x):
        return x + 1, x * 2

    report = estimate(f, jax.ShapeDtypeStruct((2, 2), "float32"))
    assert report.estimated_peak_bytes == 3 * 4 * 4
    assert sum(buffer.is_output for buffer in report.peak.live_buffers) == 2


def test_zero_sized_array():
    report = estimate(lambda x: x + 1, jax.ShapeDtypeStruct((0, 10), jnp.float32))
    assert report.estimated_peak_bytes == 0
