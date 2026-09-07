import jax
import jax.numpy as jnp

import jaxoom


def test_public_api_and_budget():
    report = jaxoom.estimate(lambda x: jnp.tanh(jnp.exp(x)), jax.ShapeDtypeStruct((8,), "float32"), memory_limit="1 KiB")
    assert isinstance(report, jaxoom.MemoryReport)
    assert report.estimated_peak_bytes == 2 * 8 * 4
    assert report.assessment == "LIKELY FIT"
    assert "sequential JAXPR" in report.render()


def test_matmul_and_abstract_large_shape():
    report = jaxoom.estimate(
        lambda x, w: x @ w,
        jax.ShapeDtypeStruct((100_000, 1024), "float16"),
        jax.ShapeDtypeStruct((1024, 8), "float16"),
    )
    assert report.estimated_peak_bytes >= 100_000 * 1024 * 2
    assert report.peak.primitive == "dot_general"


def test_grad_and_value_and_grad_are_analyzable():
    def loss(x):
        return jnp.sum(x * x)

    assert jaxoom.estimate(jax.grad(loss), jax.ShapeDtypeStruct((4, 4), "float32")).equations_analyzed > 0
    assert jaxoom.estimate(jax.value_and_grad(loss), jax.ShapeDtypeStruct((4, 4), "float32")).equations_analyzed > 0


def test_nested_construct_lowers_confidence():
    report = jaxoom.estimate(lambda x: jax.nn.relu(x), jax.ShapeDtypeStruct((4,), "float32"))
    assert report.confidence == "limited"
    assert report.unsupported_constructs
