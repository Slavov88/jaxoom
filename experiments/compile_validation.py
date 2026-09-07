"""Compare the MVP structural metric with JAX 0.6.2 compiler memory categories.

This deliberately does not equate the quantities: compiler stats are category
reports, while the MVP metric is a logical sequential live-value peak.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxoom


def elementwise(x):
    return jnp.tanh(jnp.exp(x) * 2)


def matmul(x, w):
    return x @ w


def mlp(x, w1, w2):
    return jax.nn.relu(x @ w1) @ w2


def residual(x):
    y = x @ x.T
    return y + jnp.sin(y)


def reduction(x):
    return jnp.sum(x * x)


def main() -> None:
    cases = [
        ("elementwise", elementwise, (jax.ShapeDtypeStruct((32, 32), "float32"),)),
        ("matmul", matmul, (jax.ShapeDtypeStruct((32, 64), "float32"), jax.ShapeDtypeStruct((64, 16), "float32"))),
        ("mlp", mlp, (jax.ShapeDtypeStruct((32, 64), "float32"), jax.ShapeDtypeStruct((64, 128), "float32"), jax.ShapeDtypeStruct((128, 16), "float32"))),
        ("residual", residual, (jax.ShapeDtypeStruct((32, 32), "float32"),)),
        ("grad", jax.grad(reduction), (jax.ShapeDtypeStruct((32, 32), "float32"),)),
    ]
    print("case,structural,argument,output,temp,alias")
    for name, fn, args in cases:
        report = jaxoom.estimate(fn, *args)
        compiled = jax.jit(fn).lower(*args).compile()
        stats = compiled.memory_analysis()
        print(name, report.estimated_peak_bytes, stats.argument_size_in_bytes, stats.output_size_in_bytes, stats.temp_size_in_bytes, stats.alias_size_in_bytes, sep=",")


if __name__ == "__main__":
    main()
