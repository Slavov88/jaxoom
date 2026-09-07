import jax.numpy as jnp

import jaxoom


def residual(x):
    y = x @ x.T
    return y + jnp.sin(y)


report = jaxoom.estimate(residual, (jnp.ones((1024, 1024), dtype=jnp.float32)), memory_limit="16 GiB")
report.print()
