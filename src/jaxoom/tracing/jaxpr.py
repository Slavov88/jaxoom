"""Narrow adapter from Python callables to public JAXPR objects."""
from __future__ import annotations

from typing import Any, Callable

import jax


def trace(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Trace without intentionally executing the numerical workload."""
    result = jax.make_jaxpr(fn)(*args, **kwargs)
    if not hasattr(result, "jaxpr") or not hasattr(result, "consts"):
        raise TypeError(f"expected a closed JAXPR, got {type(result).__name__}")
    return result
