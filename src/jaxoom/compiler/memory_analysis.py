"""Compatibility adapter for JAX compiled memory accounting."""
from __future__ import annotations

from typing import Any, Callable

import jax

from ..types import CompilerMemoryReport


def compile_analyze(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> CompilerMemoryReport:
    """Lower and compile ``fn``, then read ``Compiled.memory_analysis()``.

    This is compiler accounting, not an exact runtime peak. All JAX-specific
    lowering and compatibility handling is intentionally isolated here.
    """
    kwargs = kwargs or {}
    backend = jax.default_backend()
    devices = jax.devices()
    platform = getattr(devices[0], "platform", None) if devices else None
    base = dict(
        backend=backend,
        platform=platform,
        jax_version=getattr(jax, "__version__", "unknown"),
        jaxlib_version=_jaxlib_version(),
    )
    try:
        compiled = jax.jit(fn).lower(*args, **kwargs).compile()
    except Exception as exc:  # compiler availability is backend/version dependent
        return CompilerMemoryReport(
            **base,
            argument_bytes=None,
            output_bytes=None,
            temporary_bytes=None,
            alias_bytes=None,
            compiler_accounted_bytes=None,
            available=False,
            limitations=(f"lower/compile failed: {type(exc).__name__}: {exc}",),
        )

    try:
        stats = compiled.memory_analysis()
    except Exception as exc:
        return CompilerMemoryReport(
            **base,
            argument_bytes=None,
            output_bytes=None,
            temporary_bytes=None,
            alias_bytes=None,
            compiler_accounted_bytes=None,
            available=False,
            limitations=(f"memory_analysis failed: {type(exc).__name__}: {exc}",),
        )
    if stats is None:
        return CompilerMemoryReport(
            **base,
            argument_bytes=None,
            output_bytes=None,
            temporary_bytes=None,
            alias_bytes=None,
            compiler_accounted_bytes=None,
            available=False,
            limitations=("compiled.memory_analysis() returned None.",),
        )

    names = {
        "argument_bytes": "argument_size_in_bytes",
        "output_bytes": "output_size_in_bytes",
        "temporary_bytes": "temp_size_in_bytes",
        "alias_bytes": "alias_size_in_bytes",
    }
    values = {field: _nonnegative_int(getattr(stats, name, None)) for field, name in names.items()}
    missing = tuple(name for field, name in names.items() if values[field] is None)
    accounted = None
    if not missing:
        accounted = (
            values["argument_bytes"]
            + values["output_bytes"]
            + values["temporary_bytes"]
            - values["alias_bytes"]
        )
    limitations = (
        "compiler_accounted_bytes = arguments + outputs + temporaries - aliases; it is not runtime peak memory.",
        "reported fields and their meaning may vary by backend and JAX/jaxlib version.",
    )
    if missing:
        limitations += ("missing memory_analysis fields: " + ", ".join(missing),)
    return CompilerMemoryReport(
        **base,
        **values,
        compiler_accounted_bytes=accounted,
        available=accounted is not None,
        limitations=limitations,
    )


def _nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _jaxlib_version() -> str:
    try:
        import jaxlib

        return getattr(jaxlib, "__version__", "unknown")
    except Exception:
        return "unknown"
