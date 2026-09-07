"""Exact byte-size calculation for supported JAX array avals."""
from __future__ import annotations

import math
from typing import Any


def array_nbytes(aval: Any) -> int:
    """Return the logical dense byte size of a JAX abstract array value."""
    shape = tuple(aval.shape)
    if any(not isinstance(dim, int) for dim in shape):
        raise TypeError(f"dynamic dimensions are not supported: {shape!r}")
    try:
        itemsize = int(aval.dtype.itemsize)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"unsupported dtype {getattr(aval, 'dtype', None)!r}") from exc
    if itemsize < 0:
        raise TypeError(f"invalid dtype itemsize: {itemsize}")
    return math.prod(shape) * itemsize


def format_bytes(value: int) -> str:
    """Format bytes using binary units."""
    if value < 0:
        raise ValueError("byte count cannot be negative")
    if value < 1024:
        return f"{value} B"
    number = float(value)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        number /= 1024
        if number < 1024 or unit == "TiB":
            return f"{number:.2f} {unit}"
    raise AssertionError("unreachable")


def parse_memory_limit(value: str | int | None) -> int | None:
    """Parse an intentionally small binary-unit memory-limit grammar."""
    if value is None or isinstance(value, int):
        if isinstance(value, int) and value < 0:
            raise ValueError("memory limit cannot be negative")
        return value
    text = value.strip().upper().replace(" ", "")
    units = {"B": 1, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4}
    for unit, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if text.endswith(unit):
            number = text[: -len(unit)]
            try:
                return int(float(number) * multiplier)
            except ValueError as exc:
                raise ValueError(f"invalid memory limit: {value!r}") from exc
    raise ValueError("memory_limit must be bytes or a value such as '8 GiB'")
