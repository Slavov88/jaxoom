"""Peak calculation for the sequential live-value model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..types import BufferInfo, PeakPoint
from .liveness import LifetimeModel


@dataclass(frozen=True)
class PeakAnalysis:
    peak: PeakPoint
    largest_buffers: tuple[BufferInfo, ...]


def calculate_peak(model: LifetimeModel, n_eqns: int, primitives: Iterable[str]) -> PeakAnalysis:
    """Calculate a conservative peak: outputs are allocated before final-use inputs release."""
    by_id = {buffer.variable_id: buffer for buffer in model.buffers}
    live: set[str] = {
        buffer.variable_id for buffer in model.buffers if buffer.is_input and buffer.birth == -1
    }
    largest = tuple(sorted(model.buffers, key=lambda item: (-item.nbytes, item.variable_id)))
    best = _point(None, None, live, by_id)
    primitive_names = tuple(primitives)

    for index in range(n_eqns):
        live = {key for key in live if by_id[key].last_use >= index}
        before = _point(index, primitive_names[index], live, by_id)
        if before.live_bytes > best.live_bytes:
            best = before
        live.update(model.equation_outputs[index])
        allocated = _point(index, primitive_names[index], live, by_id)
        if allocated.live_bytes > best.live_bytes:
            best = allocated
        live = {
            key for key in live
            if not (by_id[key].last_use == index and not by_id[key].is_output)
        }

    after = _point(n_eqns if n_eqns else None, "<return>" if n_eqns else None, live, by_id)
    if after.live_bytes > best.live_bytes:
        best = after
    return PeakAnalysis(best, largest)


def _point(equation: int | None, primitive: str | None, live: set[str], by_id: dict[str, BufferInfo]) -> PeakPoint:
    buffers = tuple(sorted((by_id[key] for key in live), key=lambda item: (-item.nbytes, item.variable_id)))
    return PeakPoint(equation, primitive, sum(item.nbytes for item in buffers), buffers)
