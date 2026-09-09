"""Compilation-free discrete batch-size planning."""
from __future__ import annotations

from typing import Any, Callable

from .api import assess, estimate
from .device import device_budget
from .types import BatchSizePlan, BatchSizeTrial, DeviceBudget


def plan_batch_size(
    fn: Callable[..., Any],
    args_for_batch: Callable[[int], Any],
    *,
    memory_limit: str | int = "auto",
    min_batch_size: int = 1,
    max_batch_size: int = 1024,
    max_evaluations: int = 64,
    device_budget_override: DeviceBudget | None = None,
) -> BatchSizePlan:
    """Find the largest conservatively assessed batch without compiling it.

    ``args_for_batch`` explicitly owns the relationship between a batch size
    and positional arguments. It must return a tuple or list of arguments
    suitable for :func:`estimate`, normally abstract JAX values.
    """
    _validate_bounds(min_batch_size, max_batch_size, max_evaluations)
    auto = memory_limit == "auto"
    frozen_budget = device_budget_override if auto else None
    if auto and frozen_budget is None:
        frozen_budget = device_budget()
    parsed_limit = None if auto else _parse_explicit_limit(memory_limit)
    source = "AUTO_DEVICE_SNAPSHOT" if auto else "EXPLICIT"
    trials: list[BatchSizeTrial] = []
    evaluated: set[int] = set()
    warnings: list[str] = []

    def evaluate(batch: int) -> BatchSizeTrial:
        if batch in evaluated:
            return next(item for item in trials if item.batch_size == batch)
        if len(trials) >= max_evaluations:
            raise _SearchLimit
        try:
            args = args_for_batch(batch)
            if not isinstance(args, (tuple, list)):
                raise TypeError("args_for_batch must return a tuple or list of positional arguments")
            report = estimate(fn, *tuple(args))
            assessment = assess(report, parsed_limit if not auto else "auto", device_budget=frozen_budget)
            trial = BatchSizeTrial(batch, assessment, None)
        except _SearchLimit:
            raise
        except Exception as exc:
            trial = BatchSizeTrial(batch, None, f"{type(exc).__name__}: {exc}")
        evaluated.add(batch)
        trials.append(trial)
        return trial

    status = "COMPLETE"
    low_fit: int | None = None
    high_risk: int | None = None
    try:
        first = evaluate(min_batch_size)
        if _fits(first):
            low_fit = min_batch_size
            if min_batch_size == max_batch_size:
                status = "UPPER_BOUND_REACHED"
            else:
                candidate = min_batch_size
                while candidate < max_batch_size:
                    candidate = min(max_batch_size, max(candidate + 1, candidate * 2))
                    trial = evaluate(candidate)
                    if _fits(trial):
                        low_fit = candidate
                        if candidate == max_batch_size:
                            status = "UPPER_BOUND_REACHED"
                            break
                    else:
                        high_risk = candidate
                        break
                if high_risk is not None:
                    while high_risk - low_fit > 1:
                        middle = (low_fit + high_risk) // 2
                        if _fits(evaluate(middle)):
                            low_fit = middle
                        else:
                            high_risk = middle
        else:
            status = "NOTHING_FITS"
            high_risk = min_batch_size
    except _SearchLimit:
        status = "SEARCH_LIMIT_REACHED"
        warnings.append(f"maximum evaluation count reached ({max_evaluations})")

    monotonic = _is_monotonic(trials)
    if not monotonic:
        status = "NON_MONOTONIC"
        warnings.append("evaluated structural estimates decreased materially with larger batch size")
        fitting = [trial.batch_size for trial in trials if _fits(trial)]
        low_fit = max(fitting) if fitting else None
        high_risk = min((trial.batch_size for trial in trials if not _fits(trial)), default=None)
    ordered = sorted(trials, key=lambda trial: trial.batch_size)
    next_trial = next((trial for trial in ordered if low_fit is not None and trial.batch_size > low_fit and not _fits(trial)), None)
    if next_trial is None:
        next_trial = next((trial for trial in ordered if not _fits(trial)), None)
    return BatchSizePlan(
        recommended_batch_size=low_fit,
        max_tested_batch_size=max(evaluated) if evaluated else None,
        next_failing_or_riskier=next_trial,
        trials=tuple(ordered),
        evaluations=len(trials),
        memory_limit_bytes=(frozen_budget.assessment_budget_bytes if auto and frozen_budget else parsed_limit),
        memory_limit_source=source,
        device_budget=frozen_budget,
        status=status,
        upper_bound_reached=status == "UPPER_BOUND_REACHED",
        monotonic=monotonic,
        warnings=tuple(warnings),
    )


def _fits(trial: BatchSizeTrial) -> bool:
    return trial.assessment is not None and trial.assessment.calibrated and trial.assessment.interval.upper_bytes <= (trial.assessment.memory_limit_bytes or -1)


def _is_monotonic(trials: list[BatchSizeTrial]) -> bool:
    previous = None
    for trial in sorted(trials, key=lambda item: item.batch_size):
        if trial.assessment is None:
            continue
        value = trial.assessment.structural_peak_bytes
        if previous is not None and value < previous - 1024:
            return False
        previous = value
    return True


def _parse_explicit_limit(value: str | int) -> int:
    from .analysis.tensor_size import parse_memory_limit

    parsed = parse_memory_limit(value)
    if parsed is None:
        raise ValueError("memory_limit must be explicit or 'auto'")
    return parsed


def _validate_bounds(minimum: int, maximum: int, evaluations: int) -> None:
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        raise TypeError("batch bounds must be integers")
    if minimum < 1 or maximum < minimum:
        raise ValueError("batch sizes must satisfy 1 <= min_batch_size <= max_batch_size")
    if evaluations < 1:
        raise ValueError("max_evaluations must be positive")


class _SearchLimit(Exception):
    pass
