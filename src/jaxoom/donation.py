"""Compiler-confirmed buffer donation advice."""
from __future__ import annotations

import itertools
import warnings
from dataclasses import replace
from typing import Any, Callable

import jax
import numpy as np

from .compiler.memory_analysis import _compile_analyze
from .types import CompilerMemoryReport, DonationCandidate, DonationLeaf, DonationReport


SAFETY_WARNING = "Caller safety unknown: donated argument buffers must not be used after the compiled call."


def analyze_donation(
    fn: Callable[..., Any],
    *args: Any,
    max_compilations: int = 32,
    **kwargs: Any,
) -> DonationReport:
    """Find compiler-confirmed memory opportunities from positional donation.

    This compiles the baseline and selected ``donate_argnums`` variants. It
    does not transform the callable and cannot determine whether the caller
    will reuse a donated input.
    """
    if max_compilations < 1:
        raise ValueError("max_compilations must be at least 1")

    baseline = _compile_analyze(fn, args, kwargs)
    leaves_by_arg = [_array_leaves(arg) for arg in args]
    output_leaves: tuple[DonationLeaf, ...] = ()
    static_error: str | None = None
    try:
        output = jax.eval_shape(fn, *args, **kwargs)
        output_leaves = tuple(leaf for leaf in _array_leaves(output))
    except Exception as exc:
        static_error = f"output shape analysis failed: {type(exc).__name__}: {exc}"

    compatible: list[int] = []
    candidates: list[DonationCandidate] = []
    for index, leaves in enumerate(leaves_by_arg):
        compatible_count = sum(_compatible(leaf, output_leaf) for leaf in leaves for output_leaf in output_leaves)
        is_compatible = bool(leaves) and compatible_count > 0 and static_error is None
        if is_compatible:
            compatible.append(index)
            continue
        warnings_for_candidate = (SAFETY_WARNING,)
        reason = static_error or "no input array leaf has an output leaf with the same shape and dtype"
        candidates.append(
            DonationCandidate(
                argnums=(index,),
                input_bytes=sum(leaf.nbytes for leaf in leaves),
                input_leaves=tuple(leaves),
                compatible_leaf_count=compatible_count,
                status="NOT_COMPATIBLE",
                baseline_compiler_bytes=baseline.compiler_accounted_bytes,
                donated_compiler_bytes=None,
                compiler_saving_bytes=None,
                saving_fraction=None,
                baseline_alias_bytes=baseline.alias_bytes,
                donated_alias_bytes=None,
                alias_gain_bytes=None,
                compiler_confirmed=False,
                warnings=warnings_for_candidate,
                error=reason,
            )
        )

    evaluations = 1
    evaluated: dict[tuple[int, ...], DonationCandidate] = {}

    def evaluate(argnums: tuple[int, ...]) -> DonationCandidate:
        nonlocal evaluations
        if argnums in evaluated:
            return evaluated[argnums]
        if evaluations >= max_compilations:
            candidate = _candidate_from_failure(argnums, leaves_by_arg, baseline, "compiler evaluation budget exhausted")
            evaluated[argnums] = candidate
            return candidate
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            donated = _compile_analyze(fn, args, kwargs, donate_argnums=argnums)
        evaluations += 1
        warning_text = tuple(dict.fromkeys([str(item.message) for item in captured] + list(donated.limitations)))
        candidate = _compare_candidate(argnums, leaves_by_arg, baseline, donated, warning_text)
        evaluated[argnums] = candidate
        return candidate

    for index in compatible:
        evaluated[(index,)] = evaluate((index,))

    remaining = max(0, max_compilations - evaluations)
    if len(compatible) <= 6:
        combinations = (
            combo
            for size in range(2, len(compatible) + 1)
            for combo in itertools.combinations(compatible, size)
        )
        for combo in itertools.islice(combinations, remaining):
            evaluate(tuple(combo))
    else:
        useful = [candidate.argnums[0] for candidate in evaluated.values() if candidate.compiler_confirmed]
        current = tuple(sorted(useful[:1]))
        for index in compatible:
            if index in current or evaluations >= max_compilations:
                continue
            proposal = tuple(sorted(current + (index,)))
            result = evaluate(proposal)
            if result.compiler_confirmed:
                current = proposal

    candidates.extend(evaluated.values())
    candidates.sort(key=_candidate_sort_key)
    best = next((candidate for candidate in candidates if candidate.compiler_confirmed), None)
    limitations = (
        SAFETY_WARNING,
        "Compiler-confirmed saving is compiler-accounted memory, not runtime peak memory.",
        "Donation is evaluated for positional arguments; donating a pytree argument may donate eligible leaves within it.",
    )
    if static_error:
        limitations += (static_error,)
    if evaluations >= max_compilations and (len(compatible) > 0):
        limitations += (f"Compiler evaluation budget: {max_compilations}.",)
    return DonationReport(
        baseline=baseline,
        candidates=tuple(candidates),
        best=best,
        compiler_evaluations=evaluations,
        backend=baseline.backend,
        jax_version=baseline.jax_version,
        limitations=limitations,
    )


def _array_leaves(value: Any) -> tuple[DonationLeaf, ...]:
    leaves, _ = jax.tree_util.tree_flatten(value)
    result = []
    for leaf in leaves:
        metadata = _leaf_metadata(leaf)
        if metadata is not None:
            result.append(metadata)
    return tuple(result)


def _leaf_metadata(leaf: Any) -> DonationLeaf | None:
    if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
        return None
    try:
        shape = tuple(int(dim) for dim in leaf.shape)
        dtype = jax.dtypes.canonicalize_dtype(leaf.dtype)
        nbytes = int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)
    except (TypeError, ValueError, OverflowError):
        return None
    return DonationLeaf(shape=shape, dtype=str(dtype), nbytes=nbytes)


def _compatible(left: DonationLeaf, right: DonationLeaf) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype


def _candidate_from_failure(
    argnums: tuple[int, ...],
    leaves_by_arg: list[tuple[DonationLeaf, ...]],
    baseline: CompilerMemoryReport,
    error: str,
) -> DonationCandidate:
    return DonationCandidate(
        argnums=argnums,
        input_bytes=sum(leaf.nbytes for index in argnums for leaf in leaves_by_arg[index]),
        input_leaves=tuple(leaf for index in argnums for leaf in leaves_by_arg[index]),
        compatible_leaf_count=0,
        status="COMPILATION_FAILED",
        baseline_compiler_bytes=baseline.compiler_accounted_bytes,
        donated_compiler_bytes=None,
        compiler_saving_bytes=None,
        saving_fraction=None,
        baseline_alias_bytes=baseline.alias_bytes,
        donated_alias_bytes=None,
        alias_gain_bytes=None,
        compiler_confirmed=False,
        warnings=(SAFETY_WARNING,),
        error=error,
    )


def _compare_candidate(
    argnums: tuple[int, ...],
    leaves_by_arg: list[tuple[DonationLeaf, ...]],
    baseline: CompilerMemoryReport,
    donated: CompilerMemoryReport,
    warning_text: tuple[str, ...],
) -> DonationCandidate:
    saving = None
    fraction = None
    if baseline.compiler_accounted_bytes is not None and donated.compiler_accounted_bytes is not None:
        saving = baseline.compiler_accounted_bytes - donated.compiler_accounted_bytes
        fraction = saving / baseline.compiler_accounted_bytes if baseline.compiler_accounted_bytes else None
    alias_gain = None
    if baseline.alias_bytes is not None and donated.alias_bytes is not None:
        alias_gain = donated.alias_bytes - baseline.alias_bytes
    available = baseline.available and donated.available and saving is not None
    status = "COMPILER_CONFIRMED" if available and saving > 0 else "NOT_BENEFICIAL" if available else "COMPILATION_FAILED"
    error = None if available else "; ".join(donated.limitations)
    return DonationCandidate(
        argnums=argnums,
        input_bytes=sum(leaf.nbytes for index in argnums for leaf in leaves_by_arg[index]),
        input_leaves=tuple(leaf for index in argnums for leaf in leaves_by_arg[index]),
        compatible_leaf_count=sum(len(leaves_by_arg[index]) for index in argnums),
        status=status,
        baseline_compiler_bytes=baseline.compiler_accounted_bytes,
        donated_compiler_bytes=donated.compiler_accounted_bytes,
        compiler_saving_bytes=saving,
        saving_fraction=fraction,
        baseline_alias_bytes=baseline.alias_bytes,
        donated_alias_bytes=donated.alias_bytes,
        alias_gain_bytes=alias_gain,
        compiler_confirmed=status == "COMPILER_CONFIRMED",
        warnings=(SAFETY_WARNING,) + warning_text,
        error=error,
    )


def _candidate_sort_key(candidate: DonationCandidate) -> tuple[int, int, float, int, tuple[int, ...]]:
    return (
        0 if candidate.compiler_confirmed else 1,
        -(candidate.compiler_saving_bytes or 0),
        -(candidate.saving_fraction or float("-inf")),
        len(candidate.argnums),
        candidate.argnums,
    )
