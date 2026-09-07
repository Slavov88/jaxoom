"""Sequential JAXPR variable lifetimes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..types import BufferInfo
from .tensor_size import array_nbytes


@dataclass(frozen=True)
class LifetimeModel:
    buffers: tuple[BufferInfo, ...]
    equation_inputs: tuple[tuple[str, ...], ...]
    equation_outputs: tuple[tuple[str, ...], ...]
    unsupported_constructs: tuple[str, ...]


def _is_data_var(value: Any) -> bool:
    return type(value).__name__ in {"Var", "DropVar"}


def _is_drop_var(value: Any) -> bool:
    return type(value).__name__ == "DropVar"


def _key(value: Any) -> str:
    return f"v{id(value)}"


def build_lifetimes(closed_jaxpr: Any) -> LifetimeModel:
    """Build lifetimes using conservative input+output overlap at each equation."""
    jaxpr = closed_jaxpr.jaxpr
    n_eqns = len(jaxpr.eqns)
    all_vars: dict[str, tuple[Any, str, int, bool]] = {}
    uses: dict[str, list[int]] = {}
    outputs = {_key(v) for v in jaxpr.outvars if _is_data_var(v) and not _is_drop_var(v)}

    def register(var: Any, producer: str, birth: int, is_input: bool) -> None:
        if not _is_data_var(var) or _is_drop_var(var):
            return
        key = _key(var)
        if key not in all_vars:
            all_vars[key] = (var, producer, birth, is_input)
            uses[key] = []

    for var in (*jaxpr.invars, *jaxpr.constvars):
        register(var, "input", -1, True)

    equation_inputs: list[tuple[str, ...]] = []
    equation_outputs: list[tuple[str, ...]] = []
    unsupported: set[str] = set()
    for index, eqn in enumerate(jaxpr.eqns):
        in_keys: list[str] = []
        for var in eqn.invars:
            if _is_data_var(var) and not _is_drop_var(var):
                register(var, "input", -1, True)
                key = _key(var)
                uses[key].append(index)
                in_keys.append(key)
        out_keys: list[str] = []
        for var in eqn.outvars:
            if _is_data_var(var) and not _is_drop_var(var):
                register(var, str(eqn.primitive), index, False)
                out_keys.append(_key(var))
        equation_inputs.append(tuple(in_keys))
        equation_outputs.append(tuple(out_keys))
        if _has_nested_computation(eqn):
            unsupported.add(str(eqn.primitive))

    buffers: list[BufferInfo] = []
    for key, (var, producer, birth, is_input) in all_vars.items():
        try:
            nbytes = array_nbytes(var.aval)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(f"cannot determine dense size for {var!r}") from exc
        last_use = max(uses[key], default=-1)
        if key in outputs:
            last_use = max(last_use, n_eqns)
        buffers.append(
            BufferInfo(
                variable_id=key,
                shape=tuple(var.aval.shape),
                dtype=str(var.aval.dtype),
                nbytes=nbytes,
                producer=producer,
                birth=birth,
                last_use=last_use,
                is_input=is_input,
                is_output=key in outputs,
            )
        )
    return LifetimeModel(
        tuple(buffers), tuple(equation_inputs), tuple(equation_outputs), tuple(sorted(unsupported))
    )


def _has_nested_computation(eqn: Any) -> bool:
    primitive = str(eqn.primitive)
    if primitive in {"scan", "while", "cond", "call", "pjit", "custom_jvp_call", "custom_vjp_call", "remat2"}:
        return True
    return any("jaxpr" in key or key in {"branches", "body_jaxpr", "cond_jaxpr"} for key in eqn.params)
