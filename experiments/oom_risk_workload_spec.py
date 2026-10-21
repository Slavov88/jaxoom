"""Internal restricted adapter for frozen V2; deliberately absent from jaxoom.

This module reproduces a historical classifier, not a generic callable API.
The 3 GiB denominator is a reference constant, not an enforced runtime quota.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import textwrap
from types import MappingProxyType
from typing import Any

SPEC_VERSION = "frozen-v2-workload-1"
FEATURE_VERSION = "frozen-v2-legacy-zero-1"
PUBLIC_API_ENABLED = False
REFERENCE_BYTES = 3 * 1024**3
UPPER_RATIO = 1.66765051935476
FIT_THRESHOLD = 0.20
MODEL_HASH = "9e7f03a6ca027950e84a42179df3b8743d8f0b81157b500027fb765de1a42e96"
MODEL_FILE_SHA256 = "5b2eeedf9a180af1c79678e94ecf3401b0256b4906e1d8f5180f058d3b354ae3"
TEMPLATE_SOURCE_SHA256 = "b0afa5a7e04f6d254ff8a807772cb0c0ff0958858a579d058f22564916dca04b"
MODEL_PATH = Path(__file__).with_name("oom_risk_model_v2_2026-09-12.json")
FEATURE_NAMES = (
    "structural_peak_over_budget", "calibrated_upper_over_budget",
    "largest_over_budget", "top_two_peak_live_over_budget", "peak_live_over_budget",
    "dtype_bytes", "config_numeric_count", "config_numeric_max", "config_numeric_log_product",
)
FAMILY_FIELDS = MappingProxyType({
    "attention": ("batch", "heads", "sequence", "head_dim"),
    "convolution": ("batch", "height", "width", "channels", "out_channels"),
    "mlp": ("batch", "width", "depth"),
    "training": ("batch", "width", "depth"),
    "autodiff": ("batch", "width", "depth"),
    "transformer": ("sequence", "width", "heads", "layers"),
    "matmul": ("m", "k", "n", "depth"),
    "reduction": ("batch", "length", "width", "depth"),
    "scan": ("batch", "length", "width", "depth"),
})


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True, init=False)
class WorkloadSpec:
    """An immutable, canonical descriptor of one pinned experiment template.

    Dimensions are positive Python integers, never booleans or floats. The
    unrolling limit is a tracing resource guard, not a validated OOD policy.
    Historical scan depth remains metadata-only; changing that needs a new model.
    """

    family: str
    configuration: tuple[tuple[str, int], ...]
    dtype: str
    version: str

    def __init__(self, family: str, configuration: Mapping[str, int], dtype: str,
                 *, version: str = SPEC_VERSION):
        if type(version) is not str or version != SPEC_VERSION:
            raise ValueError("unsupported workload specification version")
        if type(family) is not str or family not in FAMILY_FIELDS:
            raise ValueError("unsupported workload family")
        if type(dtype) is not str or dtype not in ("float16", "float32"):
            raise ValueError("dtype must be float16 or float32")
        if not isinstance(configuration, Mapping):
            raise TypeError("configuration must be a mapping")
        config = dict(configuration)
        if set(config) != set(FAMILY_FIELDS[family]):
            raise ValueError(f"configuration fields must be exactly {FAMILY_FIELDS[family]}")
        if any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in config.values()):
            raise ValueError("dimensions must be positive Python integers <= 2**31 - 1")
        if any(config.get(n, 1) > 64 for n in ("depth", "layers")):
            raise ValueError("depth/layers exceed the internal tracing resource limit of 64")
        if family == "transformer" and config["width"] % config["heads"]:
            raise ValueError("transformer width must be divisible by heads")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "configuration", tuple(sorted(config.items())))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "version", version)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "family": self.family,
                "configuration": dict(self.configuration), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkloadSpec:
        if not isinstance(value, Mapping) or set(value) != {"version", "family", "configuration", "dtype"}:
            raise ValueError("specification requires exactly version, family, configuration, dtype")
        return cls(value["family"], value["configuration"], value["dtype"], version=value["version"])

    @property
    def spec_hash(self) -> str:
        return digest(self.to_dict())


def _require_spec(spec: WorkloadSpec) -> None:
    if type(spec) is not WorkloadSpec:
        raise TypeError("a WorkloadSpec is required; arbitrary callables are unsupported")
    # Revalidate at the trust boundary, including forged/deserialized objects.
    if WorkloadSpec.from_dict(spec.to_dict()) != spec:
        raise ValueError("noncanonical workload specification")


def _template_factory():
    from experiments.oom_risk_expansion_v2 import candidate_functions

    source = textwrap.dedent(inspect.getsource(candidate_functions)).strip().replace("\r\n", "\n")
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != TEMPLATE_SOURCE_SHA256:
        raise RuntimeError("frozen V2 workload templates have changed")
    return candidate_functions


def build_workload(spec: WorkloadSpec):
    """Construct both the function and its abstract arguments from the same spec."""
    _require_spec(spec)
    return _template_factory()(spec.family, dict(spec.configuration), spec.dtype)


def load_frozen_model() -> dict[str, Any]:
    raw = MODEL_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_FILE_SHA256:
        raise RuntimeError("frozen V2 artifact bytes have changed")
    model = json.loads(raw)
    hashed_fields = {"features": model["feature_names"], "means": model["means"],
                     "scales": model["scales"], "coefficients": model["coefficients"]}
    if model["model_hash"] != MODEL_HASH or digest(hashed_fields) != MODEL_HASH:
        raise RuntimeError("frozen V2 model hash mismatch")
    if tuple(model["feature_names"]) != FEATURE_NAMES or model["intercept"] != model["coefficients"][0]:
        raise RuntimeError("frozen V2 schema/intercept mismatch")
    return model


def extract_features(spec: WorkloadSpec) -> dict[str, float]:
    """Fresh abstract tracing only, with exact historical V2 feature semantics."""
    import jax
    import jaxlib
    import jaxoom

    _require_spec(spec)
    if jax.__version__ != "0.11.0" or jaxlib.__version__ != "0.11.0":
        raise RuntimeError("frozen V2 tracing requires JAX and jaxlib 0.11.0")
    fn, args = build_workload(spec)
    report = jaxoom.estimate(fn, *args)
    structural = float(report.estimated_peak_bytes)
    top_two = float(sum(sorted((b.nbytes for b in report.peak.live_buffers), reverse=True)[:2]))
    nums = [float(v) for _, v in spec.configuration]
    return {
        "structural_peak_over_budget": structural / REFERENCE_BYTES,
        "calibrated_upper_over_budget": structural * UPPER_RATIO / REFERENCE_BYTES,
        # This feature was absent in every training row and imputed to zero.
        # It is deliberately NOT largest_buffer_bytes / REFERENCE_BYTES.
        "largest_over_budget": 0.0,
        "top_two_peak_live_over_budget": top_two / REFERENCE_BYTES,
        "peak_live_over_budget": structural / REFERENCE_BYTES,
        "dtype_bytes": 2.0 if spec.dtype == "float16" else 4.0,
        "config_numeric_count": float(len(nums)),
        "config_numeric_max": max(nums),
        "config_numeric_log_product": math.log1p(math.prod(max(1.0, v) for v in nums)),
    }


def _score_features(features: Mapping[str, float], model: Mapping[str, Any]) -> float:
    """Internal inference primitive; missing/nonfinite features fail closed."""
    if set(features) != set(FEATURE_NAMES) or any(not math.isfinite(v) for v in features.values()):
        raise ValueError("exact finite frozen V2 feature vector required")
    if features["largest_over_budget"] != 0.0:
        raise ValueError("frozen V2 requires the legacy zero feature")
    x = [(features[n] - mean) / scale for n, mean, scale in
         zip(FEATURE_NAMES, model["means"], model["scales"])]
    z = model["intercept"] + sum(c * v for c, v in zip(model["coefficients"][1:], x))
    return 1 / (1 + math.exp(-max(-40, min(40, z))))


def pipeline_contract() -> dict[str, Any]:
    """Hash the full adapter contract, beyond V2's original coefficient hash."""
    import jaxoom

    load_frozen_model()
    _template_factory()
    analysis_root = Path(jaxoom.__file__).resolve().parent
    expected_root = Path(__file__).resolve().parents[1] / "src" / "jaxoom"
    if analysis_root != expected_root:
        raise RuntimeError("replay must import jaxoom from this checkout's src directory")
    return {
        "spec_version": SPEC_VERSION, "feature_version": FEATURE_VERSION,
        "model_hash": MODEL_HASH, "model_file_sha256": MODEL_FILE_SHA256,
        "template_source_sha256": TEMPLATE_SOURCE_SHA256,
        "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "static_analysis_source_sha256": {
            p.relative_to(analysis_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(analysis_root.rglob("*.py"))
        },
        "feature_names": list(FEATURE_NAMES), "family_fields": dict(FAMILY_FIELDS),
        "reference_bytes": REFERENCE_BYTES, "reference_is_runtime_quota": False,
        "upper_ratio": UPPER_RATIO, "fit_threshold": FIT_THRESHOLD,
        "jax": "0.11.0", "jaxlib": "0.11.0", "dtypes": ["float16", "float32"],
        "dimension_max": 2**31 - 1, "unroll_resource_limit": 64,
        "legacy_largest_feature": 0.0, "legacy_scan_depth": "metadata_only",
        "public_api_enabled": PUBLIC_API_ENABLED,
        "runtime_applicability": "NOT_QUALIFIED; historical RTX 3050 campaign only",
    }
