"""Prospective OOM-risk expansion candidate generation and frozen-V1 selection.

This module performs static JAX tracing only. It never compiles or executes a
candidate. The selection plan is frozen before the runtime campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent
BUDGET = 3 * 1024**3
V1_MODEL_PATH = ROOT / "oom_risk_models_v1_2026-09-12.json"
V1_DATASET_PATH = ROOT / "oom_risk_dataset_v1_2026-09-12.json"
FREEZE_PATH = ROOT / "oom_risk_model_v1_frozen_2026-09-12.json"


def dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=lambda x: x.item() if hasattr(x, "item") else str(x))


def sha(value: Any) -> str:
    return hashlib.sha256(dump(value).encode()).hexdigest()


def group_id(family: str, config: dict[str, Any], dtype: str) -> str:
    return f"{family}|{dump(config)}|{dtype}"


def graph_fingerprint(family: str, config: dict[str, Any], dtype: str) -> str:
    return hashlib.sha256(group_id(family, config, dtype).encode()).hexdigest()[:16]


def load_v1() -> tuple[dict[str, Any], dict[str, Any]]:
    model_artifact = json.loads(V1_MODEL_PATH.read_text())
    dataset = json.loads(V1_DATASET_PATH.read_text())
    if model_artifact.get("status") != "COMPUTATIONALLY_VERIFIED":
        raise RuntimeError("V1 model artifact is not verified")
    return model_artifact["selected"], dataset


def freeze_v1() -> dict[str, Any]:
    model, dataset = load_v1()
    rows = dataset["rows"]
    structural_ratios = [r["features"]["calibrated_upper_bytes"] / r["features"]["structural_peak_bytes"] for r in rows if r["features"].get("calibrated_upper_bytes") is not None and r["features"].get("structural_peak_bytes")]
    central_ratios = [r["features"]["calibrated_central_bytes"] / r["features"]["structural_peak_bytes"] for r in rows if r["features"].get("calibrated_central_bytes") is not None and r["features"].get("structural_peak_bytes")]
    freeze = {
        "status": "COMPUTATIONALLY_VERIFIED",
        "frozen": True,
        "frozen_at": "2026-09-12",
        "source_model_artifact": V1_MODEL_PATH.name,
        "source_dataset_artifact": V1_DATASET_PATH.name,
        "dataset_hash": model["dataset_hash"],
        "model_hash": sha(model),
        "model": model,
        "feature_schema": model["feature_names"],
        "normalization": {"means": model["means"], "scales": model["scales"]},
        "calibration_proxy_for_static_candidates": {
            "central_ratio_median": float(np.median(central_ratios)),
            "upper_ratio_median": float(np.median(structural_ratios)),
            "source": "V1 normalized dataset; static candidate scoring only",
            "not_runtime_evidence": True,
        },
        "selection_rule": "frozen V1 coefficients and training normalization; no candidate outcomes used",
        "inference_purity": {"target_compilation": False, "target_execution": False, "runtime_allocator_state": False},
    }
    if FREEZE_PATH.exists():
        existing = json.loads(FREEZE_PATH.read_text())
        if existing.get("model_hash") != freeze["model_hash"] or existing.get("dataset_hash") != freeze["dataset_hash"]:
            raise RuntimeError("refusing to overwrite an incompatible frozen V1 artifact")
        return existing
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return freeze


def dtype_bytes(dtype: str) -> float:
    return 2.0 if dtype == "float16" else 4.0


def candidate_functions(family: str, config: dict[str, Any], dtype: str) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    d = dtype
    if family == "attention":
        b, h, s, hd = (config[k] for k in ("batch", "heads", "sequence", "head_dim"))
        width = h * hd
        def fn(x):
            q = x.reshape(b, s, h, hd); k = q; v = q
            scores = jnp.einsum("bshd,bthd->bhst", q, k) / jnp.sqrt(jnp.asarray(hd, dtype=x.dtype))
            weights = jax.nn.softmax(scores, axis=-1)
            return jnp.einsum("bhst,bthd->bshd", weights, v).reshape(b, s, width)
        return fn, (jax.ShapeDtypeStruct((b, s, width), d),)
    if family == "convolution":
        b, h, w, c, oc = (config[k] for k in ("batch", "height", "width", "channels", "out_channels"))
        def fn(x, kernel):
            return jax.lax.conv_general_dilated(x, kernel, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
        return fn, (jax.ShapeDtypeStruct((b, h, w, c), d), jax.ShapeDtypeStruct((3, 3, c, oc), d))
    if family in {"mlp", "training"}:
        b, width, depth = (config[k] for k in ("batch", "width", "depth"))
        def fn(x, *weights):
            y = x
            for w in weights[:-1]: y = jnp.tanh(y @ w)
            return y @ weights[-1]
        args = [jax.ShapeDtypeStruct((b, width), d)] + [jax.ShapeDtypeStruct((width, width), d) for _ in range(depth - 1)] + [jax.ShapeDtypeStruct((width, max(1, width // 2)), d)]
        return fn, tuple(args)
    if family == "autodiff":
        b, width, depth = (config[k] for k in ("batch", "width", "depth"))
        def base(x, w):
            y = x @ w
            for _ in range(depth): y = jnp.tanh(y)
            return jnp.mean(y ** 2)
        return jax.value_and_grad(base, argnums=1), (jax.ShapeDtypeStruct((b, width), d), jax.ShapeDtypeStruct((width, width), d))
    if family == "transformer":
        s, width, heads, layers = (config[k] for k in ("sequence", "width", "heads", "layers"))
        hd = width // heads
        def fn(x, *weights):
            y = x
            for layer in range(layers):
                w1, w2 = weights[2 * layer:2 * layer + 2]
                q = y.reshape(1, s, heads, hd)
                scores = jnp.einsum("bshd,bthd->bhst", q, q) / jnp.sqrt(jnp.asarray(hd, dtype=y.dtype))
                attended = jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(scores, axis=-1), q).reshape(1, s, width)
                y = y + attended
                hidden = jnp.tanh(y @ w1)
                y = y + hidden @ w2
            return y
        args = [jax.ShapeDtypeStruct((1, s, width), d)]
        for _ in range(layers): args += [jax.ShapeDtypeStruct((width, 2 * width), d), jax.ShapeDtypeStruct((2 * width, width), d)]
        return fn, tuple(args)
    if family == "matmul":
        m, k, n, depth = (config[k] for k in ("m", "k", "n", "depth"))
        def fn(x, w, *extra):
            y = x @ w
            for square in extra: y = jnp.tanh(y @ square)
            return y
        return fn, tuple([jax.ShapeDtypeStruct((m, k), d), jax.ShapeDtypeStruct((k, n), d)] + [jax.ShapeDtypeStruct((n, n), d) for _ in range(depth - 1)])
    if family == "reduction":
        b, length, width, depth = (config[k] for k in ("batch", "length", "width", "depth"))
        def fn(x):
            y = x
            for _ in range(depth): y = jnp.tanh(y) + jnp.mean(y, axis=-1, keepdims=True)
            return jnp.sum(y, axis=1)
        return fn, (jax.ShapeDtypeStruct((b, length, width), d),)
    if family == "scan":
        b, length, width, depth = (config[k] for k in ("batch", "length", "width", "depth"))
        def fn(x):
            def body(carry, value):
                carry = jnp.tanh(carry + value)
                return carry, carry
            carry = jnp.zeros((b, width), dtype=x.dtype)
            return jax.lax.scan(body, carry, x.transpose(1, 0, 2), length=length)[1]
        return fn, (jax.ShapeDtypeStruct((b, length, width), d),)
    raise ValueError(family)


def static_features(family: str, config: dict[str, Any], dtype: str, freeze: dict[str, Any]) -> dict[str, Any]:
    import jaxoom
    fn, args = candidate_functions(family, config, dtype)
    report = jaxoom.estimate(fn, *args)
    structural = float(report.estimated_peak_bytes)
    peak_buffers = sorted((b.nbytes for b in report.peak.live_buffers), reverse=True)
    largest = float(max((b.nbytes for b in report.largest_buffers), default=0))
    top_two = float(sum(peak_buffers[:2]))
    central_ratio = freeze["calibration_proxy_for_static_candidates"]["central_ratio_median"]
    upper_ratio = freeze["calibration_proxy_for_static_candidates"]["upper_ratio_median"]
    raw = {"structural_peak_bytes": structural, "calibrated_central_bytes": structural * central_ratio, "calibrated_upper_bytes": structural * upper_ratio, "largest_buffer_bytes": largest, "top_two_peak_live_bytes": top_two, "peak_live_bytes": structural, "allocator_limit_bytes": float(BUDGET), "external_pressure_bytes": 0.0, "dtype_bytes": dtype_bytes(dtype)}
    for name in ("structural_peak_bytes", "calibrated_central_bytes", "calibrated_upper_bytes", "largest_buffer_bytes", "top_two_peak_live_bytes", "peak_live_bytes", "external_pressure_bytes"):
        value = raw[name]; raw[name.replace("_bytes", "_over_budget")] = value / BUDGET if BUDGET else None
    nums = [float(v) for v in config.values() if isinstance(v, (int, float))]
    raw.update({"config_numeric_count": float(len(nums)), "config_numeric_max": max(nums, default=0.0), "config_numeric_log_product": math.log1p(math.prod(max(1.0, v) for v in nums)) if nums else 0.0})
    raw["log_structural"] = math.log1p(structural); raw["log_largest"] = math.log1p(largest); raw["log_top_two"] = math.log1p(top_two)
    return raw


def score(features: dict[str, Any], freeze: dict[str, Any]) -> float:
    model = freeze["model"]; values=[]
    for name in model["feature_names"]:
        values.append(float(features.get(name, 0.0)))
    x = np.array([(v - m) / s if s else 0.0 for v, m, s in zip(values, model["means"], model["scales"])], dtype=float)
    z = float(model["intercept"] + np.dot(np.array(model["coefficients"][1:], dtype=float), x))
    return float(1 / (1 + math.exp(-max(-40, min(40, z)))))


def existing_groups(dataset: dict[str, Any]) -> set[str]:
    return {r["workload_group_id"] for r in dataset["rows"]}


def candidate_specs(size: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed); specs=[]
    families = ["attention", "convolution", "mlp", "training", "autodiff", "transformer", "matmul", "reduction", "scan"]
    for i in range(size):
        family = families[i % len(families)]; dtype = "float16" if rng.random() < .45 else "float32"
        if family == "attention": config={"batch":int(rng.choice([1,2,4,8])),"heads":int(rng.choice([2,4,6,8,12,16,24,32])),"sequence":int(rng.choice(np.arange(384,8193,128))),"head_dim":int(rng.choice([32,48,64,80,96,128,160]))}
        elif family == "convolution": config={"batch":int(rng.choice([1,2,4,8,16,32,64,128,256,512,1024])),"height":int(rng.choice([32,48,64,80,96,128,160,192,224])),"width":int(rng.choice([32,48,64,80,96,128,160,192,224])),"channels":int(rng.choice([1,3,8,16,24,32,48,64])),"out_channels":int(rng.choice([8,16,24,32,48,64,96,128]))}
        elif family in {"mlp","training","autodiff"}: config={"batch":int(rng.choice([8,16,32,64,128,256,512,1024,2048])),"width":int(rng.choice([128,256,384,512,768,1024,1536,2048,3072,4096,6144,8192])),"depth":int(rng.choice([1,2,3,4,6,8,12]))}
        elif family == "transformer":
            width=int(rng.choice([256,384,512,768,1024,1536,2048])); heads=int(rng.choice([2,4,8,12,16])); width=width-(width%heads); config={"sequence":int(rng.choice(np.arange(128,4097,128))),"width":width,"heads":heads,"layers":int(rng.choice([1,2,3,4]))}
        elif family == "matmul": config={"m":int(rng.choice([128,256,512,768,1024,1536,2048,3072,4096])),"k":int(rng.choice([128,256,512,768,1024,1536,2048,3072,4096])),"n":int(rng.choice([128,256,512,768,1024,1536,2048,3072,4096])),"depth":int(rng.choice([1,2,3]))}
        elif family == "reduction": config={"batch":int(rng.choice([1,2,4,8,16,32,64,128])),"length":int(rng.choice([128,256,512,1024,2048,4096,8192])),"width":int(rng.choice([32,64,128,256,512,1024,2048])),"depth":int(rng.choice([1,2,4,8]))}
        else: config={"batch":int(rng.choice([1,2,4,8,16,32])),"length":int(rng.choice([128,256,512,1024,2048,4096])),"width":int(rng.choice([32,64,128,256,512,1024])),"depth":int(rng.choice([1,2,4,8]))}
        specs.append({"candidate_id":f"V2-CAND-{i:05d}","family":family,"configuration":config,"dtype":dtype})
    return specs


def build_pool(size: int, seed: int, freeze: dict[str, Any], dataset: dict[str, Any]) -> list[dict[str, Any]]:
    seen=existing_groups(dataset); pool=[]
    for spec in candidate_specs(size, seed):
        group=group_id(spec["family"],spec["configuration"],spec["dtype"])
        if group in seen: continue
        try: features=static_features(spec["family"],spec["configuration"],spec["dtype"],freeze)
        except Exception as exc:
            spec.update({"status":"STATIC_ERROR","error":f"{type(exc).__name__}: {exc}"}); pool.append(spec); continue
        p=score(features,freeze); aggregate=features["calibrated_upper_over_budget"]
        spec.update({"status":"STATIC_OK","workload_group_id":group,"graph_fingerprint":graph_fingerprint(spec["family"],spec["configuration"],spec["dtype"]),"shape_summary":spec["configuration"],"v1_features":features,"v1_predicted_probability":p,"aggregate_score":aggregate,"aggregate_fit":aggregate<=1.0,"candidate_selection_excluded":False})
        pool.append(spec); seen.add(group)
    return pool


def bucket(row: dict[str, Any]) -> tuple[str, str]:
    p=row["v1_predicted_probability"]; agg=row["aggregate_score"]
    if .3 <= p <= .7: return "MODEL_UNCERTAINTY", "V1 probability lies in [0.3, 0.7]"
    if p >= .7 and agg <= 1: return "HIGH_RISK_AGGREGATE_FIT", "V1 high risk while calibrated aggregate says FIT"
    if p <= .3 and .75 <= agg <= 1.1: return "LOW_RISK_NEAR_BOUNDARY", "V1 low risk but aggregate ratio is near the budget boundary"
    if (p >= .7) != (agg > 1): return "MODEL_DISAGREEMENT", "V1 logistic and aggregate gate disagree"
    return ("HIGH_RISK" if p > .7 else "LOW_RISK"), "risk coverage bucket"


def select(pool: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng=np.random.default_rng(seed); valid=[r for r in pool if r.get("status")=="STATIC_OK"]
    for r in valid:
        r["selection_bucket"],r["selection_reason"]=bucket(r)
        p=r["v1_predicted_probability"]
        r["selection_coarse_bucket"]="MODEL_UNCERTAINTY" if .3 <= p <= .7 else ("HIGH_RISK" if p > .7 else "LOW_RISK")
    family_counts=Counter(); selected=[]
    targets={"LOW_RISK":round(count*.30),"MODEL_UNCERTAINTY":round(count*.45),"HIGH_RISK":count-round(count*.30)-round(count*.45)}
    # Within each coarse bucket, prefer informative disagreement/near-boundary
    # cases, then randomize. No outcome or runtime field participates.
    preference={"LOW_RISK":{"LOW_RISK_NEAR_BOUNDARY":0,"MODEL_DISAGREEMENT":1},"HIGH_RISK":{"HIGH_RISK_AGGREGATE_FIT":0,"MODEL_DISAGREEMENT":1}}
    for coarse in ("LOW_RISK","MODEL_UNCERTAINTY","HIGH_RISK"):
        candidates=[r for r in valid if r["selection_coarse_bucket"]==coarse]
        rng.shuffle(candidates); candidates.sort(key=lambda r:preference.get(coarse,{}).get(r["selection_bucket"],2))
        for row in candidates:
            if len(selected)>=count or sum(x["selection_coarse_bucket"]==coarse for x in selected)>=targets[coarse]: break
            if family_counts[row["family"]] >= max(1,count//2): continue
            selected.append(row); family_counts[row["family"]]+=1
    remaining=[r for r in valid if r not in selected]; rng.shuffle(remaining)
    for row in remaining:
        if len(selected)>=count: break
        if family_counts[row["family"]] >= count//2: continue
        selected.append(row); family_counts[row["family"]]+=1
    # Require a small floor for every generated family when the pool supports it.
    families=sorted({r["family"] for r in valid})
    for family in families:
        while family_counts[family] < 3:
            replacement=next((r for r in valid if r not in selected and r["family"]==family),None)
            donor=next((r for r in reversed(selected) if family_counts[r["family"]]>3 and r["selection_coarse_bucket"]==replacement["selection_coarse_bucket"]),None)
            if donor is None: donor=next((r for r in reversed(selected) if family_counts[r["family"]]>3),None)
            if replacement is None or donor is None: break
            selected.remove(donor); family_counts[donor["family"]]-=1; selected.append(replacement); family_counts[family]+=1
    for row in selected: row["selected_for_runtime"] = True
    return selected


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--pool-size",type=int,default=2500); ap.add_argument("--selected-count",type=int,default=60); ap.add_argument("--seed",type=int,default=20260912); ap.add_argument("--pool-output",type=Path); ap.add_argument("--plan-output",type=Path); ap.add_argument("--freeze-output",type=Path); a=ap.parse_args()
    freeze=freeze_v1(); dataset=json.loads(V1_DATASET_PATH.read_text());
    if a.freeze_output and a.freeze_output != FREEZE_PATH: a.freeze_output.write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n")
    pool=build_pool(a.pool_size,a.seed,freeze,dataset); selected=select(pool,a.selected_count,a.seed+1)
    environment={"python_version":platform.python_version(),"jax_version":jax.__version__,"jaxlib_version":jax.lib.__version__,"backend":jax.default_backend(),"devices":[str(d) for d in jax.devices()],"allocator_environment":{k:os.environ.get(k) for k in ("XLA_PYTHON_CLIENT_PREALLOCATE","XLA_CLIENT_MEM_FRACTION","XLA_PYTHON_CLIENT_ALLOCATOR")},"static_only":True}
    plan={"status":"FROZEN_BEFORE_RUNTIME","selection_date":"2026-09-12","seed":a.seed,"v1_model_hash":freeze["model_hash"],"v1_dataset_hash":freeze["dataset_hash"],"candidate_pool_size":len(pool),"static_error_count":sum(r.get("status")!="STATIC_OK" for r in pool),"selected_count":len(selected),"environment":environment,"selection_distribution":dict(Counter(r["selection_bucket"] for r in selected)),"coarse_selection_distribution":dict(Counter(r["selection_coarse_bucket"] for r in selected)),"family_distribution":dict(Counter(r["family"] for r in selected)),"selected_candidates":[{k:r[k] for k in ("candidate_id","workload_group_id","family","configuration","dtype","graph_fingerprint","shape_summary","v1_predicted_probability","aggregate_score","selection_bucket","selection_coarse_bucket","selection_reason")} for r in selected],"outcomes_excluded":True,"no_capacity_sweeps":True}
    if a.pool_output:a.pool_output.write_text(json.dumps({"status":"STATIC_CANDIDATE_POOL","seed":a.seed,"v1_model_hash":freeze["model_hash"],"candidates":pool},indent=2,sort_keys=True)+"\n")
    if a.plan_output:a.plan_output.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
    print(json.dumps(plan,indent=2,sort_keys=True))

if __name__ == "__main__": main()
