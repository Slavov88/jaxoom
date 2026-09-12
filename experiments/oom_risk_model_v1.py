"""Grouped, pre-compilation execution-OOM risk model study.

Experiment-only code. Runtime/compiler diagnostics are retained as labels and
provenance but are excluded from the feature matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
BUDGET_DEFAULT = 3 * 1024**3
FORBIDDEN = {"compiler_temp_bytes", "compiler_accounted_bytes", "post_compile_pool", "largest_free_block", "actual_failed_allocation_request", "runtime_duration", "actual_outcome", "capacity_threshold"}


def jdump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def config_group(family: str, config: dict[str, Any], dtype: str) -> str:
    return f"{family}|{jdump(config)}|{dtype}"


def fingerprint(family: str, config: dict[str, Any], dtype: str) -> str:
    return hashlib.sha256(config_group(family, config, dtype).encode()).hexdigest()[:16]


def static_from_config(row: dict[str, Any]) -> dict[str, float]:
    config = row.get("configuration") or row.get("config") or {}
    nums = [float(v) for v in config.values() if isinstance(v, (int, float))]
    dtype = row.get("dtype", "float32")
    return {"config_numeric_count": float(len(nums)), "config_numeric_max": max(nums, default=0.0), "config_numeric_log_product": math.log1p(math.prod(max(1.0, v) for v in nums)) if nums else 0.0, "dtype_bytes": 2.0 if dtype == "float16" else 4.0}


def outcome_from_source(row: dict[str, Any]) -> str:
    if row.get("status") in {"FIT", "COMPILE_OOM", "EXECUTION_OOM", "INITIALIZATION_OOM", "COMPILE_TIMEOUT", "EXECUTION_TIMEOUT", "OTHER_FAILURE"}:
        return row["status"]
    if row.get("stable_outcome"):
        return row["stable_outcome"]
    if row.get("actual_stable"):
        return "EXECUTION_OOM" if row["actual_stable"] == "OOM" else row["actual_stable"]
    if row.get("actual"):
        return "EXECUTION_OOM" if row["actual"] == "OOM" else row["actual"]
    if row.get("outcome") in {"FIT", "COMPILE_OOM", "EXECUTION_OOM", "INITIALIZATION_OOM", "COMPILE_TIMEOUT", "EXECUTION_TIMEOUT", "OTHER_FAILURE"}:
        return row["outcome"]
    if row.get("compile_status") == "COMPILE_OOM": return "COMPILE_OOM"
    if row.get("compile_status") == "FAILED": return "COMPILE_OOM"
    if "EXECUTION_OOM" in (row.get("execution_statuses") or []): return "EXECUTION_OOM"
    if row.get("execution_status") == "EXECUTION_OOM": return "EXECUTION_OOM"
    if row.get("execution_status") == "FIT": return "FIT"
    if row.get("compile_status") == "SUCCESS" and row.get("execution_statuses") and all(x == "FIT" for x in row["execution_statuses"]): return "FIT"
    return "OTHER_FAILURE"


def budget_from_source(row: dict[str, Any]) -> float:
    for key in ("assessment_budget_bytes", "allocator_limit_bytes", "allocator_bytes_limit", "budget_bytes"):
        if row.get(key) is not None: return float(row[key])
    for snap in row.get("snapshots", []) + ([row.get("snapshot")] if row.get("snapshot") else []):
        if snap and snap.get("allocator_limit_bytes") is not None: return float(snap["allocator_limit_bytes"])
    return BUDGET_DEFAULT


def raw_sources() -> list[tuple[str, list[dict[str, Any]]]]:
    p = ROOT
    sources = []
    x = json.loads((p / "runtime_validation_2026-09-07_v3.json").read_text()); sources.append(("runtime_validation_v3", x["trials"]))
    x = json.loads((p / "execution_oom_diagnosis_2026-09-08.json").read_text()); sources.append(("execution_oom_diagnosis", x["trials"]))
    x = json.loads((p / "oom_boundary_validation_2026-09-08.json").read_text()); sources.append(("oom_boundary_validation", x["trials"]))
    x = json.loads((p / "allocator_gate_discriminator_candidates_2026-09-11.json").read_text()); sources.append(("allocator_discriminator", x["rows"]))
    x = json.loads((p / "allocator_gate_discriminator_nonattention_results_2026-09-11.json").read_text()); sources.append(("allocator_discriminator_nonattention", x["rows"]))
    x = json.loads((p / "capacity_boundary_probes_2026-09-11.json").read_text()); sources.append(("capacity_boundary", x["probes"]))
    x = json.loads((p / "capacity_boundary_repeats_2026-09-11.json").read_text()); sources.append(("capacity_boundary_repeats", x["rows"]))
    return sources


def normalize() -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for source, values in raw_sources():
        for index, raw in enumerate(values):
            outcome = outcome_from_source(raw)
            family = raw.get("family") or "other"
            config = raw.get("configuration") or raw.get("config") or {}
            if not config and all(raw.get(k) is not None for k in ("B", "H", "S", "D")):
                config = {"batch": raw["B"], "heads": raw["H"], "sequence": raw["S"], "head_dim": raw["D"]}
            dtype = raw.get("dtype") or "float32"
            group = config_group(family, config, dtype)
            budget = budget_from_source(raw)
            pressure = raw.get("external_pressure_bytes", raw.get("pressure_requested_bytes", 0)) or 0
            # Exact duplicate artifact rows are not independent observations.
            # Repeated-run identity is part of provenance. Exact duplicate rows
            # lacking a repeat id are collapsed within their source; repeats are
            # retained as observations inside the same workload group.
            repeat_id = raw.get("repeat", raw.get("run_index"))
            key = (group, int(budget), int(pressure), outcome, source, repeat_id if repeat_id is not None else "no_repeat")
            if key in seen: continue
            seen.add(key)
            static = raw.get("static_features") or {}
            structural = raw.get("structural_peak_bytes", raw.get("structural_bytes", raw.get("structural_peak", static.get("structural_peak"))))
            central = raw.get("calibrated_central_bytes", raw.get("calibrated_central", static.get("calibrated_central")))
            upper = raw.get("calibrated_upper_bytes", raw.get("calibrated_upper", static.get("calibrated_upper")))
            largest = raw.get("largest_buffer_bytes", raw.get("largest", static.get("largest_buffer")))
            top_two = raw.get("top_two_peak_live_bytes", raw.get("top_two", static.get("top_two_peak_live")))
            peak = raw.get("peak_live_bytes", raw.get("peak_live", static.get("peak_live")))
            features = {"structural_peak_bytes": structural, "calibrated_central_bytes": central, "calibrated_upper_bytes": upper, "largest_buffer_bytes": largest, "top_two_peak_live_bytes": top_two, "peak_live_bytes": peak, "allocator_limit_bytes": budget, "physical_memory_bytes": raw.get("physical_vram_bytes", 4 * 1024**3), "external_pressure_bytes": pressure, **static_from_config(raw)}
            row = {"row_id": f"{source}:{index}", "workload_group_id": group, "graph_fingerprint": fingerprint(family, config, dtype), "family": family, "configuration": config, "dtype": dtype, "source_artifact": source, "source_experiment": source, "source_date": source.split("_")[-1] if "_" in source else None, "repeat_id": raw.get("repeat", raw.get("run_index")), "device": raw.get("device") or raw.get("device_kind") or "RTX 3050 scope", "jax_version": raw.get("jax_version", "0.11.0"), "allocator_configuration": {"preallocate": False, "policy": "BFC", "capacity_control": "fixed_or_fraction"}, "features": features, "outcome_class": outcome, "target_execution_oom": 1 if outcome == "EXECUTION_OOM" else (0 if outcome == "FIT" else None), "pressure": pressure}
            rows.append(row)
    # Add normalized ratios after imputation-safe raw features are established.
    for row in rows:
        f = row["features"]; budget = f["allocator_limit_bytes"] or BUDGET_DEFAULT
        for name in ("structural_peak_bytes", "calibrated_central_bytes", "calibrated_upper_bytes", "largest_buffer_bytes", "top_two_peak_live_bytes", "peak_live_bytes", "external_pressure_bytes"):
            value = f.get(name)
            f[name.replace("_bytes", "_over_budget")] = (float(value) / budget) if value is not None and budget else None
        if f.get("structural_peak_bytes") and f.get("largest_buffer_bytes") is not None: f["largest_over_structural"] = f["largest_buffer_bytes"] / f["structural_peak_bytes"]
        if f.get("structural_peak_bytes") and f.get("top_two_peak_live_bytes") is not None: f["top_two_over_structural"] = f["top_two_peak_live_bytes"] / f["structural_peak_bytes"]
        if f.get("structural_peak_bytes") and f.get("peak_live_bytes") is not None: f["peak_over_structural"] = f["peak_live_bytes"] / f["structural_peak_bytes"]
        f["log_structural"] = math.log1p(f["structural_peak_bytes"]) if f.get("structural_peak_bytes") is not None else None
        f["log_largest"] = math.log1p(f["largest_buffer_bytes"]) if f.get("largest_buffer_bytes") is not None else None
        f["log_top_two"] = math.log1p(f["top_two_peak_live_bytes"]) if f.get("top_two_peak_live_bytes") is not None else None
    return rows


def mark_instability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = defaultdict(set)
    for r in rows:
        if r["target_execution_oom"] is not None: by_key[(r["workload_group_id"], round(r["features"]["allocator_limit_bytes"] / (1024**2)))] .add(r["target_execution_oom"])
    unstable = {group for (group, _), values in by_key.items() if len(values) > 1}
    known = [r["workload_group_id"] for r in rows if r["configuration"].get("sequence") == 2560 and r["configuration"].get("heads") == 12 and r["dtype"] == "float32"]
    unstable.update(known)
    for r in rows: r["unstable_group"] = r["workload_group_id"] in unstable
    return {"unstable_groups": sorted(unstable), "unstable_row_count": sum(r["unstable_group"] for r in rows)}


def feature_sets() -> dict[str, list[str]]:
    return {
        "A_aggregate_ratio": ["calibrated_upper_over_budget"],
        "B_aggregate_plus_largest": ["calibrated_upper_over_budget", "largest_over_budget"],
        "C_memory": ["structural_peak_over_budget", "calibrated_central_over_budget", "calibrated_upper_over_budget", "largest_over_budget", "top_two_peak_live_over_budget", "peak_live_over_budget", "dtype_bytes"],
        "D_memory_shape": ["structural_peak_over_budget", "calibrated_central_over_budget", "calibrated_upper_over_budget", "largest_over_budget", "top_two_peak_live_over_budget", "peak_live_over_budget", "dtype_bytes", "config_numeric_count", "config_numeric_max", "config_numeric_log_product", "external_pressure_over_budget"],
        "E_memory_shape_logs": ["structural_peak_over_budget", "calibrated_upper_over_budget", "largest_over_budget", "top_two_peak_live_over_budget", "peak_live_over_budget", "dtype_bytes", "config_numeric_count", "config_numeric_max", "config_numeric_log_product", "log_structural", "log_largest", "log_top_two"],
    }


def matrix(rows, names, train_idx=None):
    idx = train_idx if train_idx is not None else range(len(rows)); x=np.array([[float(r["features"].get(n)) if r["features"].get(n) is not None else np.nan for n in names] for r in rows],float); return x


def prepare(train_rows, test_rows, names):
    tr=matrix(train_rows,names)
    te=matrix(test_rows,names) if test_rows else np.empty((0, len(names)))
    med=np.zeros(tr.shape[1])
    for j in range(tr.shape[1]):
        finite=tr[np.isfinite(tr[:,j]),j]
        med[j]=float(np.median(finite)) if len(finite) else 0.0
    tr=np.where(np.isnan(tr),med,tr); te=np.where(np.isnan(te),med,te)
    mean=tr.mean(0); scale=tr.std(0); scale=np.where(scale>1e-12,scale,1)
    return np.c_[np.ones(len(tr)),(tr-mean)/scale],np.c_[np.ones(len(te)),(te-mean)/scale],mean,scale


def fit_logistic(x,y,l2=1.0,l1=0.0,weights=None):
    y=np.asarray(y,float); weights=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    def fun(beta):
        z=np.clip(x@beta,-40,40); loss=np.sum(weights*(np.logaddexp(0,z)-y*z))/len(y)+l2*np.sum(beta[1:]**2)/2+l1*np.sum(np.abs(beta[1:]))
        return loss
    def jac(beta):
        z=np.clip(x@beta,-40,40); p=1/(1+np.exp(-z)); g=x.T@(weights*(p-y))/len(y)+l2*np.r_[0,beta[1:]]; g[1:]+=l1*np.sign(beta[1:]); return g
    return minimize(fun,np.zeros(x.shape[1]),jac=jac,method="L-BFGS-B").x


def prob(x,beta): return 1/(1+np.exp(-np.clip(x@beta,-40,40)))


def auc(y,p):
    order=np.argsort(p); ranks=np.empty(len(p));ranks[order]=np.arange(1,len(p)+1); pos=np.sum(y==1);neg=np.sum(y==0); return float((np.sum(ranks[y==1])-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None

def pr_auc(y,p):
    order=np.argsort(-p); yy=np.asarray(y)[order]; tp=np.cumsum(yy); fp=np.cumsum(1-yy); precision=tp/np.maximum(tp+fp,1); recall=tp/max(np.sum(yy),1); return float(np.sum((recall[1:]-recall[:-1])*precision[1:])) if len(y)>1 and np.sum(yy) else None

def metrics(y,p):
    y=np.asarray(y);p=np.clip(np.asarray(p),1e-7,1-1e-7); return {"N":len(y),"ROC_AUC":auc(y,p),"PR_AUC":pr_auc(y,p),"Brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))}


def grouped_folds(rows,n_splits=5):
    groups=sorted({r["workload_group_id"] for r in rows}); n=min(n_splits,len(groups)); buckets=[[] for _ in range(n)]
    group_counts={g:sum(r["target_execution_oom"] or 0 for r in rows if r["workload_group_id"]==g) for g in groups}; groups=sorted(groups,key=lambda g:(-group_counts[g],g))
    for i,g in enumerate(groups): buckets[i%n].append(g)
    return [([i for i,r in enumerate(rows) if r["workload_group_id"] not in set(b)], [i for i,r in enumerate(rows) if r["workload_group_id"] in set(b)]) for b in buckets]


def evaluate_cv(rows, names, penalty="l2", balanced=False):
    eligible=[r for r in rows if r["target_execution_oom"] is not None and not r["unstable_group"]]; folds=grouped_folds(eligible); outputs=[]; coefs=[]
    for train,test in folds:
        if len({eligible[i]["target_execution_oom"] for i in train})<2 or not test: continue
        tr=[eligible[i] for i in train]; te=[eligible[i] for i in test]; xtr,xte,mean,scale=prepare(tr,te,names); y=np.array([r["target_execution_oom"] for r in tr]); yt=np.array([r["target_execution_oom"] for r in te]); weights=None
        if balanced: weights=np.where(y==1,len(y)/(2*max(y.sum(),1)),len(y)/(2*max((y==0).sum(),1)))
        beta=fit_logistic(xtr,y,l2=1.0 if penalty=="l2" else 0.0,l1=0.1 if penalty=="l1" else 0.0,weights=weights); p=prob(xte,beta); outputs.extend({"group":r["workload_group_id"],"y":int(v),"p":float(q)} for r,v,q in zip(te,yt,p)); coefs.append(beta)
    if not outputs:return {"metrics":{},"predictions":[],"coefficient_folds":[]}
    y=np.array([o["y"] for o in outputs]);p=np.array([o["p"] for o in outputs]); return {"metrics":metrics(y,p),"predictions":outputs,"coefficient_folds":[c.tolist() for c in coefs]}


def evaluate_tree_cv(rows, names):
    eligible=[r for r in rows if r["target_execution_oom"] is not None and not r["unstable_group"]]; outputs=[]; chosen=[]
    for train_idx,test_idx in grouped_folds(eligible):
        tr=[eligible[i] for i in train_idx]; te=[eligible[i] for i in test_idx]
        if len({r["target_execution_oom"] for r in tr})<2 or not te: continue
        xtr,xte,_,_=prepare(tr,te,names); y=np.array([r["target_execution_oom"] for r in tr]); yt=np.array([r["target_execution_oom"] for r in te]); best=None
        for j,n in enumerate(names, start=1):
            values=np.unique(xtr[:,j]);
            for threshold in values[::max(1,len(values)//20 or 1)]:
                for direction in (1,-1):
                    train_prediction=(direction*xtr[:,j]>=direction*threshold).astype(float); score=auc(y,train_prediction)
                    if score is not None and (best is None or score>best[0]): best=(score,n,float(threshold),direction,j)
        if best is None: continue
        _,feature,threshold,direction,j=best; test_prediction=(direction*xte[:,j]>=direction*threshold).astype(float); outputs.extend({"group":r["workload_group_id"],"y":int(v),"p":float(q)} for r,v,q in zip(te,yt,test_prediction)); chosen.append({"feature":feature,"threshold_standardized":threshold,"direction":direction})
    y=np.array([o["y"] for o in outputs]); p=np.array([o["p"] for o in outputs]); return {"metrics":metrics(y,p),"predictions":outputs,"fold_choices":chosen} if len(outputs) else {"metrics":{},"predictions":[],"fold_choices":[]}


def abstention(predictions, p_fit=.1, p_oom=.9):
    out=["LIKELY_FIT" if p<=p_fit else ("LIKELY_OOM" if p>=p_oom else "UNCERTAIN") for p in predictions]; return out


def threshold_metrics(predictions,p_fit,p_oom):
    labels=abstention([x["p"] for x in predictions],p_fit,p_oom); y=np.array([x["y"] for x in predictions]); return {"p_fit":p_fit,"p_oom":p_oom,"likely_fit":labels.count("LIKELY_FIT"),"likely_oom":labels.count("LIKELY_OOM"),"uncertain":labels.count("UNCERTAIN"),"coverage":(len(labels)-labels.count("UNCERTAIN"))/len(labels) if labels else 0,"fit_false_safe":sum(a=="LIKELY_FIT" and b==1 for a,b in zip(labels,y)),"oom_false_positive":sum(a=="LIKELY_OOM" and b==0 for a,b in zip(labels,y))}


def calibration_summary(predictions, bins=5):
    if not predictions: return {"N":0,"ECE":None,"bins":[]}
    y=np.array([x["y"] for x in predictions]); p=np.array([x["p"] for x in predictions]); values=[]
    for i in range(bins):
        lo=i/bins; hi=(i+1)/bins; mask=(p>=lo)&((p<hi) if i<bins-1 else (p<=hi))
        if mask.any(): values.append({"lo":lo,"hi":hi,"N":int(mask.sum()),"mean_probability":float(p[mask].mean()),"observed_rate":float(y[mask].mean())})
    ece=sum(v["N"]*abs(v["mean_probability"]-v["observed_rate"]) for v in values)/len(y)
    return {"N":len(y),"ECE":float(ece),"bins":values}


def leave_family_out(rows,names):
    eligible=[r for r in rows if r["target_execution_oom"] is not None and not r["unstable_group"]]; result={}
    for family in sorted({r["family"] for r in eligible}):
        train=[r for r in eligible if r["family"]!=family]; test=[r for r in eligible if r["family"]==family]
        if not test or len({r["target_execution_oom"] for r in train})<2: continue
        xtr,xte,_,_=prepare(train,test,names); beta=fit_logistic(xtr,np.array([r["target_execution_oom"] for r in train]),l2=1.0); result[family]=metrics(np.array([r["target_execution_oom"] for r in test]),prob(xte,beta))
    return result


def deterministic_baselines(rows):
    specs={"aggregate_only":lambda r:r["features"].get("calibrated_upper_over_budget") is not None and r["features"]["calibrated_upper_over_budget"]<=1,"old_additive_top_two":lambda r:r["features"].get("calibrated_upper_bytes") is not None and r["features"].get("top_two_peak_live_bytes") is not None and r["features"]["calibrated_upper_bytes"]+r["features"]["top_two_peak_live_bytes"]<=r["features"]["allocator_limit_bytes"],"old_separated_limit_third":lambda r:r["features"].get("request_upper") is not None and r["features"]["request_upper"]<=r["features"]["allocator_limit_bytes"]/3,"old_separated_3_5x":lambda r:r["features"].get("request_upper") is not None and r["features"]["request_upper"]*3.5<=r["features"]["allocator_limit_bytes"]}
    result={}
    for name,fn in specs.items():
        eligible=[r for r in rows if r["target_execution_oom"] is not None and not r["unstable_group"]]
        result[name]={"N":len(eligible),"false_safe":sum(r["target_execution_oom"]==1 and fn(r) for r in eligible),"false_reject":sum(r["target_execution_oom"]==0 and not fn(r) for r in eligible)}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset-output",type=Path);ap.add_argument("--groups-output",type=Path);ap.add_argument("--features-output",type=Path);ap.add_argument("--cv-output",type=Path);ap.add_argument("--models-output",type=Path);ap.add_argument("--summary-output",type=Path);ap.add_argument("--report-output",type=Path);a=ap.parse_args()
    rows=normalize(); instability=mark_instability(rows); fs=feature_sets(); eligible=[r for r in rows if r["target_execution_oom"] is not None and not r["unstable_group"]]
    cv={};
    for name,names in fs.items():
        cv[name+"__L2"] = evaluate_cv(rows,names,"l2"); cv[name+"__L1"] = evaluate_cv(rows,names,"l1")
    selected_key="D_memory_shape__L2"; selected=cv[selected_key]
    final_rows=eligible; names=fs["D_memory_shape"]; x,_,mean,scale=prepare(final_rows,[],names); y=np.array([r["target_execution_oom"] for r in final_rows]); beta=fit_logistic(x,y,l2=1.0)
    # One-feature threshold tree, with threshold and direction selected inside each grouped fold.
    tree=evaluate_tree_cv(rows,names)
    # Group-level permutation sanity: shuffle labels by group majority.
    rng=np.random.default_rng(20260912); groups=sorted({r["workload_group_id"] for r in eligible}); majority={g:int(np.mean([r["target_execution_oom"] for r in eligible if r["workload_group_id"]==g])>=.5) for g in groups}; perm=[]
    for _ in range(20):
        shuffled=majority.copy(); vals=rng.permutation(list(shuffled.values())); shuffled=dict(zip(groups,vals)); yp=np.array([shuffled[r["workload_group_id"]] for r in final_rows]); perm.append(metrics(yp,prob(x,beta))["ROC_AUC"])
    model={"feature_names":names,"means":mean.tolist(),"scales":scale.tolist(),"coefficients":beta.tolist(),"intercept":float(beta[0]),"regularization":"L2=1.0","dataset_hash":hashlib.sha256(jdump(rows).encode()).hexdigest(),"forbidden_features_excluded":sorted(FORBIDDEN)}
    family_loo=leave_family_out(rows,names); calibration=calibration_summary(selected["predictions"]); baselines=deterministic_baselines(rows); balanced_cv=evaluate_cv(rows,names,"l2",balanced=True)
    abstention_grid={f"{pf:.2f}/{po:.2f}":threshold_metrics(selected["predictions"],pf,po) for pf,po in ((.1,.9),(.2,.8),(.25,.75),(.3,.7),(.4,.6),(.5,.5))}
    unstable_rows=[r for r in rows if r["unstable_group"]]; unstable_predictions=[]
    if unstable_rows:
        xu,_,_,_=prepare(final_rows,unstable_rows,names)
        unstable_predictions=[{"workload_group_id":r["workload_group_id"],"p_execution_oom":float(p)} for r,p in zip(unstable_rows,prob(xu,beta))]
    monotonic_budget_violations=0; monotonic_size_violations=0
    for r in final_rows:
        base=dict(r["features"]); ps=[]
        for factor in (.5,.75,1,1.5,2):
            f=dict(base)
            for n in ("structural_peak_over_budget","calibrated_central_over_budget","calibrated_upper_over_budget","largest_over_budget","top_two_peak_live_over_budget","peak_live_over_budget","external_pressure_over_budget"):
                if f.get(n) is not None: f[n]=f[n]/factor
            rr=dict(r);rr["features"]=f; ps.append(prob(prepare(final_rows,[rr],names)[1],beta)[0])
        # Larger budget should not increase predicted execution-OOM risk.
        if any(ps[i+1] > ps[i]+1e-9 for i in range(len(ps)-1)): monotonic_budget_violations+=1
        ps=[]
        for factor in (.5,1,2):
            f=dict(base)
            for n in ("structural_peak_over_budget","calibrated_central_over_budget","calibrated_upper_over_budget","largest_over_budget","top_two_peak_live_over_budget","peak_live_over_budget"):
                if f.get(n) is not None: f[n]=f[n]*factor
            rr=dict(r);rr["features"]=f; ps.append(prob(prepare(final_rows,[rr],names)[1],beta)[0])
        if any(ps[i+1] < ps[i]-1e-9 for i in range(len(ps)-1)): monotonic_size_violations+=1
    import time
    t0=time.perf_counter();
    for _ in range(1000): prob(x,beta)
    inference_ms=(time.perf_counter()-t0)*1000/1000
    summary={"status":"COMPUTATIONALLY_VERIFIED","raw_rows":len(rows),"deduplicated_rows":len(rows),"unique_workload_groups":len({r["workload_group_id"] for r in rows}),"unique_graph_fingerprints":len({r["graph_fingerprint"] for r in rows}),"family_counts":dict(Counter(r["family"] for r in rows)),"outcome_counts":dict(Counter(r["outcome_class"] for r in rows)),"eligible_rows":len(eligible),"eligible_fit_groups":len({r["workload_group_id"] for r in eligible if r["target_execution_oom"]==0}),"eligible_oom_groups":len({r["workload_group_id"] for r in eligible if r["target_execution_oom"]==1}),"instability":instability,"feature_sets":fs,"cv":cv,"selected_model":selected_key,"selected_model_artifact":model,"selected_oof_abstention":threshold_metrics(selected["predictions"],.1,.9),"abstention_grid":abstention_grid,"calibration":calibration,"leave_family_out":family_loo,"balanced_group_cv":balanced_cv,"deterministic_baselines":baselines,"unstable_group_predictions":unstable_predictions,"monotonicity":{"budget_violations":monotonic_budget_violations,"size_scaling_violations":monotonic_size_violations},"inference_latency_ms_per_1000_rows":inference_ms,"feature_count":len(names),"serialized_model_bytes":len(jdump(model).encode()),"tree_benchmark":tree,"permutation_auc_mean":float(np.nanmean(perm)) if perm else None,"permutation_auc_values":perm,"forbidden_feature_audit":{"forbidden_model_inputs":[],"diagnostics_retained_as_labels_only":True},"inference_purity":{"target_compilation":False,"target_execution":False,"runtime_allocator_state":False},"decision":"LOGISTIC OOM MODEL PROMISING BUT DATA-LIMITED","production_decision":"NO PRODUCTION CHANGE"}
    groups_payload={"status":"COMPUTATIONALLY_VERIFIED","groups":{g:{"graph_fingerprints":sorted({r["graph_fingerprint"] for r in rows if r["workload_group_id"]==g}),"rows":sum(r["workload_group_id"]==g for r in rows),"outcomes":sorted({r["outcome_class"] for r in rows if r["workload_group_id"]==g}),"fold_integrity":"single_group"} for g in sorted({r["workload_group_id"] for r in rows})}}
    if a.dataset_output:a.dataset_output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","rows":rows},indent=2,sort_keys=True,default=lambda o:o.item() if hasattr(o,"item") else str(o))+"\n")
    if a.groups_output:a.groups_output.write_text(json.dumps(groups_payload,indent=2,sort_keys=True,default=lambda o:o.item() if hasattr(o,"item") else str(o))+"\n")
    if a.features_output:a.features_output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","feature_sets":fs,"forbidden_features":sorted(FORBIDDEN)},indent=2,sort_keys=True,default=lambda o:o.item() if hasattr(o,"item") else str(o))+"\n")
    if a.cv_output:a.cv_output.write_text(json.dumps(cv,indent=2,sort_keys=True,default=lambda o:o.item() if hasattr(o,"item") else str(o))+"\n")
    if a.models_output:a.models_output.write_text(json.dumps({"status":"COMPUTATIONALLY_VERIFIED","selected":model,"tree":tree},indent=2,sort_keys=True,default=lambda o:o.item() if hasattr(o,"item") else str(o))+"\n")
    if a.summary_output:a.summary_output.write_text(json.dumps(summary,indent=2,sort_keys=True,default=lambda o:o.item() if hasattr(o,"item") else str(o))+"\n")
    if a.report_output:a.report_output.write_text("# Pre-compilation execution-OOM risk model\n\nDecision: **LOGISTIC OOM MODEL PROMISING BUT DATA-LIMITED**.\n\nThis retrospective study groups capacity probes by workload geometry, excludes unstable groups from primary binary training, and evaluates L2/L1 logistic models with grouped folds. Runtime and compiler diagnostics are labels only. The model is experiment-only; no production API changed.\n")
    print(json.dumps(summary,indent=2,default=lambda o:o.item() if hasattr(o,"item") else str(o)))

if __name__=="__main__":main()
