"""Fit and score tiny interpretable capacity-boundary hypotheses."""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path


def load():
    p=Path(__file__).resolve().parent
    return json.loads((p/"capacity_boundary_results_2026-09-11.json").read_text())["workloads"]


def bracket_rows():
    return [r for r in load().values() if r["status"] == "BRACKETED"]


def required_upper(r):
    return r["first_fit_capacity"]


def models(dev):
    max_k=max(r["first_fit_capacity"] / r["static_features"]["request_upper"] for r in dev)
    # The fitted k is rounded upward only to the resolution supported by the
    # capacity brackets; no held-out outcome is used.
    k_dev=math.ceil(max_k * 100) / 100
    return {
        "3x_request_baseline": {"formula":"3 * request_upper", "predict":lambda r:3*r["static_features"]["request_upper"], "parameters":{"k":3}},
        "development_upper_envelope": {"formula":f"{k_dev:.2f} * request_upper", "predict":lambda r,k=k_dev:k*r["static_features"]["request_upper"], "parameters":{"k":k_dev}},
        "3.25x_request": {"formula":"3.25 * request_upper", "predict":lambda r:3.25*r["static_features"]["request_upper"], "parameters":{"k":3.25}},
        "request_plus_structural": {"formula":"request_upper + structural_peak", "predict":lambda r:r["static_features"]["request_upper"]+r["static_features"]["structural_peak"], "parameters":{}},
        "request_plus_3x_structural": {"formula":"request_upper + 3 * structural_peak", "predict":lambda r:r["static_features"]["request_upper"]+3*r["static_features"]["structural_peak"], "parameters":{}},
    }


def score(rows, model):
    # A predicted capacity below first_fit is an unsafe underprediction for a
    # conservative safe-capacity rule. Bracket uncertainty is retained.
    unsafe=[r["configuration_id"] for r in rows if model["predict"](r) < r["first_fit_capacity"]]
    over=[r["configuration_id"] for r in rows if model["predict"](r) > r["first_fit_capacity"]]
    errors=[model["predict"](r)-r["first_fit_capacity"] for r in rows]
    return {"N":len(rows),"unsafe_underprediction_count":len(unsafe),"unsafe_rows":unsafe,"overconservative_count":len(over),"max_unsafe_error_bytes":min(errors,default=0),"median_absolute_error_bytes":statistics.median([abs(x) for x in errors]) if errors else None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    rows=bracket_rows(); split=json.loads((Path(__file__).resolve().parent/"capacity_boundary_split_2026-09-11.json").read_text()); dev_ids=set(split["development"]); eval_ids=set(split["evaluation"]); dev=[r for r in rows if r["configuration_id"] in dev_ids]; ev=[r for r in rows if r["configuration_id"] in eval_ids]
    ms=models(dev); table={}
    for name,m in ms.items(): table[name]={"formula":m["formula"],"parameters":m["parameters"],"development":score(dev,m),"held_out":score(ev,m),"overall":score(rows,m)}
    selected="development_upper_envelope"
    ratios=[]
    for r in rows:
        req=r["static_features"]["request_upper"]; ratios.append({"configuration_id":r["configuration_id"],"ratio_lower":req/r["first_fit_capacity"],"ratio_upper":req/r["last_oom_capacity"]})
    out={"status":"COMPUTATIONALLY_VERIFIED","bracketed_N":len(rows),"development_N":len(dev),"held_out_N":len(ev),"models":table,"selected_model":selected,"capacity_ratios":ratios,"limit_over_3_replay":{"rows":[{"configuration_id":r["configuration_id"],"predicted_capacity":3*r["static_features"]["request_upper"],"last_oom_capacity":r["last_oom_capacity"],"first_fit_capacity":r["first_fit_capacity"],"unsafe":3*r["static_features"]["request_upper"]<r["first_fit_capacity"]} for r in rows]},"decision":"CAPACITY THRESHOLDS NOT STABLE ENOUGH"}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
