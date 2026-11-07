"""Construct the outcome-free batch-3 confirmatory panel (see FRESH_PANEL_PROTOCOL).

Plan stage never imports or evaluates model probabilities. Scoring is a separate
stage run only after the plan file is committed. No retraining, no threshold change.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("heldout2_base", ROOT / "oom_risk_heldout2_validation.py")
h2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h2)
base = h2.base
SEED = 20261108
PLAN = ROOT / "oom_risk_heldout3_plan_2026-11-08.json"
PRED = ROOT / "oom_risk_heldout3_predictions_2026-11-08.json"
MANIFEST = ROOT / "oom_risk_heldout3_validation_manifest_2026-11-08.json"
QUOTAS = {
    "attention": {"small": 10, "medium": 10, "large": 10, "near_device": 14},
    "convolution": {"small": 8, "medium": 8, "large": 8, "near_device": 10},
    "mlp": {"small": 4, "medium": 4, "large": 4, "near_device": 3},
    "training": {"small": 4, "medium": 4, "large": 4, "near_device": 3},
    "transformer": {"small": 6, "medium": 6, "large": 6, "near_device": 4},
    "autodiff": {"small": 4, "medium": 5, "large": 4, "near_device": 3},
    "scan": {"small": 3, "medium": 4, "large": 3, "near_device": 2},
    "matmul": {"small": 3, "medium": 3, "large": 3, "near_device": 2},
    "reduction": {"small": 3, "medium": 3, "large": 3, "near_device": 2},
}
def canonical(x): return h2.canonical(x)
def sha(x): return h2.sha(x)
def file_sha(p): return h2.file_sha(p)

def prior_rows():
    rows = list(h2.prior_rows())
    for path in (h2.PLAN, h2.CONVPLAN):
        rows.extend(json.loads(path.read_text())["selected_candidates"])
    return rows
def prior_groups(): return {r["workload_group_id"] for r in prior_rows()}
def prior_fps(): return {r.get("graph_fingerprint") for r in prior_rows() if r.get("graph_fingerprint")}

def audit(selected):
    old = prior_rows(); oldfp = prior_fps(); groups = prior_groups()
    return {"prior_group_overlaps": sum(x["workload_group_id"] in groups for x in selected),
            "prior_exact_fingerprint_overlaps": sum(x["graph_fingerprint"] in oldfp for x in selected),
            "near_duplicate_definition": "same family and dtype with all but one configuration scalar equal",
            "within_panel_near_duplicate_pairs": sum(base.near_duplicate(a, b) for i, a in enumerate(selected) for b in selected[i + 1:]),
            "vs_prior_near_duplicate_pairs": sum(base.near_duplicate(a, b) for a in selected for b in old),
            "prior_rows_checked": len(old)}

def select(pop, seed, used):
    rng = np.random.default_rng(seed); out = []
    for fam, scales in QUOTAS.items():
        for scale, n in scales.items():
            ch = [r for r in pop if r["family"] == fam and r["sampling_stratum"]["scale"] == scale and r["workload_group_id"] not in used]
            rng.shuffle(ch)
            if len(ch) < n: raise RuntimeError(f"insufficient {fam}/{scale}")
            out.extend(ch[:n]); used.update(x["workload_group_id"] for x in ch[:n])
    for i, r in enumerate(out): r["candidate_id"] = f"H3-{i:04d}"
    return out

def manifest():
    models = {}
    for name, path in (("v2", h2.V2), ("v3", h2.V3)):
        x = json.loads(path.read_text())
        models[name] = {"model_hash": x["model_hash"], "artifact_sha256": file_sha(path)}
    x = {"status": "FROZEN_BEFORE_BATCH3_PANEL", "date": "2026-11-08", "primary_model": "V2",
         "models": models, "thresholds": {"primary": .20}, "seed": SEED, "no_retraining": True,
         "selection_independent_of_model_predictions": True, "compile_filter_used": False,
         "kill_criterion": "any LIKELY_FIT recipe that is execution or compile OOM fails the false-safe claim"}
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != x: raise RuntimeError("refusing to overwrite frozen manifest")
    if not MANIFEST.exists(): MANIFEST.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")
    return x

def build():
    manifest(); used = prior_groups()
    pop = h2.make_population(SEED, list(QUOTAS), per_cell=400)
    pop = [r for r in pop if r["workload_group_id"] not in used]
    sel = select(pop, SEED + 100, used)
    return {"status": "FROZEN_BEFORE_SCORING", "date": "2026-11-08", "panel": "BATCH3_CONFIRMATORY",
            "generator": "oom_risk_heldout3_validation.py", "generator_sha256": file_sha(Path(__file__)),
            "seed": SEED, "population_size": len(pop), "selected_count": len(sel), "sampling_quotas": QUOTAS,
            "selection_independent_of_model_predictions": True, "compile_filter_used": False,
            "selected_candidates": sel, "overlap_audit": audit(sel), "outcomes_excluded": True}

def score_panel():
    p = json.loads(PLAN.read_text()); v1 = json.loads(h2.V1.read_text())
    v2 = json.loads(h2.V2.read_text()); v3 = json.loads(h2.V3.read_text()); items = []
    for c in p["selected_candidates"]:
        f = base.expansion().static_features(c["family"], c["configuration"], c["dtype"], v1)
        items.append({k: c[k] for k in ("candidate_id", "workload_group_id", "family", "configuration", "dtype", "graph_fingerprint", "sampling_stratum")}
                     | {"v2_probability": h2.score_model(f, v2), "v3_probability": h2.score_model(f, v3),
                        "aggregate_score": f.get("calibrated_upper_over_budget"), "static_features": f})
    payload = {"status": "FROZEN_BEFORE_RUNTIME", "panel": p["panel"], "plan_hash": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
               "v2_model_hash": v2["model_hash"], "v3_model_hash": v3["model_hash"], "predictions": items, "outcomes_excluded": True}
    payload["prediction_hash"] = sha(payload)
    PRED.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); return payload

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=("manifest", "plan", "score"), required=True); a = ap.parse_args()
    if a.stage == "manifest": print(json.dumps(manifest(), indent=2, sort_keys=True))
    elif a.stage == "plan":
        if PLAN.exists(): raise SystemExit("refusing to overwrite frozen plan")
        p = build(); PLAN.write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"selected": p["selected_count"], "audit": p["overlap_audit"],
                          "families": dict(Counter(x["family"] for x in p["selected_candidates"])),
                          "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest()}, indent=2))
    else:
        if PRED.exists(): raise SystemExit("refusing to overwrite predictions")
        print(json.dumps({"prediction_hash": score_panel()["prediction_hash"]}, indent=2))
if __name__ == "__main__": main()
