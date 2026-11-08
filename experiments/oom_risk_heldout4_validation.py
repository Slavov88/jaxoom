"""Batch-4 attention/convolution LIKELY_FIT panel (outcome-free construction).

Predeclared rule: generate a large unseen population, score it with frozen V2/V3
(model scores only, no outcomes exist), then select V2 LIKELY_FIT (p<0.20)
candidates per family: half in the boundary band [0.10,0.20), half in [0,0.10),
balanced across scales as far as available. Runtime comes after the plan commit.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("heldout3", ROOT / "oom_risk_heldout3_validation.py")
h3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h3)
h2, base = h3.h2, h3.base
SEED = 20261109
PLAN = ROOT / "oom_risk_heldout4_plan_2026-11-09.json"
PRED = ROOT / "oom_risk_heldout4_predictions_2026-11-09.json"
MANIFEST = ROOT / "oom_risk_heldout4_validation_manifest_2026-11-09_amendment1.json"
TARGET = {"attention": 80, "convolution": 80}
POP_PER_CELL = {"attention": 220, "convolution": 160}
SCALES = ("small", "medium", "large", "near_device")
BANDS = (("boundary", 0.0, 0.20),)  # amendment 1: V2 LIKELY_FIT scores never fall below ~0.126, so the low band is empty

def manifest():
    models = {}
    for name, path in (("v2", h2.V2), ("v3", h2.V3)):
        x = json.loads(path.read_text()); models[name] = {"model_hash": x["model_hash"], "artifact_sha256": h2.file_sha(path)}
    x = {"status": "FROZEN_BEFORE_BATCH4_PANEL_AMENDMENT1", "date": "2026-11-09", "primary_model": "V2", "models": models,
         "likely_fit_rule": "v2_probability < 0.20", "target_likely_fit": TARGET, "population_per_cell": POP_PER_CELL,
         "band_rule": "all LIKELY_FIT (p<0.20), balanced across scales; amendment 1 of the 2026-11-09 manifest: the low band [0,0.10) is empty because observed V2 LIKELY_FIT scores are >=0.126, so the full target is drawn from the occupied range",
         "amends": "oom_risk_heldout4_validation_manifest_2026-11-09.json",
         "amendment_made_before_any_runtime_outcome": True, "seed": SEED,
         "no_retraining": True, "selection_uses_model_scores_only": True, "selection_uses_outcomes": False,
         "excluded_prior": "all rows of batches 1-3 incl. batch-3 plan",
         "kill_criterion": "any selected LIKELY_FIT recipe that is execution or compile OOM fails the per-family false-safe claim"}
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != x: raise RuntimeError("refusing to overwrite frozen manifest")
    if not MANIFEST.exists(): MANIFEST.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")
    return x

def prior():
    rows = list(h3.prior_rows()); rows.extend(json.loads(h3.PLAN.read_text())["selected_candidates"]); return rows

def build():
    manifest(); rows = prior(); used = {r["workload_group_id"] for r in rows}; fps = {r.get("graph_fingerprint") for r in rows}
    v1 = json.loads(h2.V1.read_text()); v2 = json.loads(h2.V2.read_text()); v3 = json.loads(h2.V3.read_text())
    selected, preds, stats = [], [], {}
    for fam in TARGET:
        pop = [r for r in h2.make_population(SEED + (0 if fam == "attention" else 1), [fam], per_cell=POP_PER_CELL[fam])
               if r["workload_group_id"] not in used and r["graph_fingerprint"] not in fps]
        scored = []
        for c in pop:
            f = base.expansion().static_features(c["family"], c["configuration"], c["dtype"], v1)
            scored.append((c, f, h2.score_model(f, v2), h2.score_model(f, v3)))
        lf = [t for t in scored if t[2] < 0.20]
        rng = np.random.default_rng(SEED + 100); chosen = []
        for bname, lo, hi in BANDS:
            need = TARGET[fam]; per_scale = {s: [t for t in lf if t[0]["sampling_stratum"]["scale"] == s and lo <= t[2] < hi] for s in SCALES}
            for s in SCALES: rng.shuffle(per_scale[s])
            got = []
            while len(got) < need and any(per_scale.values()):
                for s in SCALES:
                    if per_scale[s] and len(got) < need: got.append(per_scale[s].pop())
            chosen.extend(got)
        stats[fam] = {"population": len(pop), "likely_fit_available": len(lf), "selected": len(chosen),
                      "by_band": {b: sum(lo <= t[2] < hi for t in chosen) for b, lo, hi in BANDS},
                      "by_scale": dict(Counter(t[0]["sampling_stratum"]["scale"] for t in chosen))}
        for t in chosen: used.add(t[0]["workload_group_id"]); selected.append(t)
    items, cands = [], []
    for i, (c, f, p2, p3) in enumerate(selected):
        c = dict(c); c["candidate_id"] = f"H4-{i:04d}"; cands.append(c)
        items.append({k: c[k] for k in ("candidate_id", "workload_group_id", "family", "configuration", "dtype", "graph_fingerprint", "sampling_stratum")}
                     | {"v2_probability": p2, "v3_probability": p3, "aggregate_score": f.get("calibrated_upper_over_budget"), "static_features": f})
    plan = {"status": "FROZEN_BEFORE_RUNTIME", "date": "2026-11-09", "panel": "BATCH4_ATTENTION_CONV_LIKELY_FIT",
            "generator": "oom_risk_heldout4_validation.py", "generator_sha256": h2.file_sha(Path(__file__)), "seed": SEED,
            "selected_count": len(cands), "selection_stats": stats, "selected_candidates": cands,
            "overlap_audit": h3.audit(cands), "outcomes_excluded": True}
    return plan, items

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=("manifest", "plan"), required=True); a = ap.parse_args()
    if a.stage == "manifest": print(json.dumps(manifest(), indent=2, sort_keys=True)); return
    if PLAN.exists() or PRED.exists(): raise SystemExit("refusing to overwrite frozen files")
    plan, items = build(); PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    payload = {"status": "FROZEN_BEFORE_RUNTIME", "panel": plan["panel"], "plan_hash": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
               "predictions": items, "outcomes_excluded": True}
    payload["prediction_hash"] = h2.sha(payload); PRED.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected": plan["selected_count"], "stats": plan["selection_stats"], "audit": plan["overlap_audit"]}, indent=2))
if __name__ == "__main__": main()
