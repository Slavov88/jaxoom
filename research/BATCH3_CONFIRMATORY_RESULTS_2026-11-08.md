# Batch-3 confirmatory panel: results (2026-11-08)

Status: COMPUTATIONALLY VERIFIED (observed outcomes); the bounds are exact
binomial calculations on those counts. Not a theorem.

Design: 180 recipes with zero workload-group or graph-fingerprint overlap with
977 prior rows, outcome-free plan (`3f28916`/`4b5f44e`) and V2/V3 predictions
(`26647fa`) committed before any runtime. Frozen V2, threshold p<0.20 = LIKELY_FIT.
Runner: `oom_risk_heldout3_campaign.py`, fresh process per recipe, 600 s bound,
RTX 3050 4096 MiB, WSL. Looser than the uniform qualification (no source
attestation or per-trial recovery). One run, no retries.

## Outcomes

| | Count |
|---|---|
| Recipes | 180 (no duplicates, no timeouts; max wall 403 s) |
| FIT / execution OOM / compile OOM | 148 / 21 / 11 |
| V2 LIKELY_FIT | 125, **all FIT** (zero false-safe) |
| V3 LIKELY_FIT (reported only) | 134, all FIT |
| Not LIKELY_FIT (V2) | 55: 23 FIT, 21 execution OOM, 11 compile OOM |

LIKELY_FIT by family (all FIT): convolution 20, transformer 17, attention 16,
autodiff 16, mlp 13, training 12, matmul 11, reduction 10, scan 10.

## Bounds (exact one-sided 95%, zero failures)

- New panel alone: 125 trials -> 2.37%.
- Pooled with the 157 confirmed LIKELY_FIT in the earlier primary batches: 282 -> 1.06%.
- Family-level bounds remain wide: attention has only 16 LIKELY_FIT trials
  (about 17% alone) and convolution 20 (about 14%). The clustered (near-duplicate)
  bound is not computed here and would be larger.

## Limits

Kill criterion not triggered. Same GPU, software and restricted recipe types;
calibration is campaign-conditional and the panel oversamples attention and
convolution, so it is not a workload-prevalence sample. V2 approved 125 of the 148
FIT recipes (84%). Public API stays disabled; release remains a separate decision.
