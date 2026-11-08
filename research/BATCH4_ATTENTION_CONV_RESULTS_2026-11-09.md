# Batch-4 attention/convolution LIKELY_FIT panel: results (2026-11-09)

Status: COMPUTATIONALLY VERIFIED (observed outcomes); bounds are exact binomial
calculations on those counts, not theorems.

Design: 160 unseen recipes (80 attention, 80 convolution), all frozen-V2 LIKELY_FIT
(p<0.20), zero workload-group/fingerprint overlap with 977 prior rows plus the
batch-3 plan. Selection used model scores only. Rule amendment 1 (before any
runtime outcome) replaced the empty "low" band [0,0.10) with the occupied range, because
V2 LIKELY_FIT scores never fall below about 0.126; the first manifest and the
superseded 40-per-family draft are retained (`experiments/superseded/`).
Runner: `oom_risk_heldout4_campaign.py`, fresh process per recipe, 600 s bound,
RTX 3050 4096 MiB, WSL; looser than the uniform qualification (no source
attestation, no per-trial recovery). One run.

## Outcomes

160/160 FIT, zero OOM, zero timeouts, no duplicates, max recipe wall time 35 s.
Kill criterion not triggered. All 160 were also V3 LIKELY_FIT (V3 reported only).

| Family | n | FIT | Exact 95% upper bound (alone) |
|---|---|---|---|
| attention | 80 | 80 | 3.68% |
| convolution | 80 | 80 | 3.68% |

Pooled with batch 3 only (earlier per-family counts not re-derived, so this is
conservative): attention 96 trials -> 3.07%; convolution 100 -> 2.95%.
Pooled all LIKELY_FIT trials across batches 1-4 (157+125+160 = 442, zero failures):
0.68%.

## Limits

- Scale coverage is uneven: attention LIKELY_FIT recipes are almost all small or
  medium (34 small, 33 medium, 11 large, 2 near-device), because large and
  near-device attention are rarely called LIKELY_FIT. The attention bound does
  not cover large or near-device attention. Convolution is balanced (20 per scale).
- The panel has 107 within-panel near-duplicate pairs; the clustered bound is not
  computed here and would be larger.
- Fit times were short (max 35 s), unlike the boundary cases that took hundreds of
  seconds earlier, which suggests this LIKELY_FIT set is far from compile stress.
- All 160 sit in the p 0.126-0.20 range, so they tell us nothing about
  non-LIKELY_FIT behaviour or coverage.
- Same GPU/stack and restricted recipe types; public API stays disabled.
