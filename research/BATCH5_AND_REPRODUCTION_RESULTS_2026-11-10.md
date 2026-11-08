# Batch-5 large attention panel and batch 3/4 reproductions (2026-11-10)

Status: COMPUTATIONALLY VERIFIED (observed outcomes); bounds are exact binomial
calculations, not theorems. Same GPU/stack, restricted recipe types, looser runner
than the uniform qualification (no source attestation, no per-trial recovery).

## Reproductions

Same frozen plans and runners, fresh processes, new output files:
- Batch 3: 180/180 per-recipe outcomes identical to the original
  (148 FIT, 21 execution OOM, 11 compile OOM).
- Batch 4: 160/160 identical (all FIT).
Same machine, so this shows run-to-run stability, not independence.

## Batch 5 (large attention, V2 LIKELY_FIT)

Manifest `eb22f30` and plan/predictions `9119155` were committed before runtime.
Selection used model scores only. 930 generated large/near-device attention recipes
yielded only 32 LIKELY_FIT, all large scale: V2 called **zero** near-device
attention recipes LIKELY_FIT, so the model never approves that regime (a coverage
limit, not safety evidence about it).

Result: 32/32 FIT, no OOM or timeouts, max wall 7 s. Exact 95% upper bound alone
8.94%. Pooled attention LIKELY_FIT trials from batches 3-5 (16+80+32 = 128, zero
failures): 2.31%. The 52 vs-prior and 70 within-panel near-duplicate pairs mean a
clustered bound would be larger.

## Limits

LIKELY_FIT attention is still dominated by small/medium scale; large attention
rests on 11 (batch 4) + 32 (batch 5) + a few earlier trials. Near-device attention
has no LIKELY_FIT evidence at all. Public API remains disabled.
