# Release review against the predeclared criteria (2026-11-10)

Criteria: `RELEASE_DECISION_CRITERIA_2026-11-07.md`. Status of this review:
CONJECTURED recommendation by the analyst; the release decision itself is the
owner's. The public API remains disabled.

## Precondition check

| Precondition | Status | Evidence |
|---|---|---|
| 1. Audited campaign pass, reproduced by a second fresh run | MET (same machine) | run 1 and run 2: 200/200, 157/157 LIKELY_FIT FIT, 0 differences; Windows and Linux audits pass |
| 2. Interface accepts only restricted versioned WorkloadSpec and fails closed outside the qualified environment | NOT VERIFIED here | restricted adapter and applicability policy exist and are tested on CPU; no public interface was built or reviewed in this work |
| 3. Documentation states exact scope, bounds, campaign-conditional calibration | NOT DONE | numbers exist in the result reports; no user-facing documentation written |
| 4. Out-of-scope features absent, not just undocumented | NOT VERIFIED | public WorkloadSpec/assess_oom_risk are absent today; absence in a future release needs a test |

## Evidence summary (all LIKELY_FIT, zero false-safe)

Batches 1-2 pooled 157; qualification reruns 157 x2 (same population);
batch 3: 125; batch 4: 160 (attention/convolution); batch 5: 32 (large attention).
Independent unseen LIKELY_FIT trials: 157+125+160+32 = 474, zero failures; exact
95% upper bound about 0.63%. Per family the bounds are looser (attention about
2.3%, convolution about 3.0% pooled with batch 3), near-duplicate clustering
raises them, and near-device attention has no LIKELY_FIT support.

## Recommendation

Do not release yet. The statistical evidence for the restricted scope is strong
for pooled use but gaps remain that the criteria require closing: (a) a reviewed
fail-closed public interface and tests (criteria 2, 4); (b) user-facing scope and
bound documentation (3); (c) independent-hardware reproduction before any claim
beyond RTX 3050 4 GB, JAX 0.11.0; (d) the batches 3-5 runner lacks the attested
provenance of the uniform qualification, so rerunning them under an attested
runner would make the evidence uniform. Items (a) and (b) are engineering work and
can start now; (c) and (d) are optional hardening before a broader claim.
