# Fresh unseen-panel validation: draft protocol (not yet frozen)

Status: CONJECTURED design. Nothing here has been run. To be frozen in a commit
BEFORE any panel recipe is generated or executed, per AGENTS.md rules 1-2.

## Question and hypothesis

Does frozen V2 at `p_fit=0.20` keep zero false-safe OOMs on recipes that played
no part in training, threshold choice, or the 200-recipe qualification, and can
the exact one-sided upper bound on the false-safe rate be tightened from about
1.9% (about 2.35% under strict clustering)?

Mechanism: the logistic features track allocator-visible tensor and temporary
sizes; a LIKELY_FIT call fails only if the features miss a compiler or allocator
behaviour (the weak point for attention and convolution).

## Strongest baseline

Frozen V2 as already confirmed (pooled 190 eligible, 157 LIKELY_FIT, 0 false-safe).
Comparison arm: frozen V3, reported only, not used for the decision. No model,
coefficient, template, or threshold change.

## Design

- New seeds and shape grids, disjoint from every historical spec hash (verified
  by hash before launch). Recipes generated and scored BEFORE any GPU execution,
  with predictions committed first (the earlier held-out pattern).
- Stratify by family with extra weight on attention and convolution, and sample
  near the predicted boundary (scores near the 0.20 threshold), where a
  false-safe is most likely. Cap near-duplicates per cluster.
- Same environment and runner contract as the uniform campaign: 600 s child
  bound, one attempt, idle gates, source attestation, independent audit.
- Target at least 150 new LIKELY_FIT recipes. With zero failures the exact
  one-sided 95% bound is about 3/n, so n=150 gives about 2.0% alone; pooled with
  the 157 already confirmed it is roughly 1.0%. Report the new-panel bound
  separately from the pooled one.

## Kill criterion

Any new LIKELY_FIT recipe that is an execution or compile OOM: the false-safe
claim FAILS for that scope. Record it, do not adjust the threshold or retry.
Timeouts stay unknown and count as incomplete, never as FIT.

## Cost check (rule 14)

Pilot first: generate and score 20 recipes without running the GPU to confirm
disjointness and the LIKELY_FIT yield; only then size the full panel. The run
costs roughly the same GPU time as the uniform campaign per 200 recipes.

## Not covered

Other GPUs, ROCm, generic callables, dynamic budgets, higher-than-tested scales.
