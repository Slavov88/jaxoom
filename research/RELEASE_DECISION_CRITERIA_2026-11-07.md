# Release-decision criteria for the restricted OOM-risk API

Status: CONJECTURED criteria, written before the fresh uniform 200-recipe campaign
has produced any outcome. Nothing here changes the frozen qualification protocol
(`GPU_UNIFORM_QUALIFICATION_PROTOCOL_2026-10-21.md`); it only fixes, in advance,
how that campaign's result may and may not be used. The public OOM-risk API
remains disabled until a separate explicit decision.

## What a campaign pass can support

A complete audited pass supports only: "on this RTX 3050 Laptop 4096 MiB, driver
566.07, JAX/jaxlib 0.11.0 CUDA/BFC, for the exact 200 restricted-spec recipes and
the frozen V2 model at `p_fit=0.20`, every LIKELY_FIT recipe fit under a 600 s
bound." It is integration qualification on a repeated population, not new
prediction accuracy, and not a bound for arbitrary callables, other GPUs, dynamic
budgets or AMD.

## Predeclared outcome rules

| Campaign result | Allowed interpretation | Required next action |
|---|---|---|
| Complete, audited pass on Windows and Linux | Restricted-scope qualification passes | Open a separate release review; API stays disabled meanwhile |
| Any LIKELY_FIT recipe is not FIT | Qualification FAILS; a false-safe was observed | Record as a failed direction; no threshold edit without a new frozen model and fresh panel |
| Any timeout at 600 s | Incomplete; unknown FIT/OOM | Report incomplete; do not retry within the campaign or relabel |
| Non-FIT pilot, missing/corrupt/duplicate evidence, attestation mismatch | Incomplete | Fix infrastructure only; new output directory; keep stopped evidence |
| Host storage, idle or time-budget gate stops the run | Infrastructure stop, not model evidence | Resolve resource, rerun fresh; never resume |

## Release review preconditions (all required, none sufficient)

1. Audited campaign pass as above, reproduced by a second fresh run before any
   improvement or "new best" style claim (AGENTS.md rule 8).
2. The public interface accepts only restricted, versioned WorkloadSpec inputs
   and fails closed outside the qualified environment.
3. Documentation states the exact scope, the pooled exact one-sided upper bound
   (about 1.9%, about 2.35% under strict near-duplicate clustering) and that
   calibration is campaign-conditional.
4. Out-of-scope features (generic callables, dynamic budgets, ROCm, other
   hardware) are absent, not merely undocumented.

## Classification discipline

Allocation warnings never establish OOM (H2-0038 returned FIT after the 180 s
timeout). Compile OOM and execution OOM stay distinct. A campaign failure caused
by infrastructure is reported as such and is separated from model observations.
