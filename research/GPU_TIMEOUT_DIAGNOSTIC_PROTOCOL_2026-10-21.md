# Bounded timeout diagnosis and child source verification

Frozen before new diagnostic outcomes. User explicitly requested further repo
progress and commits. This is software failure diagnosis, not a new accuracy
experiment, model update, or retroactive qualification pass.

Preserve all 201 prior launches, the five historical 180-second timeouts, model,
templates, threshold, allowlist, environment and idle guards byte-for-byte.
Select exactly the five unresolved uncertain recipes from verified complete
200-recipe acquisition, in original plan order. Never rerun a LIKELY_FIT recipe,
known FIT/OOM result, or a failed likely-fit workload. Use one diagnostic launch
per selected recipe, identical zero inputs and fresh CUDA process, with a fixed
600-second outer limit for every selected recipe and one-hour campaign cap.
This predeclared extension asks whether bounded allocator/compilation retries
eventually return a terminal result. It changes only the diagnostic waiting
budget; it must not relabel the original 180-second qualification as passed.
No second attempt, alternate seed, allocator/driver/XLA override, retraining,
new dependency or new recipe is permitted.

Before launch, verify the prior evidence chain, all recorded artifact hashes,
exact recipe selection and current source bytes. Each child independently
hashes the diagnostic entry point plus all imported qualification wrappers,
adapter, policy and replay sources. It verifies the diagnostic plan file hash,
internal content hash, original frozen plan hash and replay file hash before
telemetry/CUDA initialization or workload execution. It emits a flushed
attestation event and includes its measured hashes in its returned result.
The parent validates the actual child event/result, never replacing those fields
with its expected hashes. Missing, duplicate, changed or contradictory source,
plan, replay, identity, feature or score evidence fails closed. Raw attestation
events survive timeouts. Tests must cover tampering, duplicate/missing events,
and refusal to invoke GPU work on source mismatch.

Require fresh 224-row CUDA replay and public export absence before diagnostics.
Keep phase-specific terminal OOM, FIT, timeout and infrastructure failures
distinct. A diagnostic timeout remains unknown. After each child exits or is
killed/reaped, require and retain the original clean idle telemetry before
continuing. Stop on unexpected failure, missing/invalid attestation, unsupported
or busy environment, source/replay mismatch, failed recovery or resource cap.
Do not infer terminal OOM from BFC warnings. Even all-terminal diagnostics do
not pass the historical qualification gate or enable a public API; report the
new bound and separate evidence explicitly. Run the full regression suite,
audit persisted events/results/hash chains and commit source/evidence separately.
