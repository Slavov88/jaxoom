# Completion protocol, frozen before completion trials

The user requested continued work. Continuation1 stopped on an uncertain compile
timeout, leaving 154 unrun recipes. Preserve both stopped runs and all unknowns.
This protocol broadens evidence-acquisition recovery to uncertain COMPILE_TIMEOUT
as well as EXECUTE_TIMEOUT; it does not broaden qualification acceptance.
Reuse exact original runner/model/policy/runtime/threshold/180-second limit.
Validate both previous artifact audit manifests, original plan and child identity.
Run fresh 224-row CUDA replay, then once each of the remaining recipes in original
order. No retries, changed seeds, model fitting, thresholds, drivers, dependencies,
or prior outcome reinterpretation. After any uncertain compile/execute timeout,
subprocess.run must kill/reap the child and the frozen idle gate must pass before
another target is launched. Record recovery telemetry. Any likely-fit failure,
other failure, initialization timeout, invalid provenance, busy GPU, missing
evidence, or failed recovery stops execution. One-hour campaign cap (checked after
idle gating with 180 seconds reserved); twelve new timeouts maximum. Existing
unknowns continue blocking all_checks_pass, even if attempted coverage reaches
200/200. Public API stays disabled. Negative-control tests cover both phases,
likely-fit failures, unknown/mismatched phase states and acceptance separation.
Full regression and persisted evidence audits follow execution. The main session
owns all writes/execution; a read-only reviewer may audit. This is software
qualification of the historical population, not new independent accuracy evidence.
