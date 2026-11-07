# Uniform restricted GPU qualification: results (run 1, 2026-11-07)

Status: COMPUTATIONALLY VERIFIED for the exact scope below. Not a theorem, not a
new accuracy panel, and the public OOM-risk API stays disabled.

Run: `experiments/oom_risk_gpu_qualification_uniform_2026-11-07_run1/`, launched
from clean commit `bb7572c` under the frozen protocol
(`GPU_UNIFORM_QUALIFICATION_PROTOCOL_2026-10-21.md`), one attempt per recipe, 600 s
child bound, RTX 3050 Laptop 4096 MiB, driver 566.07, JAX/jaxlib 0.11.0, WSL Ubuntu.

## Outcome (runner summary, `summary.json`)

| Item | Result |
|---|---|
| Recipes attempted / not run | 200 / 0 (coverage complete) |
| Pilots FIT | 18/18 |
| LIKELY_FIT recipes | 157, failures 0 |
| False-safe execution OOMs | 0 |
| Outcomes | 176 FIT, 17 EXECUTION_OOM, 7 COMPILE_OOM |
| Timeouts / stop reason | 0 / none |
| Final idle and source checks | pass |

Independent persisted audit (`audit_windows.json`, `audit_linux.json`): both
`CONSISTENT_UNIFORM_QUALIFICATION_EVIDENCE`, `qualification_pass: true`. The
Windows audit used Python 3.13 with jax 0.6.2 (audit reads persisted evidence only).

## Scope and limits

- Supports only: on this GPU and software stack, for the exact 200 restricted
  recipes and frozen V2 at `p_fit=0.20`, every LIKELY_FIT recipe fit under 600 s.
- Same population as the historical acquisition (all five earlier timeouts now
  returned terminal results); this is integration qualification, not independent
  accuracy evidence. Earlier bounds are unchanged (about 1.9% exact, about 2.35%
  under strict clustering).
- Single run. Per AGENTS.md rule 8, reproduce it before treating it as more than
  one observation. No wall-clock tail guarantee; the 600 s bound is an observed
  protocol bound.
- Release/export is a separate decision under `RELEASE_DECISION_CRITERIA_2026-11-07.md`.

## Reproduction (run 2, 2026-11-08)

COMPUTATIONALLY VERIFIED. A second fresh campaign under the same frozen protocol
(`experiments/oom_risk_gpu_qualification_uniform_2026-11-08_run2/`, same commit
lineage and environment) returned the same summary (200/200, 18/18 pilots,
157/157 LIKELY_FIT FIT, 176 FIT / 17 execution OOM / 7 compile OOM, zero
timeouts), and both Windows and Linux audits passed. All 200 per-recipe outcomes
are identical to run 1 (0 differences). Same machine, same population, so this
shows run-to-run stability here, not independence or new accuracy evidence.
