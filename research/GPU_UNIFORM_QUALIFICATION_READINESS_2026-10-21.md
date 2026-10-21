# Uniform restricted GPU qualification: implementation and launch readiness

The fresh 200-recipe campaign is **NOT_RUN**. The runner, portable auditor and
102 new regression tests are implemented. Both complete CPU suites passed all
549 tests, with zero failures, errors or skips. Public OOM-risk exports remain
disabled. This stage establishes tested qualification infrastructure, not a new
GPU qualification result or an improved prediction-accuracy claim.

## Frozen contract and implementation

Protocol commit `e10a211` freezes the original 200 primary recipes, the original
18 structural pilots, a uniform 600-second child timeout, a three-hour launch
budget and a 512 MiB minimum free-space gate. Existing model, templates, feature
semantics, threshold, applicability policy and historical evidence are preserved.
See `GPU_UNIFORM_QUALIFICATION_PROTOCOL_2026-10-21.md`.

Implementation commit `c92f91c` adds:

- A fresh-process runner that checks exact source/reference/catalog identity,
  requires source-bound full Windows and Linux test reports, performs a fresh
  224-row CUDA replay and independently measures nine source hashes plus plan,
  replay and validation-file hashes in every child before GPU work.
- Fixed launch order, per-trial storage/idle guards, measured child provenance,
  raw logs, phase-aware terminal results, recovery/final checks and conservative
  stopping. No resume, outcome-driven timeout changes or retries within a run.
- A portable persisted auditor that distinguishes internally consistent stopped
  evidence from a complete qualification pass and checks raw/result identity,
  measured provenance, frozen scores, ordered coverage and telemetry.
- Seventy-two runner and thirty auditor tests. Review exposed a summary weakness
  where a forged `likely_fit=False` field could mask failure of a frozen-model
  LIKELY_FIT candidate. The summary now derives that population from the pinned
  catalog and requires every such outcome to be FIT; a regression covers it.

Neither a runner pass nor an auditor pass enables the public API. Release/export
remains a separate decision under the protocol.

## Verification evidence

`experiments/oom_risk_gpu_qualification_uniform_tests_2026-10-21/` contains:

- `windows.xml`: full suite, 549 tests, zero failures/errors/skips, exit code 0.
- `linux.xml`: full WSL Linux suite, the same 549 tests, zero failures/errors/skips,
  exit code 0. Both environments use JAX/jaxlib 0.11.0 with CPU selection.
- `validation.json`: exact source/test/auditor hashes, both JUnit file hashes and
  population sizes, and tested implementation commit. The runner's validation
  function accepted this manifest against the current files.
- `launch_preflight.json`: a timestamped storage measurement, not GPU evidence.

The frozen V2 artifact SHA256 remains
`5b2eeedf9a180af1c79678e94ecf3401b0256b4906e1d8f5180f058d3b354ae3`.
Restricted WorkloadSpec/applicability sources and all public package files are
unchanged from the prior diagnostic stage. The pinned original reference plan
still validates with exactly 200 candidates. Public `jaxoom.WorkloadSpec` and
`jaxoom.assess_oom_risk` are absent.

## Launch blocker and next action

At `2026-10-21T16:33:33.598404+00:00`, the host filesystem had 498,708,480 bytes
(475.6 MiB) free, below the frozen 536,870,912-byte requirement. The preflight is
`BLOCKED_HOST_STORAGE`: zero fresh GPU trials and no fresh replay were started.
A separate live GPU probe showed the RTX 3050 Laptop GPU idle (0 MiB used,
3964 MiB free, 0% utilization, driver 566.07); storage is the current blocker.
These observations are snapshots and must be rechecked before execution.

After space is available, validate the manifest again and run from the repository
root in WSL Ubuntu:

```bash
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 \
  /home/slavov/.venvs/jaxoom-cuda-20261021/env/bin/python \
  -m experiments.oom_risk_gpu_uniform_qualification \
  --validation experiments/oom_risk_gpu_qualification_uniform_tests_2026-10-21/validation.json \
  --output-dir experiments/oom_risk_gpu_qualification_uniform_2026-10-21_run1
```

The parent selects CPU; the existing child environment explicitly selects CUDA.
Use a new output directory and retain any stopped-run evidence. Audit persisted
results on both platforms with `experiments.oom_risk_gpu_uniform_audit` before
claiming restricted qualification. Historical 180-second acquisition and the
separate five-case 600-second diagnostics retain their original classifications.
They do not substitute for this new campaign or demonstrate general OOM accuracy.
