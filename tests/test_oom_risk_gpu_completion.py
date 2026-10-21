import pytest
from experiments import oom_risk_gpu_completion as completion
from experiments import oom_risk_gpu_continuation as c


@pytest.mark.parametrize("phase,outcome", [("compile", "COMPILE_TIMEOUT"), ("execute", "EXECUTE_TIMEOUT")])
def test_recovered_uncertain_timeouts_can_continue_but_never_pass(phase, outcome):
    row = dict(phase=phase, outcome=outcome, likely_fit=False, spec_hash="a")
    assert completion.recoverable(row)
    assert not c.summarize([row], total=1)["all_checks_pass"]
    assert not completion.recoverable(dict(row, likely_fit=True))
    assert not completion.recoverable(dict(row, phase="initialization"))


@pytest.mark.parametrize("row", [{}, dict(outcome="COMPILE_TIMEOUT", phase="compile"),
    dict(outcome="OTHER_FAILURE", likely_fit=False),
    dict(outcome="INITIALIZATION_TIMEOUT", phase="initialization", likely_fit=False)])
def test_unknown_or_unsupported_recovery_states_stop(row):
    assert not completion.recoverable(row)
