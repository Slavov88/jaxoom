from experiments.t4_allocator_residency import outcome


def test_t4_outcome_classification_prefers_execution_status():
    assert outcome({"execution_statuses": ["EXECUTION_OOM"], "status": "FIT"}) == "EXECUTION_OOM"
    assert outcome({"compile_status": "COMPILE_OOM"}) == "COMPILE_OOM"
    assert outcome({"status": "EXECUTION_TIMEOUT"}) == "EXECUTION_TIMEOUT"
