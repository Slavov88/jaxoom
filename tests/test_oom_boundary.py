from experiments.oom_boundary_validation import classify_failure, classify_prediction, summarize_trials


def test_prediction_categories_are_frozen():
    assert classify_prediction(10, 20, 20) == "PREDICTED FIT"
    assert classify_prediction(10, 30, 20) == "UNCERTAIN"
    assert classify_prediction(21, 30, 20) == "PREDICTED EXCEEDS"


def test_oom_stage_classification_is_separate():
    assert classify_failure("compile", "RESOURCE_EXHAUSTED: out of memory") == "COMPILE_OOM"
    assert classify_failure("execute", "CUDA_ERROR_OUT_OF_MEMORY") == "EXECUTION_OOM"
    assert classify_failure("execute", "shape mismatch") == "OTHER_FAILURE"


def test_summary_counts_false_fit_and_false_oom():
    rows = [
        {"trial_id": "a", "risk": "LOW", "prediction": "PREDICTED FIT", "status": "FIT"},
        {"trial_id": "b", "risk": "LOW", "prediction": "PREDICTED FIT", "status": "EXECUTION_OOM"},
        {"trial_id": "c", "risk": "LIKELY EXCEEDS BUDGET", "prediction": "PREDICTED EXCEEDS", "status": "FIT"},
    ]
    summary = summarize_trials(rows)
    assert summary["false_fit_ids"] == ["b"]
    assert summary["false_oom_ids"] == ["c"]
    assert summary["risk_x_outcome"]["LOW"]["EXECUTION_OOM"] == 1
