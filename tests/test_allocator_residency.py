from experiments.allocator_residency_validation import threshold_summary


def row(fraction, limit, status):
    return {
        "family": "attention",
        "configuration": {"sequence": 4096},
        "dtype": "float32",
        "configured_fraction": fraction,
        "snapshots": [{"allocator_limit_bytes": limit}],
        "execution_statuses": [status],
    }


def test_threshold_summary_records_capacity_bracket():
    result = threshold_summary([
        row(0.75, 300, "EXECUTION_OOM"),
        row(0.77, 320, "FIT"),
    ])
    assert result[0]["required_capacity_lower_bytes"] == 300
    assert result[0]["required_capacity_upper_bytes"] == 320
