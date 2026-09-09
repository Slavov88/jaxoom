from experiments.t4_allocator_residency_targeting import classify_candidate


def test_targeting_rejects_obviously_small_candidate():
    status, low, high = classify_candidate(
        {"calibrated_upper_bytes": 100},
        {"ratio_low": 1.0, "ratio_high": 1.5},
        1000,
        5000,
    )
    assert status == "TOO_SMALL"
    assert (low, high) == (100, 150)


def test_targeting_accepts_candidate_overlapping_capacity_range():
    status, low, high = classify_candidate(
        {"calibrated_upper_bytes": 2000},
        {"ratio_low": 1.0, "ratio_high": 1.5},
        1000,
        5000,
    )
    assert status == "LIKELY_THRESHOLD"
    assert low <= 2000 <= high
