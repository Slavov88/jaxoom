from experiments.t4_allocator_residency_refinement import bracket


def test_bracket_ignores_timeouts_and_preserves_capacity_interval():
    def row(status, limit):
        return {"status": status, "snapshots": [{"allocator_limit_bytes": limit}]}

    result = bracket([
        row("EXECUTION_OOM", 100),
        row("EXECUTION_TIMEOUT", 150),
        row("FIT", 200),
    ])
    assert result == {
        "known_oom_capacity": 100,
        "known_fit_capacity": 200,
        "bracket_width_bytes": 100,
        "usable": True,
    }


def test_bracket_requires_both_fit_and_oom():
    assert bracket([{"status": "FIT", "snapshots": [{"allocator_limit_bytes": 200}]}]) is None
