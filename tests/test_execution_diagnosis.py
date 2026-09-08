from experiments.execution_oom_diagnosis import parse_oom


def test_parse_explicit_allocation_request():
    result = parse_oom("RESOURCE_EXHAUSTED: out of memory while trying to allocate 1.00GiB")
    assert result["requested_bytes"] == 1024**3


def test_parse_unknown_diagnostic_without_numbers():
    assert parse_oom("RESOURCE_EXHAUSTED") == {}
