import jaxoom

from experiments.allocator_aware_safety_gate import arguments, attention, large_allocation_proxy


def test_attention_proxy_uses_two_largest_peak_live_buffers():
    report = jaxoom.estimate(attention, *arguments(5120))
    assert large_allocation_proxy(report, 1) == 838_860_800
    assert large_allocation_proxy(report, 2) == 1_677_721_600


def test_proxy_is_precompilation_static_quantity():
    report = jaxoom.estimate(attention, *arguments(4608))
    assert report.estimated_peak_bytes > 0
    assert large_allocation_proxy(report) < report.estimated_peak_bytes
