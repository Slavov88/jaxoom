from experiments.allocator_residency_campaign import CANDIDATES, config_id
from experiments.runtime_validation import workload_from_config


def test_campaign_configuration_ids_are_stable_and_unique():
    ids = [config_id(family, config, dtype) for family, config, dtype in CANDIDATES]
    assert len(ids) == len(set(ids))
    assert config_id("matmul", {"batch": 1, "m": 8, "n": 8, "k": 8}, "float32") == config_id("matmul", {"k": 8, "n": 8, "m": 8, "batch": 1}, "float32")


def test_screening_workload_families_trace():
    cases = [
        ("matmul", {"batch": 1, "m": 8, "n": 8, "k": 8}),
        ("convolution", {"batch": 1, "height": 8, "width": 8, "channels": 2, "out_channels": 4}),
        ("autodiff", {"batch": 2, "width": 8, "layers": 2}),
    ]
    for family, config in cases:
        workload_from_config(family, config, "float32")
