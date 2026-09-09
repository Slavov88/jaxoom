import jax

from experiments.runtime_validation import workload_from_config
from experiments.t4_missing_family_campaign import configuration_id


def test_scan_workloads_preserve_scan_primitive_and_static_analysis():
    workload = workload_from_config("mlp_scan", {"batch": 2, "width": 8, "layers": 3, "retain": True}, "float32")
    jaxpr = jax.make_jaxpr(workload.fn)(*workload.abstract_args).jaxpr
    assert any(str(equation.primitive) == "scan" for equation in jaxpr.eqns)
    import jaxoom
    report = jaxoom.estimate(workload.fn, *workload.abstract_args)
    assert report.estimated_peak_bytes > 0


def test_scan_family_ids_are_canonical():
    assert configuration_id("training_scan", {"batch": 2, "width": 8, "layers": 3}, "float32") == configuration_id("training_scan", {"layers": 3, "width": 8, "batch": 2}, "float32")
    assert configuration_id("training_scan", {"batch": 2, "width": 8, "layers": 3}, "float32") != configuration_id("autodiff_scan", {"batch": 2, "width": 8, "layers": 3}, "float32")
