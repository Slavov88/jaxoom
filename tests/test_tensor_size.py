import jax
import pytest

from jaxoom.analysis.tensor_size import array_nbytes, format_bytes, parse_memory_limit


@pytest.mark.parametrize(
    ("dtype", "itemsize"),
    [("bool", 1), ("int32", 4), ("float16", 2), ("bfloat16", 2), ("float32", 4), ("float64", 8)],
)
def test_dense_sizes(dtype, itemsize):
    assert array_nbytes(jax.ShapeDtypeStruct((2, 3), dtype)) == 6 * itemsize


def test_scalar_and_zero_sized():
    assert array_nbytes(jax.ShapeDtypeStruct((), "float32")) == 4
    assert array_nbytes(jax.ShapeDtypeStruct((0, 100), "float32")) == 0


def test_binary_format_and_limit():
    assert format_bytes(0) == "0 B"
    assert format_bytes(1536) == "1.50 KiB"
    assert parse_memory_limit("1.5 GiB") == int(1.5 * 1024**3)
    assert parse_memory_limit(1024) == 1024


def test_invalid_limit():
    with pytest.raises(ValueError):
        parse_memory_limit("1 GB")
