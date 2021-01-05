from climtas.daskutil import *


def compare_compute(s):
    a = throttled_compute(s, n=10)
    (b,) = dask.compute(s)
    numpy.testing.assert_array_equal(a, b)


def test_throttled_compute():
    s = numpy.random.random((10, 10))
    compare_compute(s)

    s = dask.array.from_array(s, chunks=(5, 5))
    compare_compute(s)

    s = dask.array.random.random((10, 10), chunks=(5, 5))
    compare_compute(s)

    t = dask.array.random.random((10, 10), chunks=(2, 2))
    s = s @ t
    compare_compute(s)


def test_visualize_block():
    import dask.dot

    s = dask.array.random.random((10, 10), chunks=(5, 5))
    s = s + 1
    v = visualize_block(s)

    assert "label=add" in v.source
