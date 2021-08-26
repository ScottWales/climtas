import dask
import pandas
import numpy
from climtas.helpers import *


def test_blockwise():
    da = dask.array.zeros((10, 10), chunks=(5, 5))

    def func(da):
        return pandas.DataFrame({"mean": da.mean()}, index=[1])

    meta = pandas.DataFrame({"mean": pandas.Series([], dtype=da.dtype)})

    df = map_blocks_array_to_dataframe(func, da, meta=meta)
    df = df.compute()

    numpy.testing.assert_array_equal(df.to_numpy(), [[0], [0], [0], [0]])

    def func(da, block_info=None):
        return pandas.DataFrame.from_records([block_info[0]], index=[1])

    df = map_blocks_array_to_dataframe(func, da, meta=meta)
    df = df.compute()

    numpy.testing.assert_array_equal(
        df["chunk-location"].sort_values().apply(lambda x: x[0]),
        numpy.array([0, 0, 1, 1,]),
    )

    numpy.testing.assert_array_equal(
        df["chunk-location"].sort_values().apply(lambda x: x[1]),
        numpy.array([0, 1, 0, 1,]),
    )


def test_blockwise_xarray():
    da = dask.array.zeros((10, 10), chunks=(5, 5))
    xda = xarray.DataArray(da, dims=["t", "x"])

    def func(da, block_info=None):
        meta = locate_block_in_dataarray(
            da, xda.name, xda.dims, xda.coords, block_info[0]
        )
        return pandas.DataFrame({"mean": meta.mean().values}, index=[1])

    meta = pandas.DataFrame({"mean": pandas.Series([], dtype=da.dtype)})

    df = map_blocks_array_to_dataframe(func, xda.data, meta=meta)
    df = df.compute()

    numpy.testing.assert_array_equal(df.to_numpy(), [[0], [0], [0], [0]])


def test_throttled_compute():

    # Numpy array
    s = numpy.random.random((10, 10))

    tc = throttled_compute(s, n=1)
    (dc,) = dask.compute(s)
    numpy.testing.assert_array_equal(tc, dc)

    # Numpy array converted to dask
    s = dask.array.from_array(s, chunks=(5, 5))

    tc = throttled_compute(s, n=1)
    (dc,) = dask.compute(s)
    numpy.testing.assert_array_equal(tc, dc)

    # Pure dask array
    s = dask.array.random.random((10, 10), chunks=(5, 5))

    tc = throttled_compute(s, n=1)
    (dc,) = dask.compute(s)
    numpy.testing.assert_array_equal(tc, dc)

    # Xarray + Numpy
    s = numpy.random.random((10, 10))
    s = xarray.DataArray(s, name="foo")

    tc = throttled_compute(s, n=1)
    (dc,) = dask.compute(s)
    numpy.testing.assert_array_equal(tc, dc)
    assert tc.name == "foo"

    # Xarray + Dask
    s = dask.array.random.random((10, 10), chunks=(5, 5))
    s = xarray.DataArray(s, name="foo")

    tc = throttled_compute(s, n=1)
    (dc,) = dask.compute(s)
    numpy.testing.assert_array_equal(tc, dc)
    assert tc.name == "foo"


def test_array_blocks_to_dataframe():
    meta = pandas.DataFrame({"a": [0, 1, 2], "b": [2, 3, 4]})

    array = numpy.array([0, 1, 2, 3])
    a = dask.array.from_array(array, chunks=(2,))

    def mapper(x):
        if x[0] == 0:
            return meta.iloc[0:1]
        else:
            return meta.iloc[1:]

    b = dask.array.map_blocks(mapper, a, dtype="object")

    result = array_blocks_to_dataframe(b, meta).compute()

    assert meta.equals(result)
