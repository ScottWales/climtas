from climtas.apply_doy import *

import xarray
import pandas
from climtas.helpers import chunk_count, graph_size


def test_map_doy_chunking():
    time = pandas.date_range("20010101", "20030101")
    data = numpy.random.rand(len(time))

    da = xarray.DataArray(data, coords=[("time", time)])
    da = da.chunk({"time": 365})

    default = da.groupby("time.dayofyear").map(lambda x: x + 1)

    new = map_doy(lambda x: x + 1, da)

    # The new graph should be much smaller than the old one
    assert chunk_count(new) < 20
    assert graph_size(new) < 20

    assert chunk_count(default) > 600
    assert graph_size(default) > 600

    xarray.testing.assert_equal(default, new)


def test_reduce_doy_chunking():
    time = pandas.date_range("20010101", "20030101")
    data = numpy.random.rand(len(time))

    da = xarray.DataArray(data, coords=[("time", time)])
    da = da.chunk({"time": 365})

    default = da.groupby("time.dayofyear").sum()

    new = reduce_doy(numpy.sum, da)

    # The new graph should be much smaller than the old one
    assert chunk_count(new) < 10
    assert graph_size(new) < 10

    assert chunk_count(default) == 365
    assert graph_size(default) > 600

    xarray.testing.assert_equal(default, new)


def test_percentile_doy():
    da = xarray.DataArray(
        [0, 2, 1, 0],
        coords=[
            (
                "time",
                pandas.to_datetime(["20010101", "20010102", "20020101", "20020102"]),
            )
        ],
    )
    p = percentile_doy(da, 90)

    numpy.testing.assert_array_equal(p.data, [0.9, 1.8])
