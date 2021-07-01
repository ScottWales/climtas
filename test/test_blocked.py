#!/usr/bin/env python
# Copyright 2020 Scott Wales
# author: Scott Wales <scott.wales@unimelb.edu.au>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from climtas.blocked import *
import xarray
import pandas
import dask.array
from climtas.helpers import *
import pytest


@pytest.fixture(params=["daily", "daily_2d", "daily_dask", "daily_dask_2d"])
def sample(request):
    time = pandas.date_range("20020101", "20050101", freq="D", closed="left")

    samples = {
        "daily": xarray.DataArray(
            numpy.random.random(time.size), coords=[("time", time)]
        ),
        "daily_2d": xarray.DataArray(
            numpy.random.random((time.size, 10)),
            coords=[("time", time), ("x", numpy.arange(10))],
        ),
        "daily_dask": xarray.DataArray(
            dask.array.random.random(time.size), coords=[("time", time)]
        ),
        "daily_dask_2d": xarray.DataArray(
            dask.array.random.random((time.size, 10), chunks=(100, 5)),
            coords=[("time", time), ("x", numpy.arange(10))],
        ),
    }

    return samples[request.param]


@pytest.fixture(params=["daily", "daily_dask"])
def sample_hr(request):
    time = pandas.date_range("20020101", "20050101", freq="H", closed="left")

    samples = {
        "daily": xarray.DataArray(
            numpy.random.random(time.size), coords=[("time", time)]
        ),
        "daily_dask": xarray.DataArray(
            dask.array.random.random(time.size), coords=[("time", time)]
        ),
    }

    return samples[request.param]


def test_groupby_dayofyear(sample):
    time = pandas.date_range("20020101", "20050101", freq="D", closed="left")
    daily = xarray.DataArray(numpy.random.random(time.size), coords=[("time", time)])

    blocked_doy = blocked_groupby(sample, time="dayofyear")
    xarray_doy = sample.groupby("time.dayofyear")

    for op in "min", "max", "mean", "sum":
        xarray.testing.assert_equal(
            getattr(blocked_doy, op)(), getattr(xarray_doy, op)()
        )

    sample = sample.sel(time=slice("20020101", "20031231"))

    blocked_doy = blocked_groupby(sample, time="dayofyear")
    xarray_doy = sample.groupby("time.dayofyear")

    for op in "min", "max", "mean", "sum":
        xarray.testing.assert_equal(
            getattr(blocked_doy, op)()[0:365], getattr(xarray_doy, op)()
        )


def test_groupby_dayofyear_dask():
    time = pandas.date_range("20020101", "20050101", freq="D", closed="left")
    daily = xarray.DataArray(
        dask.array.zeros(time.size, chunks=50), coords=[("time", time)]
    )

    blocked_doy_max = blocked_groupby(daily, time="dayofyear").max()
    xarray_doy_max = daily.groupby("time.dayofyear").max()

    # We should be making less chunks than xarray's default
    assert chunk_count(blocked_doy_max) <= 0.1 * chunk_count(xarray_doy_max)

    # We should be have a less complex graph than xarray's default
    assert graph_size(blocked_doy_max) <= 0.2 * graph_size(xarray_doy_max)


def test_groupby_monthday(sample):
    blocked_doy = blocked_groupby(sample, time="monthday")

    sample.coords["monthday"] = sample.time.dt.month * 100 + sample.time.dt.day
    xarray_doy = sample.groupby("monthday")

    for op in "min", "max", "mean", "sum":
        numpy.testing.assert_array_equal(
            getattr(blocked_doy, op)(), getattr(xarray_doy, op)()
        )


def test_groupby_monthday_dask():
    time = pandas.date_range("20020101", "20050101", freq="D", closed="left")
    daily = xarray.DataArray(
        dask.array.zeros(time.size, chunks=50), coords=[("time", time)]
    )

    blocked_doy_max = blocked_groupby(daily, time="monthday").max()

    daily.coords["monthday"] = daily.time.dt.month * 100 + daily.time.dt.day
    xarray_doy_max = daily.groupby("monthday").max()

    # We should be making less chunks than xarray's default
    assert chunk_count(blocked_doy_max) <= 0.1 * chunk_count(xarray_doy_max)

    # We should be have a less complex graph than xarray's default
    assert graph_size(blocked_doy_max) <= 0.2 * graph_size(xarray_doy_max)


def test_groupby_climatology(sample):

    climatology = blocked_groupby(sample, time="dayofyear").mean()
    delta = blocked_groupby(sample, time="dayofyear") - climatology

    climatology_xr = sample.groupby("time.dayofyear").mean()
    delta_xr = sample.groupby("time.dayofyear") - climatology_xr

    # Climtas matches Xarray
    numpy.testing.assert_array_equal(delta_xr, delta)

    climatology_md = blocked_groupby(sample, time="monthday").mean()
    delta_md = blocked_groupby(sample, time="monthday") - climatology_md

    # January anomalies match
    numpy.testing.assert_array_equal(delta_md[:31, ...], delta[:31, ...])

    assert type(delta.data) == dask.array.Array
    assert type(delta_md.data) == dask.array.Array


def test_groupby_percentile(sample):
    climatology = blocked_groupby(sample, time="dayofyear").percentile(90)

    climatology_xr = (
        sample.load().groupby("time.dayofyear").reduce(numpy.percentile, q=90)
    )

    numpy.testing.assert_array_equal(climatology_xr[:-1], climatology[:-1])


def test_groupby_apply(sample):
    if sample.ndim == 2:
        # Only testing 1d
        return

    blocked_double = blocked_groupby(sample, time="dayofyear").apply(lambda x: x * 2)
    xarray.testing.assert_equal(sample * 2, blocked_double)

    sample = sample.load()
    sample[:] = sample.time.dt.year[:]
    blocked_rank = blocked_groupby(sample, time="dayofyear").rank()

    assert blocked_rank.sel(time="20020101").values.item() == 1
    assert blocked_rank.sel(time="20030101").values.item() == 2
    assert blocked_rank.sel(time="20040101").values.item() == 3


def test_resample_safety(sample):
    if sample.ndim == 2:
        # Only testing 1d
        return

    # Not a coordinate
    sliced = sample
    with pytest.raises(Exception):
        blocked_resample(sliced, x=24)

    # Samples doesn't evenly divide length
    sliced = sample[0:15]
    with pytest.raises(Exception):
        blocked_resample(sliced, time=24)

    # Irregular
    sliced = xarray.concat([sample[0:15], sample[17:26]], dim="time")
    assert sliced.size == 24
    with pytest.raises(Exception):
        blocked_resample(sliced, time=24)


def test_resample(sample_hr):
    expected = sample_hr.resample(time="D").mean()

    result = blocked_resample(sample_hr, time=24).mean()
    xarray.testing.assert_equal(expected, result)
    xarray.testing.assert_identical(expected, result)

    result = blocked_resample(sample_hr, time="D").mean()
    xarray.testing.assert_identical(expected, result)

    result = blocked_resample(sample_hr, {"time": "D"}).mean()
    xarray.testing.assert_identical(expected, result)


def test_groupby_safety(sample):
    # Not a coordinate
    sliced = sample
    with pytest.raises(Exception):
        blocked_groupby(sliced, x="dayofyear")

    # Samples don't cover a full year
    sliced = sample[1:365]
    with pytest.raises(Exception):
        blocked_groupby(sliced, time="dayofyear")

    sliced = sample[0:364]
    with pytest.raises(Exception):
        blocked_groupby(sliced, time="dayofyear")

    sliced = xarray.concat([sample[0:15], sample[17:365]], dim="time")
    with pytest.raises(Exception):
        blocked_groupby(sliced, time="dayofyear")


def test_percentile(sample):
    a = approx_percentile(sample, 90, dim="time")

    if isinstance(sample.data, dask.array.Array):
        a = a[0, ...]

        b = numpy.zeros(sample[0].shape)
        for ii in numpy.ndindex(b.shape):
            b[ii] = dask.array.percentile(
                sample.data[
                    numpy.s_[
                        :,
                    ]
                    + ii
                ],
                90,
            )[0]
    else:
        b = numpy.percentile(
            sample.data,
            90,
            axis=0,
        )

    numpy.testing.assert_array_equal(a, b)


def test_dask_approx_percentile():
    data = numpy.random.random((500, 100))
    data[0, 0] = 0
    data[0, 1] = 1
    data[0, 2] = numpy.nan

    sample = dask.array.from_array(data, chunks=(100, 10))

    a = dask_approx_percentile(sample, 90, axis=0, skipna=False)

    a = a[0, ...]

    # Compare with applying dask.percentile along the time axis
    b = numpy.zeros(sample[0].shape)
    for ii in numpy.ndindex(b.shape):
        b[ii] = dask.array.percentile(
            sample[
                numpy.s_[
                    :,
                ]
                + ii
            ],
            90,
        )[0]

    numpy.testing.assert_array_equal(a, b)

    # Test values are close to numpy with chunking
    sample = dask.array.from_array(data, chunks=(100, 10))

    # skipna=False works like numpy.percentile
    a = dask_approx_percentile(sample, 90, axis=0, skipna=False)
    b = numpy.percentile(data, 90, axis=0)

    numpy.testing.assert_allclose(a[0, ...], b, rtol=0.1)

    # skipna=True works like numpy.nanpercentile
    a = dask_approx_percentile(sample, 90, axis=0, skipna=True)
    b = numpy.nanpercentile(data, 90, axis=0)

    numpy.testing.assert_allclose(a[0, ...], b, rtol=0.1)

    # Test values match numpy without chunking
    sample = dask.array.from_array(data, chunks=data.shape)

    # skipna=True works like numpy.nanpercentile
    a = dask_approx_percentile(sample, 90, axis=0, skipna=True)
    b = numpy.nanpercentile(data, 90, axis=0)

    numpy.testing.assert_array_equal(a[0, ...], b)

    # skipna=False works like numpy.percentile
    a = dask_approx_percentile(sample, 90, axis=0, skipna=False)
    b = numpy.percentile(data, 90, axis=0)

    numpy.testing.assert_array_equal(a[0, ...], b)
