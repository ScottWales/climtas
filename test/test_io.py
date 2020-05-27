#!/usr/bin/env python
# Copyright 2019 Scott Wales
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

import xarray
import numpy
import dask
import pandas

from climtas import io


def test_to_netcdf_throttled(tmpdir, distributed_client):
    def helper(path, data):
        da = xarray.DataArray(data, dims=["t", "x", "y"], name="test")
        io.to_netcdf_throttled(da, path)
        out = xarray.open_dataset(str(path)).test
        xarray.testing.assert_identical(da, out)

    path = tmpdir / "numpy.nc"
    data = numpy.zeros([10, 10, 10])
    helper(path, data)

    path = tmpdir / "dask.nc"
    data = dask.array.zeros([10, 10, 10])
    helper(path, data)

    data = dask.array.random.random([10, 10, 10]) + numpy.random.random([10, 10, 10])
    helper(path, data)


def test_to_netcdf_throttled_serial(tmpdir):
    def helper(path, data):
        da = xarray.DataArray(data, dims=["t", "x", "y"], name="test")
        io.to_netcdf_throttled(da, path)
        out = xarray.open_dataset(str(path)).test
        xarray.testing.assert_identical(da, out)

    path = tmpdir / "numpy.nc"
    data = numpy.zeros([10, 10, 10])
    helper(path, data)

    path = tmpdir / "dask.nc"
    data = dask.array.zeros([10, 10, 10])
    helper(path, data)


def test_to_netcdf_series(tmpdir):
    path = tmpdir / "data_{start.year}.nc"
    data = xarray.DataArray(numpy.zeros([20]), coords=[('time', pandas.date_range('20010101', freq='MS', periods=20))], name='test')

    io.to_netcdf_series(data, path, groupby="time.year")

    assert (tmpdir / "data_2001.nc").exists()
    assert (tmpdir / "data_2002.nc").exists()

    data.coords['group'] = ('time', [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
    path = tmpdir / "data_{group}.nc"
    io.to_netcdf_series(data, path, groupby="group")
    assert (tmpdir / "data_0.nc").exists()
    assert (tmpdir / "data_1.nc").exists()
