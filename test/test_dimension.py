#!/usr/bin/env python
# Copyright 2018 ARC Centre of Excellence for Climate Extremes
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
from __future__ import print_function

from climtas.dimension import *

import pytest
import xarray
import numpy


def test_remove_degenerate_axes():
    a = xarray.DataArray([1, 2], dims=["i"])
    o = remove_degenerate_axes(a)

    numpy.testing.assert_array_equal(a.data, o.data)

    b = xarray.DataArray([[1, 2], [1, 2]], dims=["i", "j"])
    o = remove_degenerate_axes(b)

    numpy.testing.assert_array_equal([1, 2], o.data)


def test_identify_lat_lon():
    da = xarray.DataArray([[0, 0], [0, 0]], coords=[("lat", [0, 1]), ("lon", [0, 1])])

    # Missing CF metadata is an error
    with pytest.raises(Exception):
        lat, lon = identify_lat_lon(da)

    # Should find units, axis or standard_name attributes
    da.lat.attrs["units"] = "degrees_north"
    da.lon.attrs["axis"] = "X"
    lat, lon = identify_lat_lon(da)
    assert lat.equals(da.lat)
    assert lon.equals(da.lon)


def test_identify_time():
    da = xarray.DataArray([0, 0], coords=[("time", [0, 1])])

    # Missing CF metadata is an error
    with pytest.raises(Exception):
        time = identify_time(da)

    # Units should be identified
    da.time.attrs["units"] = "days since 2006-01-09"
    time = identify_time(da)
    assert time.equals(da.time)

    # Units should work with CF decoding
    da = xarray.decode_cf(xarray.Dataset({"da": da})).da
    time = identify_time(da)
    assert time.equals(da.time)
