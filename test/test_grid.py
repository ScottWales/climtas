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

from climtas.grid import *
import xarray
import numpy
import tempfile


def test_latlon_grid_to_scrip():
    d = xarray.DataArray(
        data=numpy.ones((2, 4)), coords=[("lat", [-45, 45]), ("lon", [0, 90, 180, 270])]
    )
    d.lat.attrs["units"] = "degrees_north"
    d.lon.attrs["units"] = "degrees_east"

    center_lon, center_lat = numpy.meshgrid(d.lon, d.lat)
    d[:, :] = center_lon

    s = identify_grid(d).to_scrip()

    assert s.grid_dims[0] == 4
    assert s.grid_dims[1] == 2

    # Bottom left corner of bottom left cell
    assert s.grid_corner_lat[0, 0] == -90
    assert s.grid_corner_lon[0, 0] == -45

    # Top left corner of bottom left cell
    assert s.grid_corner_lat[0, 3] == 0
    assert s.grid_corner_lon[0, 3] == -45
