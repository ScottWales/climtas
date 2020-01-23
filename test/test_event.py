#!/usr/bin/env python
# Copyright 2020 ARC Centre of Excellence for Climate Extremes
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

from climtas.event import *

import xarray


def test_find_events():
    da = xarray.DataArray([[0, 1, 1, 1, 0]], dims=["x", "time"])
    events = find_events(da > 0)

    events = events.set_index(["time", "x"])
    assert events.index[0][0] == 1
    assert events.index[0][1] == 0
    assert events["event_duration"].iloc[0] == 3
    assert events["event_duration"].loc[1, 0] == 3
    assert len(events) == 1

    da = xarray.DataArray([[0, 1, 1, 1, 0], [1, 1, 0, 1, 1]], dims=["x", "time"])
    events = find_events(da > 0)

    events = events.set_index(["time", "x"])
    assert events["event_duration"].loc[1, 0] == 3
    assert events["event_duration"].loc[0, 1] == 2
    assert events["event_duration"].loc[3, 1] == 2
    assert len(events) == 3

    da = xarray.DataArray([[0, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dims=["x", "time"])
    events = find_events(da > 0)

    events = events.set_index(["time", "x"])
    assert events["event_duration"].loc[1, 0] == 3
    assert events["event_duration"].loc[0, 1] == 4
    assert len(events) == 2

    da = xarray.DataArray([[0, 1, 1, 1, 0], [1, 1, 0, 1, 1]], dims=["x", "time"])
    events = find_events(da > 0, min_duration=3)

    events = events.set_index(["time", "x"])
    assert events["event_duration"].loc[1, 0] == 3
    assert len(events) == 1


def test_find_events_1d():
    da = xarray.DataArray([0, 1, 1, 1, 0], dims=["time"])
    events = find_events(da > 0)

    events = events.set_index(["time"])
    assert events["event_duration"].loc[1] == 3
    assert len(events) == 1


def test_map_events():
    da = xarray.DataArray([0, 1, 1, 1, 0], dims=["time"])
    events = find_events(da > 0)

    sums = map_events(da, events, lambda x: x.sum())
    assert sums.iloc[0] == 3


def test_atleastn():
    sample = [[0, 1, 1, 1, 0, 1, 1, 1, 1],
              [1, 0, 1, 1, 0, 0, 0, 0, 1]]

    expect = [[0, 1, 1, 1, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]]
    expect = numpy.array(expect)
    expect = numpy.where(expect > 0, expect, numpy.nan)

    da =  xarray.DataArray( sample, dims=["x","time"])
    #filtered = atleastn(da.where(da > 0), 3)
    #numpy.testing.assert_array_equal(filtered, expect)

    da = da.chunk({'x': 2})
    filtered = atleastn(da.where(da > 0), 3)
    numpy.testing.assert_array_equal(filtered, expect)
