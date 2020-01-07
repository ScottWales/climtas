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

"""Functions for locating and analysing 'events' within a dataset

Locate where events are with :func:`find_events`, then analyse them with
:func:`map_events()` to create a :class:`pandas.DataFrame`.
"""

import numpy
import dask
import pandas
import xarray


def find_events(da):
    """Find 'events' in a DataArray mask

    Events are defined as being active when the array value is truthy. You
    should generally pass in the results of a comparison against some kind of
    threshold

    >>> da = xarray.DataArray([0,1,1,1,0,1,1], dims=['time'])
    >>> find_events(da > 0)
       time  event_duration
    0     1               3
    1     5               2

    It's assumed that events are reasonably sparse for large arrays

    Args:
        da (:class:`xarray.DataArray`): Input mask, valid when an event is
            active. Must have a 'time' dimension, dtype is expected to be bool
            (or something else that is truthy when an event is active)

    Returns:
        A :class:`pandas.DataFrame` containing event start points and
        durations. This will contain columns for each dimension in da, as well
        as an 'event_duration' column
    """

    duration = numpy.atleast_1d(numpy.zeros_like(da.isel(time=0), dtype="i4"))

    columns = ["time", *[d for d in da.dims if d != "time"], "event_duration"]
    records = []

    def add_events(locations):
        end_locations = numpy.nonzero(locations)
        end_durations = duration[end_locations]
        start_times = t - end_durations

        # Reset events that have ended
        duration[end_locations] = 0

        if len(end_durations) == 0:
            return

        if len(columns) == 2:
            # 1d input dataset
            data = numpy.stack([start_times, end_durations], axis=1)
        else:
            data = numpy.concatenate(
                [start_times[None, :], end_locations, end_durations[None, :]], axis=0
            ).T

        df = pandas.DataFrame(data=data, columns=columns)
        records.append(df)

    for t in range(da.sizes["time"]):
        current_step = numpy.atleast_1d(da.isel(time=t))

        # Add the current step
        duration += numpy.where(current_step, 1, 0)

        # End points are where we have an active duration but no event in the current step
        add_events(numpy.logical_and(duration > 0, numpy.logical_not(current_step)))

    # Add events still active at the end
    t += 1
    add_events(duration > 0)

    return pandas.concat(records, ignore_index=True)


def map_events(da, events, func, *args, **kwargs):
    """Map a function against multiple events

    The output is the value from func evaluated at each of the events. Events
    should at a minimum have columns for each coordinate in da as well as an
    'event_duration' column that records how long each event is, as is returned by
    :func:`find_events`:

    >>> da = xarray.DataArray([0,1,1,1,0,1,1], dims=['time'])
    >>> events = find_events(da > 0)
    >>> map_events(da, events, lambda x: x.sum().item())
    0    3
    1    2
    dtype: int64

    You may wish to filter the events DataFrame first to combine close events or to
    remove very short events.

    If func returns a dict results will be converted into columns. This will be
    more efficient than running map_events once for each operation:

    >>> da = xarray.DataArray([0,1,1,1,0,1,1], dims=['time'])
    >>> events = find_events(da > 0)
    >>> map_events(da, events, lambda x: {'mean': x.mean().item(), 'std': x.std().item()})
       mean  std
    0   1.0  0.0
    1   1.0  0.0

    :meth:`pandas.DataFrame.join` can be used to link up the results with their
    corresponding coordinates:

    >>> da = xarray.DataArray([0,1,1,1,0,1,1], dims=['time'])
    >>> events = find_events(da > 0)
    >>> sums = map_events(da, events, lambda x: {'sum': x.sum().item()})
    >>> events.join(sums)
       time  event_duration  sum
    0     1               3    3
    1     5               2    2

    Args:
        da (:class:`xarray.DataArray`): Source data values
        events (:class:`pandas.DataFrame`): Event start & durations, e.g. from
            :func:`find_events`
        func ((:class:`xarray.DataArray`, \*args, \*\*kwargs) -> Dict[str, Any]): Function to apply to each event
        \*args, \*\*kwargs: Passed to func

    Returns:
        :class:`pandas.DataFrame` with each row the result of applying func to
        the corresponding event row. Behaves like
        :meth:`pandas.DataFrame.apply` with result_type='expand'
    """

    def map_func(e):
        coords = {k: e.loc[k] for k in da.dims}
        coords["time"] = slice(coords["time"], coords["time"] + e["event_duration"])

        values = da.isel(coords)
        return func(values, *args, **kwargs)

    return events.apply(map_func, axis="columns", result_type="expand")
