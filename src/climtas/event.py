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
from tqdm.auto import tqdm
import typing as T


def find_events(da: xarray.DataArray, min_duration: int = 1) -> pandas.DataFrame:
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
        min_duration (:class:`int`): Minimum event duration to return

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
        records.append(df[df.event_duration >= min_duration])

    for t in tqdm(range(da.sizes["time"])):
        current_step = numpy.atleast_1d(da.data.take(t, axis=da.get_axis_num("time")))

        # Add the current step
        duration = duration + numpy.where(current_step, 1, 0)

        # End points are where we have an active duration but no event in the current step
        add_events(numpy.logical_and(duration > 0, numpy.logical_not(current_step)))

    # Add events still active at the end
    t += 1
    add_events(duration > 0)

    return pandas.concat(records, ignore_index=True)


def map_events(
    da: xarray.DataArray, events: pandas.DataFrame, func, *args, **kwargs
) -> pandas.DataFrame:
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


def atleastn(da: xarray.DataArray, n: int, dim: str = "time") -> xarray.DataArray:
    """
    Filter to return values with at least n contiguous points around them

    >>> da = xarray.DataArray([0,1.4,0.8,1,-0.1,2.9,0.6], dims=['time'])
    >>> atleastn(da.where(da > 0), 3)
    <xarray.DataArray (time: 7)>
    array([nan, 1.4, 0.8, 1. , nan, nan, nan])
    Dimensions without coordinates: time

    Args:
        da (:class:`xarray.DataArray`): Pre-filtered event values
        n (:class:`int`): Minimum event length
        dim (:class:`str`): Dimension to work on

    Returns:
        :class:`xarray.DataArray` with events from da that are longer than n
        along dimension dim
    """

    def atleastn_helper(array, axis, n):
        count = numpy.zeros_like(numpy.take(array, 0, axis=axis), dtype="i4")
        mask = numpy.empty_like(numpy.take(array, 0, axis=axis), dtype="bool")
        mask = True

        for i in range(array.shape[axis]):
            array_slice = numpy.take(array, i, axis=axis)

            # Increase the count when there is a valid value, reset when there is not
            count = numpy.where(numpy.isfinite(array_slice), count + 1, 0)

            # Add new points when the contiguous count exceeds the threshold
            mask = numpy.where(count >= n, False, mask)

        out_slice = numpy.take(array, array.shape[axis] // 2, axis=axis)
        r = numpy.where(mask, numpy.nan, out_slice)

        return r

    def atleastn_dask_helper(array, axis, n):
        r = dask.array.map_blocks(
            atleastn_helper, array, drop_axis=axis, axis=axis, n=n, dtype=array.dtype
        )
        return r

    if isinstance(da.data, dask.array.Array):
        reducer = atleastn_dask_helper
    else:
        reducer = atleastn_helper

    r = da.rolling({dim: n * 2 - 1}, center=True, min_periods=1).reduce(reducer, n=n)
    return r


def event_coords(da: xarray.DataArray, events: pandas.DataFrame) -> pandas.DataFrame:
    """
    Converts the index values returned by :func:`find_events` to coordinate values

    >>> da = xarray.DataArray([0,1,1,1,0,1,1], coords=('time', pandas.date_range('20010101', periods=7, freq='D'))
    >>> events = find_events(da > 0)
    >>> event_coords(da, events)
       time  event_duration
    0     20010102               3 days
    1     20010106               2 days

    Args:
        da (:class:`xarray.DataArray`): Source data values
        events (:class:`pandas.DataFrame`): Event start & durations, e.g. from
            :func:`find_events`

    Returns:
        :class:`pandas.DataFrame` with the same columns as 'events', but with
        index values converted to coordinates
    """
    coords = {}
    for d in da.dims:
        coords[d] = da[d].values[events[d].values]

    end = da['time'].values[events['time'].values + events['event_duration'].values-1]
    coords['event_duration'] = end - coords['time']

    return pandas.DataFrame(coords, index=events.index)
