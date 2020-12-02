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

from dask.dataframe.core import new_dd_object
from dask.delayed import tokenize
import numpy
import dask
import dask.dataframe
import pandas
import xarray
import typing as T
import sparse
from .helpers import map_blocks_to_delayed


def find_events(
    da: xarray.DataArray, min_duration: int = 1, use_dask: bool = None
) -> pandas.DataFrame:
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

    If `use_dask` is True or `use_dask` is None and the source dataset is only chunked
    horizontally then events will be searched for in each Dask chunk and the
    results aggregated afterward.

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

    chunks = getattr(da, "chunks", None)

    if use_dask is None and chunks is not None:
        # Decide whether to apply on blocks
        time_ax = da.get_axis_num("time")

        if len(chunks[time_ax]) > 1:
            use_dask = False

    if chunks is None or use_dask is False:
        events = find_events_block(
            da, min_duration=min_duration, offset=(0,) * da.ndim, load=False
        )

    else:
        events_map = map_blocks_to_delayed(
            da, find_events_block, kwargs={"min_duration": min_duration}
        )

        events_map = dask.compute(events_map)[0]

        events = join_events(
            [events for offset, events in events_map],
            offsets=[offset for offset, events in events_map],
            dims=da.dims,
        )

    # Final cleanup of events at the start/end of chunks
    return events[events.event_duration >= min_duration]


def find_events_block(
    da: xarray.DataArray,
    offset: T.Iterable[int],
    min_duration: int = 1,
    load: bool = True,
) -> pandas.DataFrame:
    """Find 'events' in a section of a DataArray

    See :func:`find_events`

    Intended to run on a Dask block, with the results from each block merged.

    If an event is active at the first or last timestep it is returned
    regardless of its duration, so it can be joined with neighbours
    """

    if load:
        da = da.load()

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

        # Add an event if:
        #    * duration >= min_duration
        #    * start_times == 0
        #    * t == da.sizes['time']

        if t == da.sizes["time"]:
            records.append(df)
        else:
            records.append(
                df[numpy.logical_or(df.event_duration >= min_duration, df.time == 0)]
            )

    t = 0
    for t in range(da.sizes["time"]):
        current_step = numpy.atleast_1d(
            numpy.take(da.data, t, axis=da.get_axis_num("time"))
        )

        try:
            current_step = current_step.compute()
        except:
            pass

        # Add the current step
        duration = duration + numpy.where(current_step, 1, 0)

        # End points are where we have an active duration but no event in the current step
        add_events(numpy.logical_and(duration > 0, numpy.logical_not(current_step)))

    # Add events still active at the end
    t += 1
    add_events(duration > 0)

    if len(records) > 0:
        result = pandas.concat(records, ignore_index=True)
    else:
        result = pandas.DataFrame(None, columns=columns, dtype="int64")

    for d, off in zip(da.dims, offset):
        result[d] += off

    return result


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
        axis = axis[0]
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

    >>> da = xarray.DataArray([0,1,1,1,0,1,1], coords=[('time', pandas.date_range('20010101', periods=7, freq='D'))])
    >>> events = find_events(da > 0)
    >>> event_coords(da, events)
            time event_duration
    0 2001-01-02         3 days
    1 2001-01-06            NaT

    If 'events' has an 'event_duration' column this will be converted to a time
    duration. If the event goes to the end of the data the duration is marked
    as not a time, as the end date is unknown.

    Args:
        da (:class:`xarray.DataArray`): Source data values
        events (:class:`pandas.DataFrame`): Event start & durations, e.g. from
            :func:`find_events` or :func:`extend_events`

    Returns:
        :class:`pandas.DataFrame` with the same columns as 'events', but with
        index values converted to coordinates
    """
    coords = {}
    for d in da.dims:
        coords[d] = da[d].values[events[d].values]

    if "event_duration" in events:
        end_index = events["time"].values + events["event_duration"].values
        end = da["time"].values[numpy.clip(end_index, 0, da.sizes["time"] - 1)]
        coords["event_duration"] = end - coords["time"]
        coords["event_duration"][end_index >= da.sizes["time"]] = numpy.timedelta64(
            "nat"
        )

    return pandas.DataFrame(coords, index=events.index)


def extend_events(events: pandas.DataFrame):
    """
    Extend the 'events' DataFrame to hold indices for the full event duration

    :func:`find_events` returns only the start index of events. This will
    extend the DataFrame to cover the indices of the entire event. In addition
    to the indices a column 'event_id' gives the matching index in 'event' for
    the row

    >>> da = xarray.DataArray([0,1,1,1,0,1,1], coords=[('time', pandas.date_range('20010101', periods=7, freq='D'))])
    >>> events = find_events(da > 0)
    >>> extend_events(events)
       time  event_id
    0     1         0
    1     2         0
    2     3         0
    3     5         1
    4     6         1

    Args:
        da (:class:`xarray.DataArray`): Source data values
        events (:class:`pandas.DataFrame`): Event start & durations, e.g. from
            :func:`find_events`
    """

    def extend_row(row):
        repeat = numpy.repeat(row.values[None, :], row["event_duration"], axis=0)

        df = pandas.DataFrame(repeat, columns=row.index)

        df["time"] = row["time"] + numpy.arange(row["event_duration"])
        df["event_id"] = row.name
        del df["event_duration"]

        return df

    return pandas.concat(
        events.apply(extend_row, axis="columns").values, ignore_index=True
    )


def event_da(
    da: xarray.DataArray, events: pandas.DataFrame, values: numpy.ndarray
) -> xarray.DataArray:
    """
    Create a :obj:`xarray.DataArray` with 'values' at event locations

    Args:
        da (:class:`xarray.DataArray`): Source data values
        events (:class:`pandas.DataFrame`): Index values, e.g. from
            :func:`find_events` or :func:`extend_events`
        values (:class:`numpy.ndarray`-like): Value to give to each location specified by event

    Returns:
        :obj:`xarray.DataArray` with the same axes as `da` and values at
        `event` given by `values`.g
    """

    s = sparse.COO(
        [events[d] for d in da.dims], values, shape=da.shape, fill_value=numpy.nan
    )

    # It's more helpful to have a dense array, but let's only convert
    # what we need when we need it using dask

    # Add dask chunking
    try:
        chunks = da.data.chunks
    except AttributeError:
        chunks = "auto"
    d = dask.array.from_array(s, chunks=chunks)

    # Add an operation that converts chunks to dense when needed
    def dense_block(b):
        return b.todense()

    dense = d.map_blocks(dense_block, dtype=d.dtype)

    # Use the input data's coordinates
    return xarray.DataArray(dense, coords=da.coords)


def join_events(
    events: T.List[pandas.DataFrame],
    offsets: T.List[T.List[int]] = None,
    dims: T.Tuple[T.Hashable, ...] = None,
) -> pandas.DataFrame:
    """
    Join consecutive events in multiple dataframes

    The returned events will be in an arbitrary order, the index may not match
    entries from the input dataframes.

    >>> events = [
    ...    pandas.DataFrame([[1, 2]], columns=["time", "event_duration"]),
    ...    pandas.DataFrame([[3, 1]], columns=["time", "event_duration"]),
    ... ]

    >>> join_events(events)
       time  event_duration
    0     1               3

    Args:
        events: List of results from :func:`find_events`

    Returns:
        :obj:`pandas.DataFrame` where results that end when the next event
        starts are joined together
    """

    if offsets is not None:
        # Join like offsets
        if dims is None:
            raise Exception("Must supply dims with offsets")

        df = pandas.DataFrame(offsets, columns=dims)
        df["events"] = events

        results = []

        spatial_dims = [d for d in dims if d != "time"]

        for k, g in df.groupby(spatial_dims, sort=False):
            results.append(dask.delayed(join_events)(g.events.values))

        results = dask.compute(results)[0]

        return pandas.concat(results, ignore_index=True)

    results = []
    for i in range(0, len(events) - 1):
        next_t0 = events[i + 1]["time"].min()

        stop = events[i]["time"] + events[i]["event_duration"]

        results.append(events[i][stop < next_t0])

        events[i + 1] = _merge_join(events[i][stop >= next_t0], events[i + 1])

    results.append(events[-1])

    return pandas.concat(results)


def _single_gridpoint_join(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Join events on a single gridpoint
    """

    i_start = 0

    for i in range(1, df.shape[0]):
        last_end = df.iloc[i_start].time + df.iloc[i_start].event_duration

        if last_end == df.iloc[i].time:
            # Last event ends when this one starts, merge
            df.iloc[i_start].event_duration += df.iloc[i].event_duration
            df.iloc[i].event_duration = 0
        else:
            i_start = i

    return df[df.event_duration > 0]


def _merge_join(df_x: pandas.DataFrame, df_y: pandas.DataFrame) -> pandas.DataFrame:
    """
    Join events on time == time + event_duration
    """

    spatial_dims = list(df_x.columns[1:-1])

    df_x = df_x.copy()
    df_x["stop"] = df_x["time"] + df_x["event_duration"]

    # First step of df_y
    t0 = df_y["time"].min()

    # Merge df_x with the first step from df_y
    merged = df_x.merge(
        df_y[df_y["time"] == t0],
        left_on=["stop", *spatial_dims],
        right_on=["time", *spatial_dims],
        how="outer",
    )

    merged["time_x"].fillna(merged["time_y"], inplace=True, downcast="infer")
    merged["event_duration_x"].fillna(0, inplace=True, downcast="infer")
    merged["event_duration_y"].fillna(0, inplace=True, downcast="infer")

    result = pandas.DataFrame()
    result["time"] = merged["time_x"]
    for d in spatial_dims:
        result[d] = merged[d]
    result["event_duration"] = merged["event_duration_x"] + merged["event_duration_y"]

    return pandas.concat([result, df_y[df_y["time"] > t0]], ignore_index=True)


def event_values(
    da: xarray.DataArray, events: pandas.DataFrame, use_dask=True
) -> dask.dataframe.DataFrame:
    """
    Gets the values from da where an event is active

    Note that for a large dataset with many events this can consume a
    considerable amount of memory. `use_dask=True` will return an uncomputed
    :obj:`dask.dataframe.DataFrame` instead of a :obj:`pandas.DataFrame`,
    potentially saving memory. You can compute the results later with
    :meth:`dask.dataframe.DataFrame.compute()`.

    >>> da = xarray.DataArray(
    ...     [0,3,6,2,0,8,6],
    ...     coords=[('time',
    ...              pandas.date_range('20010101', periods=7, freq='D'))])
    >>> events = find_events(da > 0)
    >>> event_values(da, events)
       time  event_id  value
    0     1         0      3
    1     2         0      6
    2     3         0      2
    3     5         1      8
    4     6         1      6

    See the Dask DataFrame documentation for performance information. In general basic
    reductions (min, mean, max, etc.) should be fast.

    >>> values = event_values(da, events)
    >>> values.groupby('event_id').value.mean()
    event_id
    0    3.666667
    1    7.000000
    Name: value, dtype: float64

    For custom aggregations setting an index may help performance. This sorts the data though,
    so may use a lot of memory

    >>> values = values.set_index('event_id')
    >>> values.groupby('event_id').value.apply(lambda x: x.min())
    event_id
    0    2
    1    6
    Name: value, dtype: int64

    Args:
        da (:class:`xarray.DataArray`): Source data values
        events (:class:`pandas.DataFrame`): Event start & durations, e.g. from
            :func:`find_events`
        use_dask (bool): If true, returns an uncomputed Dask DataFrame. If false,
            computes the values before returning

    Returns:
        :obj:`dask.dataframe.DataFrame` with columns event_id, time, value
    """

    chunks = getattr(da, "chunks", None)

    if chunks is None:
        print("no chunks")
        return event_values_block(da, events, offset=(0,) * da.ndim)

    print("map")
    events_map = map_blocks_to_delayed(
        da, event_values_block, kwargs={"events": events}, name="event_values_block"
    )

    if not use_dask:
        print("no dask")
        events_map = dask.compute(events_map)[0]
        df = pandas.concat([e for o, e in events_map])

    else:
        print("Using dask")
        # df = dask.dataframe.from_delayed(
        #     [e for o, e in events_map],
        #     meta=[("time", "int"), ("event_id", "int"), ("value", da.dtype)],
        #     verify_meta=False,
        # )

        dfs = [e for o, e in events_map]

        name = "from-delayed" + "-" + tokenize(*dfs)

        dsk = {}
        dsk.update(dfs[0].dask)
        for i in range(1, len(dfs)):
            print(dfs[i].dask)
            dsk.update(
                {
                    k: v
                    for k, v in dfs[i].dask.items()
                    if k[0].startswith("event_values_block")
                }
            )

        meta = [("time", "int"), ("event_id", "int"), ("value", da.dtype)]

        meta = pandas.DataFrame({k: pandas.Series([], dtype=v) for k, v in meta})

        df = new_dd_object(dsk, name, meta, [None] * (len(dfs) + 1))

    return df


def event_values_block(
    da: xarray.DataArray,
    events: pandas.DataFrame,
    offset: T.Union[T.Iterable[int], T.Dict[T.Hashable, int]],
    load: bool = True,
):
    """
    Gets the values from da where an event is active
    """
    print("values called")

    if load:
        da = da.load()

    event_id = numpy.atleast_1d(-1 * numpy.ones(da.isel(time=0).shape, dtype="i"))
    event_duration = numpy.atleast_1d(-1 * numpy.ones(da.isel(time=0).shape, dtype="i"))

    times = []
    event_ids = []
    event_values = []

    if not isinstance(offset, dict):
        offset = dict(zip(da.dims, offset))

    t_offset = offset["time"]

    events = filter_block(da, events, offset)

    # Events that started before this block
    active_events = events[
        (events["time"] < t_offset)
        & (events["time"] + events["event_duration"] >= t_offset)
    ]

    if active_events.size > 0:
        indices = tuple(
            [active_events[c] - offset[c] for c in active_events.columns[1:-1]]
        )
        event_id[indices] = active_events.index.values
        event_duration[indices] = (
            active_events.time + active_events.event_duration - t_offset
        )

    for t in range(da.sizes["time"]):
        # Add newly active events
        active_events = events[events["time"] == t + t_offset]

        if active_events.size > 0:

            # Indices in the block of current events
            indices = tuple(
                [
                    active_events[c].values - offset[c]
                    for c in active_events.columns[1:-1]
                ]
            )

            # Set values at the current events
            event_id[indices] = active_events.index.values
            event_duration[indices] = active_events.event_duration

        # Currently active coordinates
        indices = numpy.where(event_duration > 0)

        # Get values
        event_ids.append(event_id[indices])
        event_values.append(numpy.atleast_1d(da.isel(time=t).values)[indices])
        times.append(numpy.ones_like(event_ids[-1]) * (t + t_offset))

        # Decrease durations
        event_duration -= 1

    # times = numpy.concatenate(times)
    # event_ids = numpy.concatenate(event_ids)
    # event_values = numpy.concatenate(event_values)

    if len(times) > 0:
        result = numpy.block([times, event_ids, event_values])
    else:
        return None

    df = pandas.DataFrame(result, index=["time", "event_id", "value"]).T
    return df


def filter_block(
    da: xarray.DataArray, events: pandas.DataFrame, offset: T.Dict[T.Hashable, int]
) -> pandas.DataFrame:
    """
    Filters events to within the current block horizontally
    """
    for i, d in enumerate(da.dims):
        if d != "time":
            events = events[
                (offset[d] <= events[d]) & (events[d] < offset[d] + da.shape[i])
            ]

    return events
