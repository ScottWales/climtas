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

import numpy
import dask

"""Helper functions

These functions are low-level, and mainly intended for internal use
"""


def chunk_count(da):
    """
    Returns the number of chunks in the dataset
    """
    return numpy.prod([len(c) for c in da.chunks])


def chunk_size(da):
    """
    Returns the size of the first dask chunk in the dataset
    """
    return numpy.prod(da.data.chunksize) * da.data.itemsize


def graph_size(da):
    """
    Returns number of nodes in the dask graph
    """
    return len(da.__dask_graph__())


def dask_report(da):
    """
    Print info about a dask array
    """
    print("Chunk Count:", chunk_count(da))
    print("Chunk Size:", dask.utils.format_bytes(chunk_size(da)))
    print("Graph Size:", graph_size(da))


def optimized_dask_get(graph, keys, optimizer=None, sync=True):
    """
    Compute a dask low-level graph with some optimization
    """
    try:
        client = dask.distributed.get_client()
    except ValueError:
        client = dask

    if optimizer:
        graph, _ = optimizer(graph, keys)
    else:
        graph, _ = dask.optimization.cull(graph, keys)
        graph, _ = dask.optimization.fuse(graph, keys)

    return client.get(graph, keys, sync=sync)


def throttle_futures(graph, key_list, optimizer=None, max_tasks=None):
    """
    Run futures in parallel, with a maximum of 'max_tasks' at once

    Args:
        graph: Dask task graph
        key_list: Iterable of keys from 'graph' to compute
        max_tasks: Maximum number of tasks to run at once (if none use the
            number of workers)
    """
    try:
        client = dask.distributed.get_client()
    except ValueError:
        # No cluster, run in serial
        return [optimized_dask_get(graph, k) for k in key_list]

    futures = []
    keys = iter(key_list)

    if max_tasks is None:
        max_tasks = len(client.cluster.workers) + 1

    # Build up initial max_tasks future list
    for i in range(min(max_tasks, len(key_list))):
        futures.append(
            optimized_dask_get(graph, next(keys), optimizer=optimizer, sync=False)
        )

    # Add new futures as the existing ones are completed
    ac = dask.distributed.as_completed(futures, with_results=True)
    results = []
    for f, result in ac:
        try:
            ac.add(
                optimized_dask_get(graph, next(keys), optimizer=optimizer, sync=False)
            )
        except StopIteration:
            pass
        results.append(result)

    return results


def apply_by_dayofyear(da, func, dim="time", **kwargs):
    """
    Group da by 'time.dayofyear', then apply 'func' to each grouping before
    expanding back to a timeseries

    Rechunks the data to avoid excessive Dask chunks
    """

    def chunk_apply_by_dayofyear(x):
        # Xarray tests the return shape of the function by calling it with a
        # size 0 array, we don't change the shape
        if x.size == 0:
            return x

        axis = x.get_axis_num(dim)
        group = x.groupby(f"{dim}.dayofyear")
        ranking = group.map(func, shortcut=True, axis=axis, **kwargs)

        return ranking

    time_chunked = da.chunk({dim: None})
    ranking = time_chunked.map_blocks(chunk_apply_by_dayofyear)

    return ranking


def apply_by_monthday(da, func, dim="time", **kwargs):
    """
    Group da by ('time.month', 'time.dayofyear'), then apply 'func' to each
    grouping before expanding back to a timeseries

    Rechunks the data to avoid excessive Dask chunks
    """

    def chunk_apply_by_monthday(x):
        # Xarray tests the return shape of the function by calling it with a
        # size 0 array, we don't change the shape
        if x.size == 0:
            return x

        monthday = x[dim].dt.month * 100 + x[dim].dt.day
        x.coords["monthday"] = monthday

        axis = x.get_axis_num(dim)
        group = x.groupby("monthday")
        ranking = group.map(func, shortcut=True, axis=axis, **kwargs)

        return ranking

    time_chunked = da.chunk({dim: None})
    ranking = time_chunked.map_blocks(chunk_apply_by_monthday)

    return ranking
