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


def optimized_dask_get(graph, keys):
    """
    Compute a dask low-level graph with some optimization
    """
    try:
        client = dask.distributed.get_client()
    except ValueError:
        client = dask

    graph, _ = dask.optimization.cull(graph, keys)
    graph, _ = dask.optimization.fuse(graph, keys)

    return client.get(graph, keys)

def apply_by_dayofyear(da, func, **kwargs):
    """
    Group da by 'time.dayofyear', then apply 'func' to each grouping before
    expanding back to a timeseries

    Rechunks the data to avoid excessive Dask chunks
    """
    def group_helper(x):
        # Xarray tests the return shape of the function by calling it with a
        # size 0 array, we don't change the shape
        if x.size == 0:
            return x

        group = x.groupby("time.dayofyear")
        ranking = group.map(func, shortcut=True, **kwargs)

        return ranking

    time_chunked = da.chunk({"time": None})
    ranking = time_chunked.map_blocks(group_helper)

    return ranking

def apply_by_monthday(da, func, **kwargs):
    """
    Group da by ('time.month', 'time.dayofyear'), then apply 'func' to each
    grouping before expanding back to a timeseries

    Rechunks the data to avoid excessive Dask chunks
    """
    def group_helper(x):
        # Xarray tests the return shape of the function by calling it with a
        # size 0 array, we don't change the shape
        if x.size == 0:
            return x

        monthday = x.time.dt.month * 100 + x.time.dt.day
        x.coords['monthday'] = monthday

        group = x.groupby("monthday")
        ranking = group.map(func, shortcut=True, **kwargs)

        return ranking

    time_chunked = da.chunk({"time": None})
    ranking = time_chunked.map_blocks(group_helper)

    return ranking
