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
import dask.delayed
import xarray
import itertools
import typing as T

"""Helper functions

These functions are low-level, and mainly intended for internal use
"""


def map_blocks_to_delayed(
    da: xarray.DataArray, func,
    *args, **kwargs,
) -> T.List[T.Tuple[T.List[int], T.Any]]:
    """
    Run some function 'func' on each dask chunk of 'da'

    The function is called like `func(da_block, offset=offset)` - `da_block` the chunk
    to work on, with coordinates from `da`; and `offset`, the location of
    `block` within `da`.

    The function is wrapped as a :obj:`dask.delayed`, :func:`chunk_map` returns
    a list of (offset, delayed result) for each chunk of `da`.

    If you're wanting to convert the results back into an array, see
    :func:`xarray.map_blocks` or :func:`dask.array.map_blocks`

    >>> def func(da_chunk, offset):
    ...    return da_chunk.mean().values

    >>> da = xarray.DataArray(numpy.eye(10), dims=['x','y'])
    >>> da = da.chunk({'x': 5})

    >>> results = map_blocks_to_delayed(da, func)
    >>> results #doctest: +ELLIPSIS
    [([0, 0], Delayed(...)), ([5, 0], Delayed(...))]

    >>> dask.compute(results)
    ([([0, 0], array(0.1)), ([5, 0], array(0.1))],)

    Args:
        da: Input DataArray
        func: Function to run
        *args, **kwargs: Passed to func

    Returns:
        List of tuples with the chunk offset in `da` and a delayed result of
        running `func` on that chunk
    """
    data = da.data

    offsets = []
    block_id = []
    for i in range(da.ndim):
        chunks = data.chunks[i]
        block_id.append(range(len(chunks)))
        offsets.append(numpy.cumsum([0, *chunks[:-1]]))

    results = []
    for chunk in itertools.product(*block_id):
        offset = [offsets[d][chunk[d]] for d in range(da.ndim)]
        block = data.blocks[chunk]

        coords = {
            da.dims[d]: da.coords[da.dims[d]][offset[d] : offset[d] + block.shape[d]]
            for d in range(da.ndim)
        }

        da_block = xarray.DataArray(block, dims=da.dims, coords=coords, name=da.name, attrs=da.attrs)

        result = dask.delayed(func)(da_block, *args, offset=offset, **kwargs)

        results.append((offset, result))

    return results


def chunk_count(da: xarray.DataArray) -> float:
    """
    Returns the number of chunks in the dataset
    """
    if da.chunks is None:
        raise Exception
    return numpy.prod([len(c) for c in da.chunks])


def chunk_size(da: xarray.DataArray) -> float:
    """
    Returns the size of the first dask chunk in the dataset
    """
    return numpy.prod(da.data.chunksize) * da.data.itemsize


def graph_size(da: xarray.DataArray) -> int:
    """
    Returns number of nodes in the dask graph
    """
    return len(da.__dask_graph__())


def dask_report(da: xarray.DataArray) -> None:
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
        max_tasks = len(client.cluster.workers)

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
