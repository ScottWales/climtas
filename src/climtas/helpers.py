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
import xarray
import pandas
import typing as T
import dask.array
import dask.dataframe

"""Helper functions

These functions are low-level, and mainly intended for internal use
"""


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


def map_blocks_array_to_dataframe(
    func: T.Callable[..., pandas.DataFrame],
    array: dask.array.Array,
    meta: pandas.DataFrame,
    prefix: T.Hashable = None,
    **kwargs
) -> dask.dataframe.DataFrame:
    """
    Apply a function `func` to each dask chunk, returning a dataframe

    'func' will be set up to run on each dask chunk of 'array', returning a
    dataframe. These dataframes are then collated into a single dask dataframe.

    The returned dataframe is in an arbitrary order, it may be sorted with
    :meth:`dask.dataframe.DataFrame.set_index`.

    >>> da = dask.array.zeros((10, 10), chunks=(5, 5))
    >>> def func(da):
    ...    return pandas.DataFrame({"mean": da.mean()}, index=[1])
    >>> meta = pandas.DataFrame({"mean": pandas.Series([], dtype=da.dtype)})
    >>> map_blocks_array_to_dataframe(func, da, meta) # doctest: +NORMALIZE_WHITESPACE
    Dask DataFrame Structure:
                      mean
    npartitions=4
                   float64
                       ...
                       ...
                       ...
                       ...
    Dask Name: map_ar_df_func, 8 tasks

    Args:
        func ((:obj:`numpy.array`, **kwargs) -> :obj:`pandas.DataFrame`):
            Function to run on each block
        array (:obj:`dask.array.Array`): Dask array to operate on
        meta (:obj:`pandas.DataFrame`): Sample dataframe with the correct
            output columns
        **kwargs: Passed to 'func'

    Returns:
        :obj:`dask.dataframe.DataFrame`, with each block the result of applying
        'func' to a block of 'array', in an arbitrary order
    """

    indices = range(array.ndim)

    # A unique name in the task graph
    if prefix is None:
        prefix = "map_ar_df_" + func.__name__
    name = prefix + "-" + dask.base.tokenize(array)

    # Setup the blockwise operation
    # The output blocks are the same as the input blocks so that each block
    # gets computed individually, we'll flatten it later
    graph = dask.blockwise.blockwise(
        func,
        name,
        indices,
        array.name,
        indices,
        numblocks={array.name: array.numblocks},
        concatenate=False,
        **kwargs,
    )

    # Add the array's graph
    graph = dask.highlevelgraph.HighLevelGraph.from_collections(
        name, graph, dependencies=[array]
    )

    # Flatten the results to 1d
    # Keys in the graph layer are (name, chunk_coord)
    # We'll replace chunk_coord with a scalar value
    layer = {}
    for i, v in enumerate(graph.layers[name].values()):
        layer[(name, i)] = v
    graph.layers[name] = layer

    # Low level dask dataframe constructor
    df = dask.dataframe.core.new_dd_object(
        graph, name, meta, [None] * (array.npartitions + 1)
    )

    return df
