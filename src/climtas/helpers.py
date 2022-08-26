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

from dask.base import tokenize
import numpy
import dask
import dask.delayed
import dask.optimization
import xarray
import pandas
import typing as T
import dask.array
import dask.dataframe
import itertools

"""Helper functions

These functions are low-level, and mainly intended for internal use
"""


def map_blocks_to_delayed(
    da: xarray.DataArray,
    func,
    axis=None,
    name="blocks-to-delayed",
    args=[],
    kwargs={},
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
        args, kwargs: Passed to func

    Returns:
        List of tuples with the chunk offset in `da` and a delayed result of
        running `func` on that chunk
    """
    data = da.data
    # data.dask = data.__dask_optimize__(data.__dask_graph__(), data.__dask_keys__())

    offsets = []
    block_id = []
    for i in range(da.ndim):
        chunks = data.chunks[i]
        block_id.append(range(len(chunks)))
        offsets.append(numpy.cumsum([0, *chunks[:-1]]))

    results = []
    for chunk in itertools.product(*block_id):
        size = [data.chunks[d][chunk[d]] for d in range(da.ndim)]
        offset = [offsets[d][chunk[d]] for d in range(da.ndim)]
        block = data.blocks[chunk]

        # block.dask, _ = dask.optimization.cull(block.__dask_graph__, block.__dask_layers__())
        # mark = time.perf_counter()
        # block.dask = block.__dask_optimize__(
        #     block.__dask_graph__(), block.__dask_keys__()
        # )
        # print("opt", time.perf_counter() - mark)

        coords = {
            da.dims[d]: da.coords[da.dims[d]][offset[d] : offset[d] + block.shape[d]]
            for d in range(da.ndim)
        }

        da_block = xarray.DataArray(
            block, dims=da.dims, coords=coords, name=da.name, attrs=da.attrs
        )

        # da_block = da[
        #     tuple(slice(offset[d], offset[d] + size[d]) for d in range(da.ndim))
        # ]

        name = name + "-" + tokenize(block.name)

        result = dask.delayed(func, name=name)(da_block, *args, offset=offset, **kwargs)

        results.append((offset, result))

    return results


def chunk_count(da: xarray.DataArray) -> numpy.number:
    """
    Returns the number of chunks in the dataset
    """
    if da.chunks is None:
        raise Exception
    return numpy.prod([len(c) for c in da.chunks]).astype("i")


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


def locate_block_in_dataarray(
    block: dask.array.Array,
    name: str,
    dims: T.List[str],
    coords: T.Dict[T.Hashable, T.Any],
    block_info: T.Dict[str, T.Any],
):
    """
    Locates the metadata of the current block

    Args:
        da (dask.array.Array): A block of xda
        xda (xarray.DataArray): Whole DataArray being operated on
        block_info: Block metadata

    Returns:
        xarray.DataArray with the block and its metadata
    """
    if block_info is not None:
        subset = {
            d: slice(x0, x1) for d, (x0, x1) in zip(dims, block_info["array-location"])
        }

        out_coords = {}
        for k, v in coords.items():
            out_coords[k] = v.isel(
                {kk: vv for kk, vv in subset.items() if kk in v.dims}
            )

    else:
        out_coords = coords

    return xarray.DataArray(block, name=name, dims=dims, coords=out_coords)


def map_blocks_array_to_dataframe(
    func: T.Callable[..., pandas.DataFrame],
    array: dask.array.Array,
    *args,
    meta: pandas.DataFrame,
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
    >>> map_blocks_array_to_dataframe(func, da, meta=meta) # doctest: +NORMALIZE_WHITESPACE
    Dask DataFrame Structure:
                      mean
    npartitions=4
                   float64
                       ...
                       ...
                       ...
                       ...
    Dask Name: func,...

    The mapping function behaves the same as :func:`dask.array.map_blocks`.
    If it has a keyword argument `block_info`, that argument will be filled
    with information about the block location.

    The block can be located within a :obj:`xarray.DataArray`, adding the
    correct coordinate metadata, with :func:`locate_block_in_dataarray`:

    >>> xda = xarray.DataArray(da, dims=['t','x'])
    >>> def func(da, block_info=None):
    ...    da = locate_block_in_dataarray(da, xda, block_info[0])
    ...    return pandas.DataFrame({"mean": da.mean().values}, index=[1])

    Args:
        func ((:obj:`numpy.array`, **kwargs) -> :obj:`pandas.DataFrame`):
            Function to run on each block
        array (:obj:`dask.array.Array`): Dask array to operate on
        meta (:obj:`pandas.DataFrame`): Sample dataframe with the correct
            output columns
        *args, **kwargs: Passed to 'func'

    Returns:
        :obj:`dask.dataframe.DataFrame`, with each block the result of applying
        'func' to a block of 'array', in an arbitrary order
    """

    if getattr(array, "npartitions", None) is None:
        return func(array, *args, **kwargs)

    # Use the array map blocks, with a dummy meta (as we won't be making an array)
    mapped = dask.array.map_blocks(
        func, array, *args, **kwargs, meta=numpy.array((), dtype="i")
    )

    return array_blocks_to_dataframe(mapped, meta)


def array_blocks_to_dataframe(
    array: dask.array.Array, meta: pandas.DataFrame
) -> dask.dataframe.DataFrame:
    """
    Convert the blocks from a dask array to a dask dataframe
    """

    # Grab the Dask graph from the array map
    graph = array.dask
    name = array.name

    # Flatten the results to 1d
    # Keys in the graph layer are (name, chunk_coord)
    # We'll replace chunk_coord with a scalar value
    layer = {}
    for i, v in enumerate(graph.layers[name].values()):
        layer[(name, i)] = v
    graph.layers[name] = dask.highlevelgraph.MaterializedLayer(layer)

    # Low level dask dataframe constructor
    df = dask.dataframe.core.new_dd_object(
        graph, name, meta, [None] * (array.npartitions + 1)
    )

    return df


from itertools import zip_longest


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args)


# An array-like value for typing
ArrayVar = T.TypeVar("ArrayVar", xarray.DataArray, dask.array.Array, numpy.ndarray)


def throttled_compute(arr: ArrayVar, *, n: int, name: T.Hashable = None) -> ArrayVar:
    """
    Compute a Dask object N chunks at a time

    Args:
        obj: Object to compute
        n: Number of chunks to process at once
        name: Dask layer name to compute (default obj.name)

    Returns:
        'obj', with each chunk computed
    """

    # Copy the input in case it's a xarray object
    obj = arr

    if isinstance(arr, xarray.DataArray):
        # Work on the data
        obj = arr.data

    if not hasattr(obj, "dask") or isinstance(arr, numpy.ndarray):
        # Short-circuit non-dask arrays
        return arr

    # Current dask scheduler
    schedule = dask.base.get_scheduler(collections=[obj])

    # Get the layer to work on
    if name is None:
        name = obj.name
    top_layer = obj.dask.layers[name]

    result = {}

    # Compute chunks N at a time
    for x in grouper(top_layer, n):
        x = [xx for xx in x if xx is not None]
        values = schedule(obj.dask, list(x))
        result.update(dict(zip(x, values)))

    # Build a new dask graph
    layer = dask.highlevelgraph.MaterializedLayer(result)
    graph = dask.highlevelgraph.HighLevelGraph.from_collections(name, layer)

    obj.dask = graph

    if isinstance(arr, xarray.DataArray):
        # Add back metadata
        obj = xarray.DataArray(
            obj, name=arr.name, dims=arr.dims, coords=arr.coords, attrs=arr.attrs
        )

    return obj


def visualize_block(arr: dask.array.Array):
    """
    Visualise the graph of a single chunk from 'arr'
    """

    name = arr.name
    graph = arr.dask
    layer = graph.layers[name]
    block = next(layer.values())
    culled = graph.cull(set(block))

    graph = dask.dot.to_graphviz(culled)

    return graph
