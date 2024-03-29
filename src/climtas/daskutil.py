"""
Utilities for working with Dask
"""

import xarray
import dask
import numpy
from itertools import zip_longest
import graphviz
import typing as T


# An array-like value for typing
ArrayVar = T.TypeVar("ArrayVar", xarray.DataArray, dask.array.Array, numpy.ndarray)


def _grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


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

    if not hasattr(obj, "dask") or isinstance(obj, numpy.ndarray):
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
    for x in _grouper(top_layer, n):
        x = [xx for xx in x if xx is not None]

        graph = obj.dask.cull(set(x))
        values = schedule(graph, list(x))
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


def visualize_block(arr: dask.array.Array, sizes=True) -> graphviz.Digraph:
    """
    Visualise the graph of a single chunk from 'arr'

    In a Jupyter notebook the graph will automatically display, otherwise use
    :meth:`graphviz.Digraph.render` to create an image.

    Args:
        arr: Array to visualise
        sizes: Calculate the sizes of each node and display as the node label
               if True
    """
    import dask.dot

    name = arr.name
    graph = arr.dask
    layer = graph.layers[name]
    block = next(iter(layer.keys()))
    culled = graph.cull(set([block]))

    attrs = {}
    if sizes:
        attrs = graph_sizes(arr)

    graph = dask.dot.to_graphviz(culled, data_attributes=attrs)

    return graph


def graph_sizes(arr: dask.array.Array) -> T.Dict[T.Hashable, T.Dict]:
    """
    Get the node sizes for each node in arr's Dask graph, to be used in
    visualisation functions

    Sizes are returned using the 'label' graphviz attribute

    >>> import dask.dot
    >>> a = dask.array.zeros((10,10), chunks=(5,5))
    >>> sizes = graph_sizes(a)
    >>> dask.dot.to_graphviz(a.dask, data_attributes=sizes) # doctest: +ELLIPSIS
    <graphviz.graphs.Digraph object ...>

    Note: All nodes will be computed to calculate the size
    """

    keys = list(arr.dask.keys())
    sizes = dict(
        zip(
            keys,
            [
                {"label": dask.utils.format_bytes(x.nbytes)}
                if isinstance(x, numpy.ndarray)
                else {}
                for x in dask.get(arr.dask, keys)
            ],
        )
    )

    return sizes
