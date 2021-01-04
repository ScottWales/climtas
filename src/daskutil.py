"""
Utilities for working with Dask
"""


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

    if not hasattr(obj, "dask"):
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
    layer = dask.highlevelgraph.BasicLayer(result)
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
