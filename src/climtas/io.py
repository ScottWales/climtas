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

import xarray
import dask

from .helpers import optimized_dask_get, throttle_futures


def to_netcdf_throttled(ds, path, complevel=4, max_tasks=None, show_progress=True):
    """
    Save a DataArray to file by calculating each chunk separately (rather than
    submitting the whole Dask graph at once). This may be helpful when chunks
    are large, e.g. doing an operation on dayofyear grouping for a long timeseries.

    Chunks are calculated with at most 'max_tasks' chunks running in parallel -
    this defaults to the number of workers in your dask.distributed.Client, or
    is 1 if distributed is not being used.

    Args:
        da: xarray.Dataset to save
        path: Path to save to
        complevel: NetCDF compression level
        max_tasks: Maximum tasks to run at once (default number of distributed
            workers)
        show_progress: Show a progress bar with estimated completion time
    """

    encoding = {}

    if isinstance(ds, xarray.DataArray):
        ds = ds.to_dataset()

    for k, v in ds.data_vars.items():
        encoding[k] = {
            "zlib": True,
            "shuffle": True,
            "complevel": complevel,
            "chunksizes": getattr(v.data, "chunksize", None),
        }

    f = ds.to_netcdf(str(path), encoding=encoding, compute=False)

    old_graph = f.__dask_graph__()
    new_graph = {}
    store_keys = []

    # Pull out the 'store_chunk' operations from the graph and run them
    # throttled
    for k, v in old_graph.items():
        if v[0] == dask.array.core.store_chunk:
            new_graph[k] = None
            store_keys.append(k)
            continue
        new_graph[k] = v

    if show_progress:
        from tqdm.auto import tqdm

        store_keys = tqdm(store_keys)

    throttle_futures(old_graph, store_keys, max_tasks=None)

    # Finalise
    ff = optimized_dask_get(new_graph, list(f.__dask_layers__()))
