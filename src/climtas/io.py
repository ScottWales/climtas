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

from .helpers import optimized_dask_get


def to_netcdf_chunkwise(da, path, complevel=4):
    """
    Save a DataArray to file by calculating each chunk separately (rather than
    submitting the whole Dask graph at once). This may be helpful when chunks
    are large, e.g. doing an operation on dayofyear grouping for a long timeseries.

    Chunks are not calculated in parallel, so it may be inefficient with many
    processors
    """
    ds = xarray.Dataset({da.name: da})

    encoding = {
        da.name: {
            "zlib": True,
            "shuffle": True,
            "complevel": complevel,
            "chunksizes": da.data.chunksize,
        }
    }

    f = ds.to_netcdf(path, encoding=encoding, compute=False)

    # Run each of the save operations one at a time, then finalize
    old_graph = f.__dask_graph__()
    new_graph = {}
    for k, v in old_graph.items():
        if v[0] == dask.array.core.store_chunk:
            new_graph[k] = optimized_dask_get(old_graph, k)
            continue

        new_graph[k] = v

    # Finalise
    ff = optimized_dask_get(new_graph, list(f.__dask_layers__()))
