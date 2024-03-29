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

"""Functions for reading and saving data

These functions try to use sensible chunking both for dask objects read and
netcdf files written
"""

import xarray
import dask
import pandas
import typing as T
import pathlib
import logging

from .helpers import optimized_dask_get, throttle_futures


def _ds_encoding(ds, complevel):
    # Setup compression and chunking
    encoding = {}
    logging.basicConfig(level=logging.DEBUG)
    for k, v in ds.data_vars.items():

        # Get original encoding
        encoding[k] = v.encoding

        # Update encoding to enable compression
        encoding[k].update(
            {
                "zlib": True,
                "shuffle": True,
                "complevel": complevel,
                "chunksizes": getattr(v.data, "chunksize", None),
            }
        )

        # Clean up encoding
        encoding[k] = {
            kk: vv
            for kk, vv in encoding[k].items()
            if kk
            in [
                "fletcher32",
                "chunksizes",
                "complevel",
                "least_significant_digit",
                "shuffle",
                "contiguous",
                "zlib",
                "_FillValue",
                "dtype",
            ]
        }

        # Log removed keys
        removed_keys = [kk for kk in v.encoding.keys() if not kk in encoding[k].keys()]
        if len(removed_keys) > 0:
            logging.debug(f"removed encoding keys for {k}: {removed_keys}")
    return encoding


def to_netcdf_throttled(
    ds: T.Union[xarray.DataArray, xarray.Dataset],
    path: T.Union[str, pathlib.Path],
    complevel: int = 4,
    max_tasks: int = None,
    show_progress: bool = True,
):
    """
    Save a DataArray to file by calculating each chunk separately (rather than
    submitting the whole Dask graph at once). This may be helpful when chunks
    are large, e.g. doing an operation on dayofyear grouping for a long timeseries.

    Chunks are calculated with at most 'max_tasks' chunks running in parallel -
    this defaults to the number of workers in your dask.distributed.Client, or
    is 1 if distributed is not being used.

    This is a very basic way to handle backpressure, where data is coming in
    faster than it can be processed and so fills up memory. Ideally this will
    be fixed in Dask itself, see e.g.
    https://github.com/dask/distributed/issues/2602

    In particular, it will only work well if the chunks in the dataset are
    independent (e.g. if doing operations over a timeseries for a single
    horizontal chunk so the horizontal chunks are isolated).

    Args:
        da (:class:`xarray.Dataset` or :class:`xarray.DataArray`): Data to save
        path (:class:`str` or :class:`pathlib.Path`): Path to save to
        complevel (:class:`int`): NetCDF compression level
        max_tasks (:class:`int`): Maximum tasks to run at once (default number of distributed
            workers)
        show_progress (:class:`bool`): Show a progress bar with estimated completion time
    """

    if isinstance(ds, xarray.DataArray):
        ds = ds.to_dataset()

    # Setup compression and chunking
    encoding = _ds_encoding(ds, complevel)

    # Prepare storing the data to netcdf, but don't evaluate
    f = ds.to_netcdf(str(path), encoding=encoding, compute=False)

    # This is some very low-level dask operations. behind the scenes dask
    # stores its objects as a graph of operations and their dependencies.
    # We're going to grab a specific operation, 'dask.array.core.store_chunk',
    # and run each instance of that operation in a throttled manner, so they
    # don't all just get submitted at once and overwhelm memory, at the expense
    # of having to do stuff like reading input multiple times rather than just
    # once.

    # We also need to make a new graph, where the tasks that have 'store_chunk'
    # as a dependency know that their pre-requisite has been completed. To do
    # this we just need to fix up the 'store_chunk' tasks, other tasks that
    # 'store_chunk' depends on will be automatically cleaned up when dask
    # optimises the graph

    old_graph = f.__dask_graph__()  # type: ignore
    new_graph = {}  # type: ignore
    store_keys = []

    # Pull out the 'store_chunk' operations from the graph and put them in a
    # list
    for k, v in old_graph.items():
        try:
            if v[0] == dask.array.core.store_chunk:
                store_keys.append(k)
                new_graph[k] = None  # Mark the task done in new_graph
                continue
        except ValueError:
            # Found a numpy array or similar, so comparison fails
            pass
        except IndexError:
            pass
        new_graph[k] = v

    if show_progress:
        from tqdm.auto import tqdm

        store_keys = tqdm(store_keys)

    # Run the 'store_chunk' tasks with 'old_graph'
    throttle_futures(old_graph, store_keys, max_tasks=max_tasks)

    # Finalise any remaining operations with 'new_graph'
    optimized_dask_get(new_graph, list(f.__dask_layers__()))  # type: ignore


def to_netcdf_series(
    ds: T.Union[xarray.DataArray, xarray.Dataset],
    path: T.Union[str, pathlib.Path],
    groupby: str,
    complevel: int = 4,
):
    """
    Split a dataset into multiple parts, and save each part into its own file

    path should be a :meth:`str.format()`-compatible string. It is formatted
    with three arguments: `start` and `end`, which are
    :obj:`pandas.Timestamp`, and `group` which is the name of the current
    group being output (e.g. the year when using `groupby='time.year'`). These
    can be used to name the file, e.g.::

        path_a = 'data_{group}.nc'
        path_b = 'data_{start.month}_{end.month}.nc'
        path_c = 'data_{start.year:04d}{start.month:02d}{start.day:02d}.nc'

    Note that `start` and `end` are the first and last timestamps of the
    group's data, which may not match the boundary start and end dates

    Args:
        da (:class:`xarray.Dataset` or :class:`xarray.DataArray`): Data to save
        path (:class:`str` or :class:`pathlib.Path`): Path template to save to
        groupby (:class:`str`): Grouping, as used by :meth:`xarray.DataArray.groupby`
        complevel (:class:`int`): NetCDF compression level
    """

    if isinstance(ds, xarray.DataArray):
        ds = ds.to_dataset()

    dim = groupby.split(".")[0]

    encoding = _ds_encoding(ds, complevel)

    for key, part in ds.groupby(groupby):
        start = pandas.Timestamp(part[dim].values[0])
        end = pandas.Timestamp(part[dim].values[-1])

        fpath = str(path).format(start=start, end=end, group=key)
        part.to_netcdf(fpath, encoding=encoding)
