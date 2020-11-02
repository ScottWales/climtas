#!/g/data/hh5/public/apps/nci_scripts/python-analysis3
# Copyright 2020 Scott Wales
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
"""
Profiling dask data processing

* :func:`benchmark` runs a function with different chunks, returning the time
    taken for each chunk setting

* :func:`profile` runs a function with a single chunk setting, returning the
    time taken in different dask stages and chunk information

Profile results
===============

    time_total
        Total time taken to process the data (seconds)
    time_open
        Time spent opening the dataset (seconds)
    time_function
        Time spent running the function (seconds)
    time_optimize
        Time spent optimizing the Dask graph (seconds)
    time_load
        Time spent computing the data with Dask (seconds)
    chunks
        Chunk shape
    nchunks_in
        Number of chunks in loaded data
    nchunks_out
        Number of chunks in function output
    chunksize_in
        Size of chunks in loaded data
    chunksize_out
        Size of chunks in function output
    tasks_in
        Dask graph size in loaded data
    tasks_out
        Dask graph size in function output
    tasks_optimized
        Dask graph size after optimizing function output
"""

from typing import Dict, Any, List
import xarray
import dask
import time
import pandas
import numpy


def benchmark(
    paths: str,
    variable: str,
    chunks: Dict[str, List[int]],
    function,
    run_count: int = 3,
    mfdataset_args: Dict[str, Any] = {},
):
    """
    Profile a function on different chunks of data

    Opens a dataset with :func:`xarray.open_mfdataset` with one of the chunk
    options, then runs function on variable

    >>> def func(da):
    ...     return t2m.mean()
    >>> climtas.profile.benchmark(
    ...     '/g/data/ub4/era5/netcdf/surface/t2m/2019/t2m_era5_global_20190101_*.nc',
    ...     variable='t2m',
    ...     function=func,
    ...     chunks={'time':[93, 93], 'latitude': [91, 91], 'longitude': [180, 180*2]}) #doctest: +SKIP

    Args:
        paths: Paths to open (as :func:`xarray.open_mfdataset`)
        variable: Variable in the dataset to use
        chunks: Mapping of dimension name to a list of chunk sizes, one entry
            for each run
        function: Function that takes a :obj:`xarray.DataArray` (the variable)
            and returns a :obj:`xarray.DataArray` to test the performance of
        run_count: Number of times to run each profile (the minimum time is returned)
        mfdataset_args: Extra arguments to pass to :func:`xarray.open_mfdataset`

    Returns:
        :obj:`pandas.DataFrame` with information from :func:`profile` for each
        run
    """

    css = []
    results = []
    for values in zip(*chunks.values()):
        cs = dict(zip(chunks.keys(), values))
        results.append(
            profile(paths, variable, cs, function, run_count, mfdataset_args)
        )

    r = pandas.DataFrame(results)

    return r


def profile(
    paths: str,
    variable: str,
    chunks: Dict[str, int],
    function,
    run_count: int = 3,
    mfdataset_args: Dict[str, Any] = {},
):
    """
    Run a function run_count times, returning the minimum time taken

    >>> def func(da):
    ...     return t2m.mean()
    >>> climtas.profile.profile(
    ...     '/g/data/ub4/era5/netcdf/surface/t2m/2019/t2m_era5_global_20190101_*.nc',
    ...     variable='t2m',
    ...     function=func,
    ...     chunks={'time':93, 'latitude': 91, 'longitude': 180}) #doctest: +SKIP
    {'time_total': 9.561158710159361,
     'time_open': 0.014718276914209127,
     'time_function': 0.0033595040440559387,
     'time_optimize': 0.01087462529540062,
     'time_load': 9.529402975924313,
     'chunks': {'time': 93, 'latitude': 91, 'longitude': 180},
     'nchunks_in': 512,
     'nchunks_out': 1,
     'chunksize_in': '6.09 MB',
     'chunksize_out': '4 B',
     'tasks_in': 513,
     'tasks_out': 1098,
     'tasks_optimized': 1098}

    Args:
        paths: Paths to open (as :func:`xarray.open_mfdataset`)
        variable: Variable in the dataset to use
        chunks: Mapping of dimension name to chunk sizes
        function: Function that takes a :obj:`xarray.DataArray` (the variable)
            and returns a :obj:`dask.array.Array` to test the performance of
        run_count: Number of times to run each profile (the minimum time is returned)
        mfdataset_args: Extra arguments to pass to :func:`xarray.open_mfdataset`

    Returns:
        Dict[str, int] :ref:`profiling information<Profile results>`
    """

    result = profile_once(paths, variable, chunks, function, mfdataset_args)

    for n in range(run_count - 1):
        r = profile_once(paths, variable, chunks, function, mfdataset_args)

        for k, v in r.items():
            if k.startswith("time_") and v < result[k]:
                result[k] = v

    return result


def profile_once(
    paths: str,
    variable: str,
    chunks: Dict[str, int],
    function,
    mfdataset_args: Dict[str, Any] = {},
):
    """
    Run a single profile instance

    >>> def func(da):
    ...     return t2m.mean()
    >>> climtas.profile.profile_once(
    ...     '/g/data/ub4/era5/netcdf/surface/t2m/2019/t2m_era5_global_20190101_*.nc',
    ...     variable='t2m',
    ...     function=func,
    ...     chunks={'time':93, 'latitude': 91, 'longitude': 180}) #doctest: +SKIP
    {'time_total': 9.561158710159361,
     'time_open': 0.014718276914209127,
     'time_function': 0.0033595040440559387,
     'time_optimize': 0.01087462529540062,
     'time_load': 9.529402975924313,
     'chunks': {'time': 93, 'latitude': 91, 'longitude': 180},
     'nchunks_in': 512,
     'nchunks_out': 1,
     'chunksize_in': '6.09 MB',
     'chunksize_out': '4 B',
     'tasks_in': 513,
     'tasks_out': 1098,
     'tasks_optimized': 1098}

    Args:
        paths: Paths to open (as :func:`xarray.open_mfdataset`)
        variable: Variable in the dataset to use
        chunks: Mapping of dimension name to chunk sizes
        function: Function that takes a :obj:`xarray.DataArray` (the variable)
            and returns a :obj:`dask.array.Array` to test the performance of
        run_count: Number of times to run each profile (the minimum time is returned)
        mfdataset_args: Extra arguments to pass to :func:`xarray.open_mfdataset`

    Returns:
        Dict[str, int] :ref:`profiling information<Profile results>`
    """

    results = {}

    total_start = time.perf_counter()

    open_start = time.perf_counter()
    with xarray.open_mfdataset(paths, chunks=chunks, **mfdataset_args) as data:
        open_end = time.perf_counter()

        var = data[variable]
        tasks_in = len(var.data.__dask_graph__())
        chunks_in = var.data.npartitions
        chunksize_in = dask.utils.format_bytes(
            numpy.prod(var.data.chunksize) * var.dtype.itemsize
        )

        func_start = time.perf_counter()
        r = function(var).data
        func_end = time.perf_counter()

        tasks = len(r.__dask_graph__())
        chunksize = dask.utils.format_bytes(numpy.prod(r.chunksize) * r.dtype.itemsize)
        chunks_out = r.npartitions

        opt_start = time.perf_counter()
        opt = dask.optimize(r)
        opt_end = time.perf_counter()

        tasks_opt = len(r.__dask_graph__())

        load_start = time.perf_counter()
        dask.compute(opt)
        load_end = time.perf_counter()

    total_end = time.perf_counter()

    results["time_total"] = total_end - total_start
    results["time_open"] = open_end - open_start
    results["time_function"] = func_end - func_start
    results["time_optimize"] = opt_end - opt_start
    results["time_load"] = load_end - load_start
    results["chunks"] = chunks
    results["nchunks_in"] = chunks_in
    results["nchunks_out"] = chunks_out
    results["chunksize_in"] = chunksize_in
    results["chunksize_out"] = chunksize
    results["tasks_in"] = tasks_in
    results["tasks_out"] = tasks
    results["tasks_optimized"] = tasks_opt

    return results
