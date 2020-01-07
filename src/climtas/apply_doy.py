#!/usr/bin/env python
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

import numpy
import dask
import scipy.stats
import xarray
from . import helpers

"""Functions for analysis on each day of the year

"""

def map_doy(func, da, *, dim="time", grouping="dayofyear"):
    """Map a function to a dataset grouped by day of year

    To avoid an explosion of chunk count the time axis is converted to a
    contiguous dimension

    Two grouping methods are available, their behaviour differs in leap years.
    * grouping='dayofyear' will group Feb 29 in a leap year with Mar 1 in a
      non-leap year, leap years will have an additional day 366 consisting of
      Dec 31 values
    * grouping='monthday' will group Feb 29 in a leap year into its own group

    Args:
        func ((:class:`xarray.DataArray`)->:class:`numpy.Array`) Function to
            apply to the grouping, the returned array should be the same shape
            as the input.
        da (:class:`xarray.DataArray`): Data to analyse
        dim (:class:`str`): Time dimension name
        grouping ('dayofyear' or 'monthday'): Grouping method to use (affects
            leap year behaviour) See above

    Returns:
        :class:`xarray.DataArray` with the same shape as da
    """

    # Make chunks continuous on the time axis
    time_chunked = da.chunk({dim: None})

    # Set up grouping
    if grouping == 'dayofyear':
        group_coord = da[dim].dt.dayofyear
    if grouping == 'monthday':
        group_coord = da[dim].dt.month * 100 + da[dim].dt.day

    group_coord.name = grouping

    def chunk_apply_doy(x):
        # Xarray tests the return shape of the function by calling it with a
        # size 0 array, we don't change the shape
        if x.size == 0:
            return x

        x.coords[grouping] = group_coord

        group = x.groupby(grouping)
        return group.map(func)

    result = time_chunked.map_blocks(chunk_apply_doy)
    result.coords[grouping] = group_coord

    return result


def reduce_doy(func, da, *, dim="time", grouping="dayofyear"):
    """Map a function to a dataset grouped by day of year

    To avoid an explosion of chunk count the time axis is converted to a
    contiguous dimension

    Two grouping methods are available, their behaviour differs in leap years.
    * grouping='dayofyear' will group Feb 29 in a leap year with Mar 1 in a
      non-leap year, leap years will have an additional day 366 consisting of
      Dec 31 values
    * grouping='monthday' will group Feb 29 in a leap year into its own group

    Args:
        func ((:class:`numpy.Array`,
            axis=:class:`int`)->:class:`numpy.Array`) Function to apply to the
            grouping, should reduce the time axis. The axis argument is the
            time dimension given by dim
        da (:class:`xarray.DataArray`): Data to analyse
        dim (:class:`str`): Time dimension name
        grouping ('dayofyear' or 'monthday'): Grouping method to use (affects
            leap year behaviour) See above

    Returns:
        :class:`xarray.DataArray` with reduced time axis
    """

    # Make chunks continuous on the time axis
    time_chunked = da.chunk({dim: None})

    # Set up grouping
    if grouping == 'dayofyear':
        group_coord = da[dim].dt.dayofyear
    if grouping == 'monthday':
        group_coord = da[dim].dt.month * 100 + da[dim].dt.day

    group_coord.name = grouping
    groups = da[dim].groupby(group_coord).groups

    # We implement our own version of groupby.reduce as we'll be changing the
    # chunk sizes
    def group_func(x):
        outputs = []
        for k, v in groups.items():
            outputs.append(func(x[...,v], axis=-1))

        return numpy.stack(outputs, axis=-1)

    output_dims = [[grouping]]
    output_sizes = {grouping: len(groups)}

    result = xarray.apply_ufunc(group_func, time_chunked, input_core_dims=[[dim]], output_core_dims=output_dims, dask='parallelized', output_dtypes=[da.dtype], output_sizes=output_sizes)

    result.coords[grouping] = (grouping, [k for k in groups.keys()])

    return result


def rank_doy(da, *, dim="time", grouping="dayofyear"):
    """Calculate the ranking of each cell at the equivalent day of year

    Args:
        da (:class:`xarray.DataArray`): Data to analyse
        grouping ('dayofyear' or 'monthday'): Grouping method to use (affects
            leap year behaviour) See :func:`apply_doy`.

    Returns:
        :class:`xarray.DataArray` of equivalent shape to da
    """

    def func(x):
        axis = x.get_axis_num(dim)
        return numpy.apply_along_axis(scipy.stats.rankdata, axis, x)

    return map_doy(func, da, dim=dim, grouping=grouping)


def percentile_doy(da, p, *, dim="time", grouping="dayofyear"):
    """Calculate the pth percentile of each cell at the equivalent day of year

    Args:
        da (:class:`xarray.DataArray`): Data to analyse, must include a 'time'
            dimension
        p (:class:`int` between 0 and 100): Percentile to calculate
        grouping ('dayofyear' or 'monthday'): Grouping method to use (affects
            leap year behaviour) See :func:`apply_doy`.

    Returns:
        :class:`xarray.DataArray` with time axis reduced to dayofyear / monthday
    """

    def func(x, axis):
        r = numpy.nanpercentile(x, p, axis=axis)
        return r

    return reduce_doy(func, da, dim=dim, grouping=grouping)
