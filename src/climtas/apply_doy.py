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
from . import helpers

"""Functions for analysis on each day of the year

"""


def apply_doy(func, da, *, dim="time", grouping="dayofyear"):
    """Apply a function to a dataset grouped by day of year

    Two grouping methods are available, their behaviour differs in leap years.
    * grouping='dayofyear' will group Feb 29 in a leap year with Mar 1 in a
      non-leap year, leap years will have an additional day 366 consisting of
      Dec 31 values
    * grouping='monthday' will group Feb 29 in a leap year into its own group

    Args:
        func ((:class:`xarray.DataArray`,
            axis=:class:`int`)->:class:`numpy.Array`) Function to apply to the
            grouping. Recieves an axis argument that matches the time dimension
            given by dim. 
        da (:class:`xarray.DataArray`): Data to analyse
        dim (:class:`str`): Time dimension name
        grouping ('dayofyear' or 'monthday'): Grouping method to use (affects
            leap year behaviour) See above

    Returns:
        :class:`xarray.DataArray`, time axis reduced according to func
    """
    applyer = {
        "dayofyear": helpers.apply_by_dayofyear,
        "monthday": helpers.apply_by_monthday,
    }
    return applyer[grouping](da, func)


def rank_doy(da, *, dim="time", grouping="dayofyear"):
    """Calculate the ranking of each cell at the equivalent day of year

    Args:
        da (:class:`xarray.DataArray`): Data to analyse
        grouping ('dayofyear' or 'monthday'): Grouping method to use (affects
            leap year behaviour) See :func:`apply_doy`.

    Returns:
        :class:`xarray.DataArray` of equivalent shape to da
    """

    def func(x, axis):
        return numpy.apply_along_axis(scipy.stats.rankdata, axis, x)

    return apply_doy(func, da, dim=dim, grouping=grouping)


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
        return numpy.nanpercentile(x, p, axis=axis)

    return apply_doy(func, da, dim=dim, grouping=grouping)
