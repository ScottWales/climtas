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

from climtas import blocked_groupby

import xarray
import pandas
import numpy

import pytest


@pytest.mark.xfail
def test_rank_by_dayofyear():
    data = [3, 1, 2]
    dates = ["19900101", "19910101", "19920101"]

    da = xarray.DataArray(data, coords=[("time", pandas.to_datetime(dates))])

    ranked = apply_doy.rank_doy(da)
    numpy.testing.assert_array_equal(ranked.data, [3, 1, 2])


@pytest.mark.xfail
def test_leap_year():
    data = [3, 1, 2, 5, 4]
    dates = ["19920229", "19930301", "19940301", "19950301", "19960229"]

    da = xarray.DataArray(data, coords=[("time", pandas.to_datetime(dates))])

    ranked = apply_doy.rank_doy(da)
    numpy.testing.assert_array_equal(ranked.data, [3, 1, 2, 5, 4])

    ranked = apply_doy.rank_doy(da, grouping="monthday")
    numpy.testing.assert_array_equal(ranked.data, [1, 1, 2, 3, 2])
