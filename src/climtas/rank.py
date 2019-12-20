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

import numpy
import xarray
import scipy.stats

def rank_along_axis(da, axis):
    return numpy.apply_along_axis(scipy.stats.rankdata, axis, da)

def rank_by_dayofyear(da):
    def group_helper(x):
        # Xarray tests the return shape of the function by calling it with a
        # size 0 array, we don't change the shape
        if x.size == 0:
            return x

        axis = da.get_axis_num('time')
        group = x.groupby("time.dayofyear")
        ranking = group.map(rank_along_axis, axis=axis, shortcut=True)

        return ranking

    time_chunked = da.chunk({"time": None})
    ranking = time_chunked.map_blocks(group_helper)

    return ranking
