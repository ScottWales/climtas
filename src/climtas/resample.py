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

def resample_regular(da, *, count):
    """Reduce the time axis by a factor of count

    This can be used to convert e.g. hourly data to daily data. It does not
    change the number of Dask chunks in the dataset.

    The count argument must evenly divide the size of the first dimension of da

    Args:
        da (:class:`xarray.DataArray`): Input data
        count (:class:`int`): Factor to reduce the axis by

    Returns:
        A :class:`xarray.DataArray` with the first dimension reduced by a
        factor of count and a new 'sample' dimension of size count. The samples
        can be reduced using normal xarray operations, e.g.
        :meth:`xarray.DataArray.mean`.
    """
    shape = list(da.shape)
    shape[0] //= count
    shape.insert(1, count)

    data = da.data.reshape(shape)

    dims = list(da.dims)
    dims.insert(1, 'sample')

    result = xarray.DataArray(data, dims=dims)

    result.coords[dims[0]] = da.coords[dims[0]][::count]
    for d in dims[2:]:
        result.coords[d] = da.coords[d]

    return result
