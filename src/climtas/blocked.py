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

""" Xarray operations that act per block
"""

import xarray
import numpy
import pandas
import dask.array


class BlockedResampler:
    def __init__(self, da, dim, count):
        self.da = da
        self.dim = dim
        self.axis = self.da.get_axis_num(dim)
        self.count = count

    def map(self, op):
        """Apply an arbitrary operation to each resampled group

        Args:
            op ((:class:`numpy.array`, axis) -> (:class:`numpy.array`)):
                Function to reduce out the resampled dimension
        """

        def resample_op(block, op, axis, count):
            # Rehape the block, reducing the shape along 'axis' by one and
            # adding a new axis one after it, then reduce out that new axis
            # using op
            shape = list(block.shape)
            shape[axis] //= count
            shape.insert(axis + 1, count)
            reshaped = block.reshape(shape)
            return op(reshaped, axis=(axis + 1))

        data = self.da.data
        if not isinstance(data, dask.array.Array):
            data = self.da.chunk().data

        # Make a new chunk list, with the size along self.axis reduced
        new_chunks = list(data.chunks)
        new_chunks[self.axis] = tuple(c // self.count for c in data.chunks[self.axis])

        # Check even division
        for i, c in enumerate(data.chunks[self.axis]):
            if c % self.count != 0:
                print(data.chunks[self.axis])
                raise Exception(
                    f"count ({self.count}) must evenly divide chunk.shape[{self.axis}] for all chunks, fails at chunk {i} with size {c}"
                )

        # Map the op onto the blocks
        new_data = data.map_blocks(
            resample_op,
            op=op,
            axis=self.axis,
            count=self.count,
            chunks=new_chunks,
            dtype=data.dtype,
        )

        result = xarray.DataArray(new_data, dims=self.da.dims, attrs=self.da.attrs)
        for k, v in self.da.coords.items():
            if k == self.dim:
                v = v[:: self.count]
            result.coords[k] = v

        return result

    def mean(self):
        return self.map(numpy.mean)

    def min(self):
        return self.map(numpy.min)

    def max(self):
        return self.map(numpy.max)

    def sum(self):
        return self.map(numpy.sum)


def blocked_resample(da, indexer=None, **kwargs):
    """Create a blocked resampler

    The input data is grouped into blocks of length count along dim for further
    operations (see :class:`BlockedResampler`)

    Count must evenly divide the size of each block along the target axis

    Unlike Xarray's resample this will maintain the same number of Dask chunks

    >>> time = pandas.date_range('20010101','20010110', freq='H', closed='left')
    >>> hourly = xarray.DataArray(numpy.random.random(time.size), coords=[('time', time)])

    >>> blocked_daily_max = blocked_resample(hourly, time=24).max()
    >>> xarray_daily_max = hourly.resample(time='1D').max()
    >>> xarray.testing.assert_equal(blocked_daily_max, xarray_daily_max)
    
    Args:
        da (:class:`xarray.DataArray`): Resample target
        **kwargs ({dim:count}): Mapping of dimension name to count along that axis

    Returns:
        :class:`BlockedResampler`
    """
    if indexer is None:
        indexer = kwargs
    assert len(indexer) == 1
    dim, count = list(indexer.items())[0]
    return BlockedResampler(da, dim=dim, count=count)
