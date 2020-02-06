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

""" Xarray operations that act per Dask block
"""

import xarray
import numpy
import pandas
import dask.array


class BlockedResampler:
    """A blocked resampling operation

    Works like :class:`xarray.core.resample.DataarrayResample`, with the
    constraint that the resampling is a regular interval, and that the
    resampling interval evenly divides the length along *dim* of every Dask
    chunk in *da*.

    The benefit of this restriction is that no extra Dask chunks are
    created by the resampling, which is important for large datasets.
    """

    def __init__(self, da, dim, count):
        """
        Args:
            da (:class:`xarray.DataArray`): Input DataArray
            dim (:class:`str`): Dimension to group over
            count (:class:`int`): Grouping length
        """
        self.da = da
        self.dim = dim
        self.axis = self.da.get_axis_num(dim)
        self.count = count

    def reduce(self, op):
        """Apply an arbitrary operation to each resampled group

        The function *op* is applied to each group. The grouping axis is given
        by *axis*, this axis should be reduced out by *op* (e.g. like
        :func:`numpy.mean` does)

        Args:
            op ((:class:`numpy.array`, axis) -> (:class:`numpy.array`)):
                Function to reduce out the resampled dimension

        Returns:
            A resampled :class:`xarray.DataArray`, where every *self.count*
            values along *self.dim* have been reduced by *op*
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
        """ Reduce the samples using numpy.mean
        """
        return self.reduce(numpy.mean)

    def min(self):
        """ Reduce the samples using numpy.min
        """
        return self.reduce(numpy.min)

    def max(self):
        """ Reduce the samples using numpy.max
        """
        return self.reduce(numpy.max)

    def sum(self):
        """ Reduce the samples using numpy.count
        """
        return self.reduce(numpy.sum)


def blocked_resample(da, indexer=None, **kwargs):
    """Create a blocked resampler

    Mostly works like :func:`xarray.resample`, however unlike Xarray's resample
    this will maintain the same number of Dask chunks

    The input data is grouped into blocks of length count along dim for further
    operations (see :class:`BlockedResampler`)

    Count must evenly divide the size of each block along the target axis

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


class BlockedGroupby():
    def __init__(self, da, grouping, dim='time'):
        self.da = da
        self.grouping = grouping
        self.dim = dim

        # Check the time axis
        expected_time = pandas.date_range(da[dim].data[0], periods=da.sizes[dim], freq='D')

        if not numpy.array_equal(da[dim], expected_time):
            raise Exception("Expected {dim} to be regularly spaced daily data")

    def _group_year(self, d, axis, empty):
        if d.shape[axis] == 365:
            if self.grouping == 'dayofyear':
                return dask.array.concatenate([d.data, empty], axis=axis)
            elif self.grouping == 'monthday':
                return dask.array.concatenate([
                    d.isel({self.dim: slice(None, 31+28)}).data, empty, d.isel({self.dim: slice(31+28, None)}).data], axis=axis)
            else:
                raise Exception()
        elif d.shape[axis] == 366:
            return d.data

    def _ungroup_year(self, d, axis, source):
        if d.shape[axis] == 365:
            if self.grouping == 'dayofyear':
                return dask.array.take(source, slice(0,365), axis=axis)
            elif self.grouping == 'monthday':
                return dask.array.concatenate([
                    source.take(axis, slice(0, 31+28)), source.take(axis, slice(31+28+1, None))],
                    axis=axis)
            else:
                raise Exception()
        elif d.shape[axis] == 366:
            return source

    def _block_data(self, da):
        years = da.groupby(f'{self.dim}.year')
        axis = da.get_axis_num(self.dim)
        blocks = []

        expand = dask.array.full_like(da.isel({self.dim: slice(0,1)}).data, numpy.nan)

        for y, d in years:
            blocks.append(self._group_year(d, axis, expand))

        data = dask.array.stack(blocks, axis=0)

        return data, axis+1

    def block_dataarray(self):
        data, block_axis = self._block_data(self.da)

        dims = list(self.da.dims)
        dims.insert(0, 'year')
        dims[block_axis] = self.grouping

        da = xarray.DataArray(data, dims=dims)

        for d in dims:
            if d in self.da.coords:
                try:
                    da.coords[d] = self.da.coords[d]
                except ValueError:
                    # Repeat dimension name or something
                    pass

        if self.grouping == 'dayofyear':
            da.coords['dayofyear'] = ('dayofyear', range(1,367))

        if self.grouping == 'monthday':
            # Grab dates from a sample leap year
            basis = pandas.date_range('20040101','20050101', freq='D', closed='left')
            da.coords['month'] = ('monthday', basis.month)
            da.coords['day'] = ('monthday', basis.day)

        return da

    def reduce(self, op, **kwargs):
        block_da = self.block_dataarray()

        return block_da.reduce(op, dim='year', **kwargs)

    def mean(self):
        """ Reduce the samples using numpy.mean
        """
        return self.block_dataarray().mean(dim='year')

    def min(self):
        """ Reduce the samples using numpy.min
        """
        return self.block_dataarray().min(dim='year')

    def max(self):
        """ Reduce the samples using numpy.max
        """
        return self.block_dataarray().max(dim='year')

    def sum(self):
        """ Reduce the samples using numpy.sum
        """
        return self.block_dataarray().sum(dim='year')

    def percentile(self, q):
        """ Reduce the samples using numpy.percentile
        """
        return self.block_dataarray().reduce(numpy.nanpercentile, q=q)

    def _binary_op(self, other, op):
        if not isinstance(other, xarray.DataArray):
            raise Exception()

        if self.grouping not in other.dims:
            raise Exception()

        axis = self.da.get_axis_num(self.dim)
        expand = dask.array.full_like(self.da.isel({self.dim: slice(0,1)}).data, numpy.nan)
        blocks = []

        # Apply the operation to each year
        for y, d in self.da.groupby(f'{self.dim}.year'):
            grouped = self._group_year(d, axis, expand)
            result = getattr(grouped, op)(other)
            blocks.append(self._ungroup_year(d, axis, result))

        # Combine back into a timeseries
        data = dask.array.concatenate(blocks, axis=axis)

        # Return a dataarray with the original coordinates
        result = xarray.DataArray(data, dims=self.da.dims, coords=self.da.coords)

        return result

    def __add__(self, other):
        return self._binary_op(other, '__add__')
    def __sub__(self, other):
        return self._binary_op(other, '__sub__')
    def __mul__(self, other):
        return self._binary_op(other, '__mul__')
    def __div__(self, other):
        return self._binary_op(other, '__div__')


def blocked_groupby(da, indexer=None, **kwargs):
    """Create a blocked groupby

    Mostly works like :func:`xarray.groupby`, however this will have better
    chunking behaviour at the expense of only working with data regularly
    spaced in time.

    *grouping* may be one of:

     - 'dayofyear': Group by number of days since the start of the year
     - 'monthday': Group by ('month', 'day')

    >>> time = pandas.date_range('20020101','20050101', freq='D', closed='left')
    >>> hourly = xarray.DataArray(numpy.random.random(time.size), coords=[('time', time)])

    >>> blocked_doy_max = blocked_groupby(hourly, time='dayofyear').max()
    >>> xarray_doy_max = hourly.groupby('time.dayofyear').max()
    >>> xarray.testing.assert_equal(blocked_doy_max, xarray_doy_max)

    Args:
        da (:class:`xarray.DataArray`): Resample target
        **kwargs ({dim:grouping}): Mapping of dimension name to grouping type

    Returns:
        :class:`BlockedResampler`
    """
    if indexer is None:
        indexer = kwargs
    assert len(indexer) == 1
    dim, grouping = list(indexer.items())[0]

    if grouping in ['dayofyear', 'monthday']:
        return BlockedGroupby(da, dim=dim, grouping=grouping)
    else:
        raise NotImplementedError(f'Grouping {grouping} is not implemented')

