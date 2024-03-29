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
import scipy.stats
import typing as T
from typing_extensions import Protocol
from dask.array.percentile import merge_percentiles
import numbers


class DataArrayFunction(Protocol):
    def __call__(self, da: xarray.DataArray, **kwargs) -> xarray.DataArray:
        ...


class NumpyFunction(Protocol):
    def __call__(
        self, __d: numpy.typing.ArrayLike, axis: int, **kwargs
    ) -> numpy.ndarray:
        ...


class BlockedResampler:
    """A blocked resampling operation, created by :func:`blocked_resample`

    Works like :class:`xarray.core.resample.DataarrayResample`, with the
    constraint that the resampling is a regular interval, and that the
    resampling interval evenly divides the length along *dim* of every Dask
    chunk in *da*.

    The benefit of this restriction is that no extra Dask chunks are
    created by the resampling, which is important for large datasets.
    """

    def __init__(self, da: xarray.DataArray, dim: str, count: int):
        """
        Args:
            da (:class:`xarray.DataArray`): Input DataArray
            dim (:class:`str`): Dimension to group over
            count (:class:`int`): Grouping length
        """
        self.da = da
        self.dim = dim
        self.axis = T.cast(int, self.da.get_axis_num(dim))
        self.count = count

        # Safety checks
        if dim not in da.coords:
            raise Exception(f"{dim} is not a coordinate")
        if da.sizes[dim] % count != 0:
            raise Exception(
                f"Count {count} does not evenly divide the size of {dim} ({da.sizes[dim]})"
            )
        expected_coord = pandas.date_range(
            da.coords[dim].data[0],
            freq=(
                pandas.to_datetime(da.coords[dim].data[1])
                - pandas.to_datetime(da.coords[dim].data[0])
            ),
            periods=da.sizes[dim],
        )
        if not numpy.array_equal(da.coords[dim], expected_coord):
            raise Exception(f"{dim} has an irregular period")

    def reduce(self, op: T.Callable, **kwargs) -> xarray.DataArray:
        r"""Apply an arbitrary operation to each resampled group

        The function *op* is applied to each group. The grouping axis is given
        by *axis*, this axis should be reduced out by *op* (e.g. like
        :func:`numpy.mean` does)

        Args:
            op ((:class:`numpy.array`, axis, \*\*kwargs) -> :class:`numpy.array`):
                Function to reduce out the resampled dimension
            **kwargs: Passed to *op*

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
            return op(reshaped, axis=(axis + 1), **kwargs)

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

        # Set after we create 'result' - if the original name is None it will
        # be replaced by the dask name, so results won't be identical to xarray
        result.name = self.da.name

        return result

    def mean(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.mean`"""
        return self.reduce(numpy.mean)

    def min(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.min`"""
        return self.reduce(numpy.min)

    def max(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.max`"""
        return self.reduce(numpy.max)

    def nanmin(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.nanmin`"""
        return self.reduce(numpy.nanmin)

    def nanmax(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.nanmax`"""
        return self.reduce(numpy.nanmax)

    def sum(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.sum`"""
        return self.reduce(numpy.sum)


def blocked_resample(da: xarray.DataArray, indexer=None, **kwargs) -> BlockedResampler:
    """Create a blocked resampler

    Mostly works like :func:`xarray.resample`, however unlike Xarray's resample
    this will maintain the same number of Dask chunks

    The input data is grouped into blocks of length count along dim for further
    operations (see :class:`BlockedResampler`)

    Count must evenly divide the size of each block along the target axis

    >>> time = pandas.date_range('20010101','20010110', freq='H', closed='left')
    >>> hourly = xarray.DataArray(numpy.random.random(time.size), coords=[('time', time)])

    >>> blocked_daily_max = blocked_resample(hourly, time='1D').max()
    >>> xarray_daily_max = hourly.resample(time='1D').max()
    >>> xarray.testing.assert_identical(blocked_daily_max, xarray_daily_max)

    >>> blocked_daily_max = blocked_resample(hourly, time=24).max()
    >>> xarray_daily_max = hourly.resample(time='1D').max()
    >>> xarray.testing.assert_identical(blocked_daily_max, xarray_daily_max)

    Args:
        da (:class:`xarray.DataArray`): Resample target
        indexer/kwargs (Dict[dim, count]): Mapping of dimension name to count
            along that axis. May be an integer or a time interval understood by
            pandas (that interval must evenly divide the dataset).

    Returns:
        :class:`BlockedResampler`
    """
    if indexer is None:
        indexer = kwargs
    else:
        indexer = {**indexer, **kwargs}

    if len(indexer) != 1:
        raise Exception(
            f"Only one dimension can be resampled at a time, received {indexer}"
        )

    dim, count = list(indexer.items())[0]

    if not isinstance(count, int):
        # Something like a pandas period, resample the time axis to get the count
        counts = da[dim].resample({dim: count}).count()
        if counts.min() != counts.max():
            raise Exception(
                f"Period '{count}' does not evenly divide dimension '{dim}'"
            )
        count = counts.values[0]

    if da.sizes[dim] % count != 0:
        raise Exception(f"Period '{count}' does not evenly divide dimension '{dim}'")

    return BlockedResampler(da, dim=dim, count=count)


class BlockedGroupby:
    """A blocked groupby operation, created by :func:`blocked_groupby`

    Works like :class:`xarray.core.groupby.DataArrayGroupBy`, with the
    constraint that the data contains no partial years

    The benefit of this restriction is that no extra Dask chunks are
    created by the grouping, which is important for large datasets.
    """

    def __init__(self, da: xarray.DataArray, grouping: str, dim: str = "time"):
        """
        Args:
            da (:class:`xarray.DataArray`): Input DataArray
            dim (:class:`str`): Dimension to group over
            grouping ('dayofyear' or 'monthday'): Grouping type
        """
        self.da = da
        self.grouping = grouping
        self.dim = dim

        # Check the time axis
        expected_time = pandas.date_range(
            da[dim].data[0], periods=da.sizes[dim], freq="D"
        )

        if not numpy.array_equal(da[dim], expected_time):
            raise Exception(f"Expected {dim} to be regularly spaced daily data")

        begin = pandas.to_datetime(
            da.coords[dim].data[0]
        ) - pandas.tseries.offsets.YearBegin(n=0)
        if begin != da.coords[dim][0]:
            raise Exception(f"{dim} does not start on Jan 1")

        end = pandas.to_datetime(
            da.coords[dim].data[-1]
        ) + pandas.tseries.offsets.YearEnd(n=0)
        if end != da.coords[dim][-1]:
            raise Exception(f"{dim} does not end on Dec 31")

    def _group_year(self, d, axis, empty):
        if d.shape[axis] == 365:
            if self.grouping == "dayofyear":
                return dask.array.concatenate([d.data, empty], axis=axis)
            elif self.grouping == "monthday":
                return dask.array.concatenate(
                    [
                        d.isel({self.dim: slice(None, 31 + 28)}).data,
                        empty,
                        d.isel({self.dim: slice(31 + 28, None)}).data,
                    ],
                    axis=axis,
                )
            else:
                raise Exception()
        elif d.shape[axis] == 366:
            return d.data

    def _ungroup_year(self, d, axis, source):
        if d.shape[axis] == 365:
            if self.grouping == "dayofyear":
                return dask.array.take(source, slice(0, 365), axis=axis)
            elif self.grouping == "monthday":
                return dask.array.concatenate(
                    [
                        dask.array.take(source, slice(0, 31 + 28), axis=axis),
                        dask.array.take(source, slice(31 + 28 + 1, None), axis=axis),
                    ],
                    axis=axis,
                )
            else:
                raise Exception()
        elif d.shape[axis] == 366:
            return source

    def _block_data(self, da):
        years = da.groupby(f"{self.dim}.year")
        axis = da.get_axis_num(self.dim)
        blocks = []

        expand = dask.array.full_like(da.isel({self.dim: slice(0, 1)}).data, numpy.nan)

        for y, d in years:
            blocks.append(self._group_year(d, axis, expand))

        data = dask.array.stack(blocks, axis=0)

        return data, axis + 1

    def block_dataarray(self) -> xarray.DataArray:
        """Reshape *self.da* to have a 'year' and a *self.grouping* axis

        The *self.dim* axis is grouped up into individual years, then for each
        year that group's *self.dim* is converted into *self.grouping*, so that
        leap years and non-leap years have the same length. The groups are then
        stacked together to create a new DataArray with 'year' as the first
        dimension and *self.grouping* replacing *self.dim*.

        Data for a leap year *self.grouping* in a non-leap year is NAN

        Returns:
            The reshaped :obj:`xarray.DataArray`

        See:

        * :meth:`apply` will block the data, apply a function and then
          unblock the data
        * :meth:`unblock_dataarray` will convert a DataArray shaped like this
          method's output back into a DataArray shaped like *self.da*
        """
        data, block_axis = self._block_data(self.da)

        dims = list(self.da.dims)
        dims.insert(0, "year")
        dims[block_axis] = self.grouping

        da = xarray.DataArray(data, dims=dims)

        for d in dims:
            if d in self.da.coords:
                try:
                    da.coords[d] = self.da.coords[d]
                except ValueError:
                    # Repeat dimension name or something
                    pass

        if self.grouping == "dayofyear":
            da.coords["dayofyear"] = ("dayofyear", range(1, 367))

        if self.grouping == "monthday":
            # Grab dates from a sample leap year
            basis = pandas.date_range("20040101", "20050101", freq="D", closed="left")
            da.coords["month"] = ("monthday", basis.month)
            da.coords["day"] = ("monthday", basis.day)

        return da

    def unblock_dataarray(self, da: xarray.DataArray) -> xarray.DataArray:
        """Inverse of :meth:`block_dataarray`

        Given a DataArray constructed by :meth:`block_dataarray`, returns an
        ungrouped DataArray with the original *self.dim* axis from *self.da*.

        Data for a leap year *self.grouping* in a non-leap year is dropped
        """
        axis = T.cast(int, da.get_axis_num(self.grouping)) - 1

        blocks = []

        source_groups = self.da.groupby(f"{self.dim}.year")
        result_groups = da.groupby("year")

        # Turn back into a time series
        for sg, rg in zip(source_groups, result_groups):
            s = sg[1]
            r = rg[1]

            # Align r to s's time axis
            blocks.append(self._ungroup_year(s, axis, r))

        data = dask.array.concatenate(blocks, axis=axis)

        # Return a dataarray with the original coordinates
        result = xarray.DataArray(data, dims=self.da.dims, coords=self.da.coords)

        return result

    def apply(self, op: DataArrayFunction, **kwargs) -> xarray.DataArray:
        r"""Apply a function to the blocked data

        *self.da* is blocked to replace the *self.dim* dimension with two new
        dimensions, 'year' and *self.grouping*. *op* is then run on the data,
        and the result is converted back to the shape of *self.da*.

        Use this to e.g. group the data by 'dayofyear', then rank each 'dayofyear'
        over the 'year' dimension

        Args:
            op ((:obj:`xarray.DataArray`, \*\*kwargs) -> :obj:`xarray.DataArray`): Function to apply
            **kwargs: Passed to op

        Returns:
            :obj:`xarray.DataArray` shaped like *self.da*
        """
        block_da = self.block_dataarray()
        result = op(block_da, **kwargs)

        return self.unblock_dataarray(result)

    def reduce(self, op: DataArrayFunction, **kwargs) -> xarray.DataArray:
        r"""Reduce the data over 'year' using *op*

        *self.da* is blocked to replace the *self.dim* dimension with two new
        dimensions, 'year' and *self.grouping*. *op* is then run on the data to
        remove the 'year' dimension

        Note there will be NAN values in the data when there isn't a
        *self.grouping* value for that year (e.g. dayofyear = 366 or (month, day)
        = (2, 29) in a non-leap year)

        Use this to e.g. group the data by 'dayofyear', then get the mean values
        at each 'dayofyear'

        Args:
            op ((:obj:`xarray.DataArray`, \*\*kwargs) -> :obj:`xarray.DataArray`): Function to apply
            **kwargs: Passed to op

        Returns:
            :obj:`xarray.DataArray` shaped like *self.da*, but with *self.dim*
            replaced by *self.grouping*
        """
        block_da = self.block_dataarray()

        return block_da.reduce(op, dim="year", **kwargs)

    def mean(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.mean`

        See: :meth:`reduce`
        """
        return self.block_dataarray().mean(dim="year")

    def min(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.min`

        See: :meth:`reduce`
        """
        return self.block_dataarray().min(dim="year")

    def max(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.max`

        See: :meth:`reduce`
        """
        return self.block_dataarray().max(dim="year")

    def sum(self) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.sum`

        See: :meth:`reduce`
        """
        return self.block_dataarray().sum(dim="year")

    def nanpercentile(self, q: float) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.nanpercentile` over the 'year' axis

        Slower than :meth:`percentile`, but will be correct if there's
        missing data (e.g. on leap days)

        Args:
            q (:obj:`float`): Percentile within the interval [0, 100]

        See: :meth:`reduce`, :meth:`percentile`
        """
        block_da = self.block_dataarray()
        block_da = block_da.chunk({"year": None})
        data = block_da.data.map_blocks(
            numpy.nanpercentile, q=q, axis=0, dtype=block_da.dtype, drop_axis=0
        )

        result = xarray.DataArray(data, dims=block_da.dims[1:])
        for d in block_da.coords:
            try:
                result.coords[d] = block_da.coords[d]
            except ValueError:
                pass
        return result

    def percentile(self, q: float) -> xarray.DataArray:
        """Reduce the samples using :func:`numpy.percentile` over the 'year' axis

        Faster than :meth:`nanpercentile`, but may be incorrect if there's
        missing data (e.g. on leap days)

        Args:
            q (:obj:`float`): Percentile within the interval [0, 100]

        See: :meth:`reduce`, :meth:`nanpercentile`
        """
        block_da = self.block_dataarray()
        block_da = block_da.chunk({"year": None})
        data = block_da.data.map_blocks(
            numpy.percentile, q=q, axis=0, dtype=block_da.dtype, drop_axis=0
        )

        result = xarray.DataArray(data, dims=block_da.dims[1:])
        for d in block_da.coords:
            try:
                result.coords[d] = block_da.coords[d]
            except ValueError:
                pass
        return result

    def rank(self, method: str = "average") -> xarray.DataArray:
        """Rank the samples using :func:`scipy.stats.rankdata` over the 'year' axis

        Args:
            method: See :func:`scipy.stats.rankdata`

        See: :meth:`apply`
        """

        def ranker(da: xarray.DataArray, **kwargs) -> xarray.DataArray:
            axis = da.get_axis_num("year")
            da = da.load()

            def rank_along_axis(array):
                return numpy.apply_along_axis(
                    scipy.stats.rankdata, axis, array, method=method
                )

            def blocked_rank(array):
                chunks = list(array.chunks)
                chunks[axis] = -1
                array = array.rechunk(chunks)
                return dask.array.map_blocks(rank_along_axis, array)

            if isinstance(da.data, dask.array.Array):
                aranker = blocked_rank
            else:
                aranker = rank_along_axis

            result = xarray.apply_ufunc(
                aranker, da, dask="parallelized", output_dtypes=[da.dtype]
            )
            return result

        return self.apply(ranker)

    def _binary_op(self, other: xarray.DataArray, op) -> xarray.DataArray:
        """Generic binary operation (add, subtract etc.)"""
        if not isinstance(other, xarray.DataArray):
            raise TypeError(f"Other operand must be a DataArray (got {type(other)})")

        if self.grouping not in other.dims:
            raise KeyError(
                f"Grouping {self.grouping} not present in other DataArray dimensions ({other.dims})"
            )

        axis = self.da.get_axis_num(self.dim)
        expand = dask.array.full_like(
            self.da.isel({self.dim: slice(0, 1)}).data, numpy.nan
        )
        blocks = []

        # Apply the operation to each year
        for y, d in self.da.groupby(f"{self.dim}.year"):
            grouped = self._group_year(d, axis, expand)
            result = getattr(grouped, op)(other.data)
            blocks.append(self._ungroup_year(d, axis, result))

        # Combine back into a timeseries
        data = dask.array.concatenate(blocks, axis=axis)

        # Return a dataarray with the original coordinates
        result = xarray.DataArray(data, dims=self.da.dims, coords=self.da.coords)

        return result

    def __add__(self, other):
        return self._binary_op(other, "__add__")

    def __sub__(self, other):
        return self._binary_op(other, "__sub__")

    def __mul__(self, other):
        return self._binary_op(other, "__mul__")

    def __div__(self, other):
        return self._binary_op(other, "__div__")


def blocked_groupby(da: xarray.DataArray, indexer=None, **kwargs) -> BlockedGroupby:
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
        indexer/kwargs (Dict[dim, grouping]): Mapping of dimension name to grouping type

    Returns:
        :class:`BlockedGroupby`
    """
    if indexer is None:
        indexer = kwargs
    assert len(indexer) == 1
    dim, grouping = list(indexer.items())[0]

    if grouping in ["dayofyear", "monthday"]:
        return BlockedGroupby(da, dim=dim, grouping=grouping)
    else:
        raise NotImplementedError(f"Grouping {grouping} is not implemented")


def _merge_approx_percentile(chunk_pcts, chunk_counts, finalpcts, pcts, axis, method):
    """
    Merge percentile blocks together

    A Nd implementation of dask.array.percentile.merge_percentiles
    """
    # First axis of chunk_pcts is the pct values
    assert chunk_pcts.shape[0] == len(pcts)
    # Remainder are the chunk results, stacked along 'axis'
    assert chunk_pcts.shape[1:] == chunk_counts.shape

    # Do a manual apply along axis, using the size of chunk_counts as it has the original dimensions
    Ni, Nk = chunk_counts.shape[:axis], chunk_counts.shape[axis + 1 :]

    # Output array has the values for each pct
    out = numpy.empty((len(finalpcts), *Ni, *Nk), dtype=chunk_pcts.dtype)

    # We have the same percentiles for each chunk
    pcts = numpy.tile(pcts, (chunk_counts.shape[axis], 1))

    # Loop over the non-axis dimensions of the original array
    for ii in numpy.ndindex(Ni):
        for kk in numpy.ndindex(Nk):
            # Use dask's merge_percentiles
            out[ii + numpy.s_[...,] + kk] = merge_percentiles(
                finalpcts,
                pcts,
                chunk_pcts[
                    numpy.s_[
                        :,
                    ]
                    + ii
                    + numpy.s_[
                        :,
                    ]
                    + kk
                ].T,
                Ns=chunk_counts[
                    ii
                    + numpy.s_[
                        :,
                    ]
                    + kk
                ],
                method=method,
            )

    return out


def dask_approx_percentile(
    array: dask.array.array,
    pcts,
    axis: int,
    interpolation="linear",
    skipna=True,
):
    """
    Get the approximate percentiles of a Dask array along 'axis', using the 'dask'
    method of :func:`dask.array.percentile`.

    Args:
        array: Dask Nd array
        pcts: List of percentiles to calculate, within the interval [0,100]
        axis: Axis to reduce
        skipna: Ignore NaN values (like :func:`numpy.nanpercentile`) if true

    Returns:
        Dask array with first axis the percentiles from 'pcts', remaining axes from
        'array' reduced along 'axis'
    """
    if isinstance(pcts, numbers.Number):
        pcts = [pcts]

    # The chunk sizes with each chunk reduced along 'axis'
    chunks = list(array.chunks)
    chunks[axis] = [1 for c in chunks[axis]]

    # Reproduce behaviour of dask.array.percentile, adding in '0' and '100' percentiles
    finalpcts = pcts.copy()
    pcts = numpy.pad(pcts, 1, mode="constant")
    pcts[-1] = 100

    # Add the percentile size to the start of 'chunks'
    chunks.insert(0, len(pcts))

    if skipna:
        pctile = numpy.nanpercentile
    else:
        pctile = numpy.percentile

    # The percentile of each chunk along 'axis'
    chunk_pcts = dask.array.map_blocks(
        pctile,
        array,
        pcts,
        axis,
        keepdims=True,
        method=interpolation,
        chunks=chunks,
        meta=numpy.array((), dtype=array.dtype),
    )
    # The count of each chunk along 'axis'
    chunk_counts = dask.array.map_blocks(
        numpy.ma.count,
        array,
        axis,
        keepdims=True,
        chunks=chunks[1:],
        meta=numpy.array((), dtype="int64"),
    )

    # Now change the chunk size to the final size
    chunks[0] = len(finalpcts)
    chunks.pop(axis + 1)
    # Merge the chunks together using Dask's merge_percentiles function
    merged_pcts = dask.array.map_blocks(
        _merge_approx_percentile,
        chunk_pcts,
        chunk_counts,
        finalpcts=finalpcts,
        pcts=pcts,
        axis=axis,
        method=interpolation,
        drop_axis=axis + 1,
        chunks=chunks,
        meta=numpy.array((), dtype=array.dtype),
    )

    return merged_pcts


def approx_percentile(
    da: T.Union[xarray.DataArray, dask.array.Array, numpy.ndarray],
    q,  # T.Union[numbers.Real, T.List[numbers.Real]]
    dim: str = None,
    axis: int = None,
    skipna: bool = True,
):
    """
    Return an approximation of the qth percentile along a dimension of da

    For large Dask datasets the approximation will compute much faster than
    :func:`numpy.percentile`

    If da contains Dask data, it will use Dask's approximate percentile
    algorithim extended to multiple dimensions, see :func:`dask.array.percentile`

    If da contains Numpy data it will use :func:`numpy.percentile`

    Args:
        da: Input dataset
        q: Percentile to calculate in the range [0,100]
        dim: Dimension name to reduce (xarray data only)
        axis: Axis number to reduce
        skipna: Ignore NaN values (like :func:`numpy.nanpercentile`)

    Returns:
        Array of the same type as da, otherwise as :func:`numpy.percentile`
    """

    if isinstance(q, numbers.Real):
        qlist = [q]
    else:
        qlist = q

    if skipna:
        pctile = numpy.nanpercentile
    else:
        pctile = numpy.percentile

    if isinstance(da, xarray.DataArray) and isinstance(da.data, dask.array.Array):
        # Xarray+Dask
        if axis is None:
            axis = T.cast(int, da.get_axis_num(dim))
        data = dask_approx_percentile(da.data, pcts=q, axis=axis, skipna=skipna)
        dims = ["percentile", *[d for i, d in enumerate(da.dims) if i != axis]]
        coords = {k: v for k, v in da.coords.items() if k in dims}
        coords["percentile"] = xarray.DataArray(qlist, dims="percentile")
        return xarray.DataArray(
            data,
            name=da.name,
            dims=dims,
            coords=coords,
        )

    if isinstance(da, xarray.DataArray):
        # Xarray+Numpy
        return da.quantile([p / 100 for p in qlist], dim=dim, skipna=skipna)

    assert dim is None
    assert axis is not None

    if isinstance(da, dask.array.Array):
        # Dask
        return dask_approx_percentile(da, pcts=q, axis=axis, skipna=skipna)

    # Other
    return pctile(da, q, axis=axis)
