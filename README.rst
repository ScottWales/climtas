climtas
==========

.. image:: https://img.shields.io/circleci/build/github/ScottWales/climtas/master
   :target: https://circleci.com/gh/ScottWales/climtas
   :alt: CircleCI

.. image:: https://img.shields.io/codecov/c/github/ScottWales/climtas/master
   :target: https://codecov.io/gh/ScottWales/climtas
   :alt: Codecov

.. image:: https://img.shields.io/readthedocs/climtas/latest
   :target: https://climtas.readthedocs.io/en/latest/
   :alt: Read the Docs (latest)

.. image:: https://img.shields.io/conda/v/ScottWales/climtas
   :target: https://anaconda.org/ScottWales/climtas
   :alt: Conda

Functions for working with large (> 10 GB) datasets using Xarray and Dask,
especially for working in the time domain

Topics
------

* `Apply a function grouping by day of year, without massive numbers of dask chunks <https://climtas.readthedocs.io/en/latest/api.html#module-climtas.blocked>`_:

Example notebook: `ERA-5 90th percentile climatology <https://nbviewer.jupyter.org/github/ScottWales/climtas/blob/master/notebooks/era5-heatwave.ipynb>`_

.. code-block:: python

    >>> import numpy; import pandas; import xarray
    >>> time = pandas.date_range("20010101", "20030101")
    >>> data = numpy.random.rand(len(time))
    >>> da = xarray.DataArray(data, coords=[("time", time)])
    >>> da = da.chunk({"time": 365})

    >>> from climtas import blocked_groupby
    >>> blocked_groupby(da, time='dayofyear').mean()
    <xarray.DataArray 'stack-...' (dayofyear: 366)>
    dask.array<mean_agg-aggregate, shape=(366,), dtype=float64, chunksize=(365,), chunktype=numpy.ndarray>
    Coordinates:
      * dayofyear  (dayofyear) int64 1 2 3 4 5 6 7 8 ... 360 361 362 363 364 365 366



* `Find and apply a function to events <https://climtas.readthedocs.io/en/latest/api.html#module-climtas.event>`_:

.. code-block:: python

    >>> from climtas.event import find_events, map_events
    >>> temp = xarray.DataArray([28,31,34,32,30,35,39], dims=['time'])
    >>> events = find_events(temp > 30)
    >>> sums = map_events(temp, events, lambda x: {'sum': x.sum().item()})
    >>> events.join(sums)
       time  event_duration  sum
    0     1               3   97
    1     5               2   74

* `Memory-saving write to NetCDF <https://climtas.readthedocs.io/en/latest/api.html#module-climtas.io>`_

Examples
--------

See the examples in the `notebooks <notebooks>`_ directory for mores ideas on how to
use these functions to analyse large datasets
