#!/usr/bin/env python
# Copyright 2018 ARC Centre of Excellence for Climate Extremes
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
from __future__ import print_function
from abc import ABCMeta, abstractmethod

import six
import xarray
import numpy
import iris
import mule
import os


"""
Different grid types
"""


def identify_grid(dataset):
    """
    Identify the grid used by a Dataset. Returns the appropriate :class:`Grid`
    object

    Args:
        dataset (xarray.DataArray): Input dataset

    Returns:
        Grid: Grid for that dataset
    """

    if isinstance(dataset, Grid):
        return dataset

    try:
        if dataset.attrs["conventions"] == "SCRIP":
            return ScripGrid(dataset)
    except KeyError:
        pass
    except AttributeError:
        pass

    try:
        if dataset.lon.ndim == 1 and dataset.lat.ndim == 1:
            return LonLatGrid(lons=dataset.lon, lats=dataset.lat)
    except AttributeError:
        pass

    raise NotImplementedError


@six.add_metaclass(ABCMeta)
class Grid(object):
    """Abstract base class for grids"""

    @abstractmethod
    def to_cdo_grid(self, outfile):
        """
        Write the grid to a format readable by CDO's regridder (either text or
        SCRIP format)

        Args:
            outfile: File-like object to write to
        """

    @abstractmethod
    def to_netcdf(self, outfile):
        """
        Create a netCDF file using the grid

        Args:
            outfile: Path or File-like object to write to

        Note that if `outfile` is a file object it will be closed
        automatically.
        """

    def to_scrip(self):
        """
        Create a SCRIP file using the grid

        Returns:
            xarray.Dataset containing SCRIP grid description
        """
        raise NotImplementedError


class LonLatGrid(Grid):
    """
    A cartesian grid, with lats and lons one dimensional arrays
    """

    def __init__(self, lats, lons, mask=None):
        """
        Args:
            lats (numpy.array): Grid latitudes
            lons (numpy.array): Grid longitude
        """

        self.lats = lats
        self.lons = lons
        self.mask = mask

        if self.mask is None:
            self.mask = numpy.ones((self.lons.size, self.lats.size))

        if self.lats.ndim != 1 or self.lons.ndim != 1:
            raise Exception("Lons and Lats must be 1D")

    def to_cdo_grid(self, outfile):
        outfile.write("gridtype = lonlat\n".encode())

        outfile.write(("xsize = %d\n" % len(self.lons)).encode())
        outfile.write(
            ("xvals = %s\n" % (",".join(["%f" % x for x in self.lons]))).encode()
        )

        outfile.write(("ysize = %d\n" % len(self.lats)).encode())
        outfile.write(
            ("yvals = %s\n" % (",".join(["%f" % x for x in self.lats]))).encode()
        )

        outfile.flush()

    def to_netcdf(self, outfile):
        ds = xarray.DataArray(
            data=numpy.zeros((len(self.lats), len(self.lons))),
            coords=[("lat", self.lats), ("lon", self.lons)],
        )
        ds.lat.attrs["units"] = "degrees_north"
        ds.lon.attrs["units"] = "degrees_east"
        ds.to_netcdf(outfile)

    def to_scrip(self):
        lat = self.lats
        lon = self.lons % 360

        top = (lat.shift(lat=-1) + lat) / 2.0
        top[-1] = 90

        bot = (lat.shift(lat=1) + lat) / 2.0
        bot[0] = -90

        left = (lon - (lon - lon.roll(lon=1).values) % 360) / 2.0
        right = lon + ((lon.roll(lon=-1).values - lon) % 360) / 2.0

        center_lon, center_lat = numpy.meshgrid(lon, lat)

        corner_lon0, corner_lat0 = numpy.meshgrid(left, bot)
        corner_lon1, corner_lat1 = numpy.meshgrid(right, bot)
        corner_lon2, corner_lat2 = numpy.meshgrid(right, top)
        corner_lon3, corner_lat3 = numpy.meshgrid(left, top)

        corner_lat = numpy.array(
            [
                x.reshape(-1)
                for x in [corner_lat0, corner_lat1, corner_lat2, corner_lat3]
            ]
        )
        corner_lon = numpy.array(
            [
                x.reshape(-1)
                for x in [corner_lon0, corner_lon1, corner_lon2, corner_lon3]
            ]
        )

        scrip = xarray.Dataset(
            coords={
                "grid_dims": (
                    ["grid_rank"],
                    numpy.array([lon.size, lat.size], dtype="i4"),
                ),
                "grid_center_lat": (["grid_size"], center_lat.reshape(-1)),
                "grid_center_lon": (["grid_size"], center_lon.reshape(-1)),
                "grid_imask": (["grid_size"], self.mask.reshape(-1).astype("i4")),
                "grid_corner_lat": (["grid_size", "grid_corners"], corner_lat.T),
                "grid_corner_lon": (["grid_size", "grid_corners"], corner_lon.T),
            }
        )

        scrip.grid_center_lat.attrs["units"] = "degrees"
        scrip.grid_center_lon.attrs["units"] = "degrees"
        scrip.grid_corner_lat.attrs["units"] = "degrees"
        scrip.grid_corner_lon.attrs["units"] = "degrees"

        return scrip


class UMGrid(LonLatGrid):
    @classmethod
    def from_mask(cls, mask_path):
        umfile = mule.load_umfile(mask_path)
        mask_field = None
        for f in umfile.fields:
            if f.lbuser4 == 30:
                mask_field = f
                break

        mask = xarray.DataArray(
            mask_field.get_data(), dims=["lat", "lon"], name=os.path.basename(mask_path)
        )
        mask.coords["lon"] = (
            mask_field.bzx + (1 + numpy.arange(mask.shape[1])) * mask_field.bdx
        )
        mask.coords["lat"] = (
            mask_field.bzy + (1 + numpy.arange(mask.shape[0])) * mask_field.bdy
        )

        mask = mask.where(mask == 0)

        mask.lon.attrs["standard_name"] = "longitude"
        mask.lat.attrs["standard_name"] = "latitude"
        mask.lon.attrs["units"] = "degrees_east"
        mask.lat.attrs["units"] = "degrees_north"

        return mask


class ScripGrid(Grid):
    def __init__(self, grid):
        self._grid = grid

    def to_cdo_grid(self, outfile):
        self._grid.to_netcdf(outfile)

    def to_netcdf(self, outfile):
        self._grid.to_netcdf(outfile)

    def to_scrip(self):
        return self._grid
