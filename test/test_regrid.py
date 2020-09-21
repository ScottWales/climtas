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

from climtas.regrid import *
import xarray
import dask.array
import numpy
import numpy.testing
import pytest


def test_cdo_generate_weights(tmpdir):
    d = xarray.DataArray(
        data=numpy.ones((2, 4)), coords=[("lat", [-45, 45]), ("lon", [0, 90, 180, 270])]
    )
    d.lat.attrs["units"] = "degrees_north"
    d.lon.attrs["units"] = "degrees_east"

    center_lon, center_lat = numpy.meshgrid(d.lon, d.lat)
    d[:, :] = center_lon

    grid = identify_grid(d)
    weights = cdo_generate_weights(d, grid)

    assert "remap_matrix" in weights


def compare_regrids(tmpdir, source, target):
    """
    Check our weight application matches CDO's
    """
    import subprocess

    grid = identify_grid(target)
    gridfile = tmpdir.join("grid")
    with open(str(gridfile), "wb") as f:
        grid.to_cdo_grid(f)

    sourcefile = tmpdir.join("source.nc")
    source.to_netcdf(str(sourcefile))

    outfile = tmpdir.join("out.nc")
    subprocess.check_call(
        ["cdo", "remapbil,%s" % str(gridfile), str(sourcefile), str(outfile)]
    )

    cdo = xarray.open_dataset(str(outfile))

    cms = regrid(source, target)

    numpy.testing.assert_array_equal(cdo["var"].data[...], cms.values[...])


def test_call_regrid(tmpdir):
    a0 = xarray.DataArray(
        [[0, 1], [2, 3]],
        name="var",
        dims=["lat", "lon"],
        coords={"lat": [-45, 45], "lon": [0, 180]},
    )
    a0.lat.attrs["units"] = "degrees_north"
    a0.lon.attrs["units"] = "degrees_east"
    a0 = a0.astype("f4")

    r = regrid(a0, a0)

    assert r is not None


def test_manual_weights(tmpdir):
    """
    'regrid' will accept a SCRIP format weights file as the target
    """
    a0 = xarray.DataArray(
        [[0, 1], [2, 3]],
        name="var",
        dims=["lat", "lon"],
        coords={"lat": [-45, 45], "lon": [0, 180]},
    )
    a0.lat.attrs["units"] = "degrees_north"
    a0.lon.attrs["units"] = "degrees_east"
    a0 = a0.astype("f4")

    grid = identify_grid(a0)
    weights = cdo_generate_weights(a0, grid)

    r = regrid(a0, weights=weights)

    assert r is not None


def test_compare_regrids(tmpdir):
    a0 = xarray.DataArray(
        [[0, 1], [2, 3]],
        name="var",
        dims=["lat", "lon"],
        coords={"lat": [-45, 45], "lon": [0, 180]},
    )
    a0.lat.attrs["units"] = "degrees_north"
    a0.lon.attrs["units"] = "degrees_east"
    a0 = a0.astype("f4")

    compare_regrids(tmpdir.mkdir("a0a0"), a0, a0)

    a1 = xarray.DataArray(
        numpy.zeros((3, 4)),
        name="var",
        dims=["lat", "lon"],
        coords={"lat": [-90, 0, 90], "lon": [0, 90, 180, 270]},
    )
    a1.lat.attrs["units"] = "degrees_north"
    a1.lon.attrs["units"] = "degrees_east"

    compare_regrids(tmpdir.mkdir("a1a1"), a1, a1)


def test_dask_regrid(tmpdir):
    d = dask.array.zeros((2, 2), chunks=(2, 2))

    a0 = xarray.DataArray(
        d, name="var", dims=["lat", "lon"], coords={"lat": [-45, 45], "lon": [0, 180]}
    )
    a0.lat.attrs["units"] = "degrees_north"
    a0.lon.attrs["units"] = "degrees_east"

    r = regrid(a0, a0)

    assert isinstance(r.data, dask.array.Array)


def test_3d_regrid(tmpdir):
    a0 = xarray.DataArray(
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]],
        name="var",
        dims=["time", "lat", "lon"],
        coords={"lat": [-45, 45], "lon": [0, 180], "time": [0, 1, 2]},
    )
    a0.lat.attrs["units"] = "degrees_north"
    a0.lon.attrs["units"] = "degrees_east"
    a0 = a0.astype("f4")

    r = regrid(a0, a0)

    compare_regrids(tmpdir.mkdir("3d"), a0, a0)


def test_latlon_dims():
    a0 = xarray.DataArray(
        numpy.zeros((2, 4)),
        name="var",
        dims=["lat", "lon"],
        coords={"lat": [-45, 45], "lon": [0, 90, 180, 270]},
    )
    a0.lat.attrs["units"] = "degrees_north"
    a0.lon.attrs["units"] = "degrees_east"

    r = regrid(a0, a0)

    assert r.lat.ndim == 1
    assert r.lon.ndim == 1
    assert r.lat.shape == (2,)
    assert r.lon.shape == (4,)

    a1 = xarray.DataArray(
        numpy.zeros((2, 4)),
        name="var",
        dims=["lat", "lon"],
        coords={"lat": [-45, 45], "lon": [-180, -90, 0, 90]},
    )
    a1.lat.attrs["units"] = "degrees_north"
    a1.lon.attrs["units"] = "degrees_east"

    r = regrid(a0, a1)

    assert r.lat.ndim == 1
    assert r.lon.ndim == 1
    assert r.lat.shape == (2,)
    assert r.lon.shape == (4,)

    a2 = xarray.DataArray(
        numpy.zeros((10, 10)),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, 10),
            "lon": numpy.linspace(-180, 180, 10, endpoint=False),
        },
    )
    a2.lat.attrs["units"] = "degrees_north"
    a2.lon.attrs["units"] = "degrees_east"

    r = regrid(a1, a2)

    assert r.lat.ndim == 1
    assert r.lon.ndim == 1
    assert r.lat.shape == (10,)
    assert r.lon.shape == (10,)


def test_big_array():
    alats = 145
    alons = 192
    ats = 1

    a = xarray.DataArray(
        numpy.zeros((ats, alats, alons)),
        name="var",
        dims=["t", "lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, alats),
            "lon": numpy.linspace(0, 360, alons, endpoint=False),
        },
    )
    a.lat.attrs["units"] = "degrees_north"
    a.lon.attrs["units"] = "degrees_east"

    blats = 10
    blons = 10

    b = xarray.DataArray(
        numpy.zeros((blats, blons)),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, blats),
            "lon": numpy.linspace(-180, 180, blons, endpoint=False),
        },
    )
    b.lat.attrs["units"] = "degrees_north"
    b.lon.attrs["units"] = "degrees_east"

    r = regrid(a, b)


def test_esmf_generate_weights():
    alats = 10
    alons = 11

    a = xarray.DataArray(
        numpy.zeros((alats, alons)),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, alats),
            "lon": numpy.linspace(0, 360, alons, endpoint=False),
        },
    )
    a.lat.attrs["units"] = "degrees_north"
    a.lon.attrs["units"] = "degrees_east"

    a[1, 3] = numpy.nan

    blats = 12
    blons = 13

    b = xarray.DataArray(
        numpy.zeros((blats, blons)),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, blats),
            "lon": numpy.linspace(-180, 180, blons, endpoint=False),
        },
    )
    b.lat.attrs["units"] = "degrees_north"
    b.lon.attrs["units"] = "degrees_east"

    b[6, 4] = numpy.nan

    w = esmf_generate_weights(a, b)

    # Weights should have the correct size
    numpy.testing.assert_array_equal(w.src_grid_dims, [alons, alats])
    numpy.testing.assert_array_equal(w.dst_grid_dims, [blons, blats])

    # Masks should pass through
    numpy.testing.assert_array_equal(w.mask_a, a.notnull().data.ravel())
    numpy.testing.assert_array_equal(w.mask_b, b.notnull().data.ravel())


@pytest.mark.parametrize(
    "weight_gen,weight_args",
    [
        (esmf_generate_weights, {}),
        (cdo_generate_weights, {}),
        (cdo_generate_weights, {"method": "ycon"}),
    ],
)
def test_regrid_weight_gen(weight_gen, weight_args):
    alats = 10
    alons = 11

    a = xarray.DataArray(
        numpy.random.rand(alats, alons),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, alats),
            "lon": numpy.linspace(0, 360, alons, endpoint=False),
        },
    )
    a.lat.attrs["units"] = "degrees_north"
    a.lon.attrs["units"] = "degrees_east"

    a[1, 3] = numpy.nan

    blats = 12
    blons = 13

    b = xarray.DataArray(
        numpy.random.rand(blats, blons),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, blats),
            "lon": numpy.linspace(-180, 180, blons, endpoint=False),
        },
    )
    b.lat.attrs["units"] = "degrees_north"
    b.lon.attrs["units"] = "degrees_east"

    b[6, 4] = numpy.nan

    w = weight_gen(a, b, **weight_args)
    c = regrid(a, weights=w)

    # Dimensions match the target grid
    numpy.testing.assert_array_equal(c.shape, b.shape)
    numpy.testing.assert_array_almost_equal(c.lat, b.lat)
    numpy.testing.assert_array_almost_equal(
        numpy.mod(c.lon, 360), numpy.mod(b.lon, 360)
    )

    # We don't just have a grid of NAN
    assert not numpy.isnan(c).all()

    # Output is masked
    assert numpy.isnan(b[6, 4])


def test_chunked_weights():
    alats = 10
    alons = 11

    a = xarray.DataArray(
        numpy.zeros((alats, alons)),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, alats),
            "lon": numpy.linspace(0, 360, alons, endpoint=False),
        },
    )
    a.lat.attrs["units"] = "degrees_north"
    a.lon.attrs["units"] = "degrees_east"

    a[1, 3] = numpy.nan

    blats = 12
    blons = 13

    b = xarray.DataArray(
        numpy.zeros((blats, blons)),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, blats),
            "lon": numpy.linspace(-180, 180, blons, endpoint=False),
        },
    )
    b.lat.attrs["units"] = "degrees_north"
    b.lon.attrs["units"] = "degrees_east"

    b[6, 4] = numpy.nan

    w = esmf_generate_weights(a, b)
    w = w.chunk({'n_s': 10})

    c = regrid(a, weights=w)


def test_regrid_mask():
    alats = 10
    alons = 11

    a = xarray.DataArray(
        numpy.zeros((alats, alons)),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, alats),
            "lon": numpy.linspace(0, 360, alons, endpoint=False),
        },
    )
    a.lat.attrs["units"] = "degrees_north"
    a.lon.attrs["units"] = "degrees_east"

    a[1, 3] = numpy.nan

    blats = 12
    blons = 13

    b = xarray.DataArray(
        numpy.zeros((blats, blons)),
        name="var",
        dims=["lat", "lon"],
        coords={
            "lat": numpy.linspace(-90, 90, blats),
            "lon": numpy.linspace(-180, 180, blons, endpoint=False),
        },
    )
    b.lat.attrs["units"] = "degrees_north"
    b.lon.attrs["units"] = "degrees_east"

    b[6, 4] = numpy.nan

    w = esmf_generate_weights(a, b)

    a[1,3] = 1e10
    c = regrid(a, weights=w)

    assert c.max().values < 10
    assert numpy.isnan(c.values[6, 4])
