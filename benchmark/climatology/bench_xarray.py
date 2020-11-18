import xarray
import climtas
import tempfile
import time
import typing as T
import csv
import os.path
import datetime
import climtas.nci


def main():
    t = climtas.profile.Timer("5 year 100x100 horiz")
    t.mark("total")
    client = climtas.nci.GadiClient()
    t.client = client

    chunks = {"time": 24 * 4, "latitude": 91, "longitude": 180}
    t.chunks.update(chunks)

    t.mark("open_mfdataset")
    t2m = xarray.open_mfdataset(
        "/g/data/ub4/era5/netcdf/surface/t2m/*/t2m_era5_global_*.nc",
        chunks=chunks,
        parallel=True,
        combine="nested",
        concat_dim="time",
        compat="override",
        coords="minimal",
    )["t2m"]
    t.mark("open_mfdataset")

    t2m = t2m.sel(time=slice("2014", "2020"))[:, :100, :100]

    t.mark("resample")
    t2m_daily = t2m.resample(time="D").mean()
    t.mark("resample")

    t2m_daily = t2m_daily.chunk({"time": 10})

    t.mark("rolling")
    t2m_smooth = t2m_daily.rolling(time=5).mean()
    t.mark("rolling")

    t2m_smooth = t2m_smooth.sel(time=slice("1980", "2019"))
    t2m_smooth = t2m_smooth.sel(time=slice("2015", "2019"))

    t.mark("groupby")
    t2m_clim = t2m_smooth.chunk({"time": -1}).groupby("time.dayofyear").quantile(0.9)
    t.mark("groupby")

    t2m_clim.name = "t2m_clim"

    with tempfile.NamedTemporaryFile("wb") as f:
        t.mark("to_netcdf")
        t2m_clim.to_netcdf(f.name)
        t.mark("to_netcdf")

    t.mark("total")

    t.record("climatology_xarray.csv")


if __name__ == "__main__":
    main()
