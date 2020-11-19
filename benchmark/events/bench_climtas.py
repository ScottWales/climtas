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

    chunks = {"time": -1, "latitude": 91, "longitude": 180}
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
    time_range = slice("2015", "2019")

    t.mark("resample")
    t2m_daily = climtas.blocked_resample(t2m, time=24).mean()
    t.mark("resample")

    t.mark("rolling")
    t2m_smooth = t2m_daily.rolling(time=5).mean()
    t.mark("rolling")

    t2m_smooth = t2m_smooth.sel(time=time_range)

    t.mark("groupby")
    t2m_clim = climtas.blocked_groupby(t2m_smooth, time="dayofyear").percentile(90)
    t.mark("groupby")

    t2m_clim.name = "t2m_clim"

    with tempfile.NamedTemporaryFile("wb") as f:
        t.mark("to_netcdf")
        climtas.io.to_netcdf_throttled(t2m_clim, f.name)
        threshold = xarray.open_dataarray(f.name).load()
        t.mark("to_netcdf")

    t.mark("find_events")
    events = climtas.event.find_events(
        t2m.sel(time=time_range) > threshold, min_duration=5
    )
    t.mark("find_events")

    t.mark("event_map")

    def stats_func(da):
        return {
            "mean": da.mean().values,
            "max": da.max().values,
        }

    stats = climtas.event.map_events(t2m.sel(time=time_range), events, stats_func)
    t.mark("event_map")

    t.mark("total")

    t.record("climatology_climtas.csv")


if __name__ == "__main__":
    main()
