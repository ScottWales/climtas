import xarray
import climtas
import tempfile
import time
import typing as T
import csv
import os.path
import datetime
import climtas.nci


def get_threshold(t, da, time_range):

    t.mark("rolling")
    da_smooth = da.rolling(time=5).mean()
    t.mark("rolling")

    da_smooth = da_smooth.sel(time=time_range)

    t.mark("groupby")
    da_clim = climtas.blocked_groupby(da_smooth, time="dayofyear").percentile(90)
    t.mark("groupby")

    da_clim.name = "climatology"

    with tempfile.NamedTemporaryFile("wb") as f:
        t.mark("to_netcdf")
        climtas.io.to_netcdf_throttled(da_clim, f.name)
        threshold = xarray.open_dataarray(f.name).load()
        t.mark("to_netcdf")

    return threshold


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

    t.mark("resample")
    t2m_daily = climtas.blocked_resample(t2m, time=24).mean()
    t.mark("resample")

    time_range = slice("2015", "2019")

    threshold = get_threshold(t, t2m_daily, time_range)

    sample = t2m_daily.sel(time=slice("2015", "2015"))

    events = climtas.event.find_events(
        sample.groupby("time.dayofyear") > threshold, min_duration=5
    )

    t.mark("event_values")
    values = climtas.event.event_values(sample, events)
    t.mark("event_values")

    t.mark("event_stats")
    stats = values.groupby("event_id").mean()
    t.mark("event_stats")

    t.mark("total")

    t.record("events_climtas.csv")


if __name__ == "__main__":
    main()
