import xarray
import climtas
import os.path
import climtas.nci
import dask


def get_threshold(t, da, time_range):
    thresh_path = f"/scratch/w35/saw562/tmp/climtas_benchmark_threshold.nc"
    if os.path.exists(thresh_path):
        threshold = xarray.open_dataarray(thresh_path).load()
        return threshold

    da_smooth = da.rolling(time=5).mean()
    da_smooth = da_smooth.sel(time=time_range)

    da_clim = climtas.blocked_groupby(da_smooth, time="dayofyear").percentile(90)

    da_clim.name = "climatology"

    climtas.io.to_netcdf_throttled(da_clim, thresh_path)
    threshold = xarray.open_dataarray(thresh_path).load()

    return threshold


def main():
    n = 400

    t = climtas.profile.Timer(f"1 year {n}x{n} horiz")
    client = climtas.nci.GadiClient()
    t.client = client

    try:

        chunks = {"time": -1, "latitude": 91, "longitude": 180}
        t.chunks.update(chunks)

        t.mark("open_mfdataset")
        t2m = xarray.open_mfdataset(
            # "/g/data/ub4/era5/netcdf/surface/t2m/*/t2m_era5_global_*.nc",
            "/g/data/rt52/era5/single-levels/reanalysis/2t/*/2t_era5_oper_sfc_*.nc",
            chunks=chunks,
            parallel=True,
            combine="nested",
            concat_dim="time",
            compat="override",
            coords="minimal",
        )["t2m"]
        t.mark("open_mfdataset")

        t2m = t2m.sel(time=slice("2014", "2020"))

        t.mark("resample")
        t2m_daily = climtas.blocked_resample(t2m, time=24).mean()
        t.mark("resample")

        time_range = slice("2015", "2019")

        t.exclude("threshold")
        threshold = get_threshold(t, t2m_daily, time_range)
        t.exclude("threshold")

        sample = t2m_daily.sel(time=slice("2015", "2015"))[:, :n, :n]
        threshold = threshold[:, :n, :n]

        t.mark("find_events")
        events = climtas.event.find_events(
            (sample.groupby("time.dayofyear") > threshold).chunk({"time": 30}),
            min_duration=5,
            use_dask=True,
        )
        t.mark("find_events")

        print(events.shape)
        print(sample)

        t.mark("event_values")
        # values = climtas.event.event_values(sample, events, use_dask=True)
        # (values,) = dask.compute(values)
        # values.to_csv("values.csv")
        t.mark("event_values")

        t.mark("event_stats")
        # stats = values.groupby("event_id")["value"].mean()
        # (stats,) = dask.compute(stats)
        t.mark("event_stats")

        t.record("results.csv")

    finally:
        client.profile(filename="dask-profile.html")

    client.close()


if __name__ == "__main__":
    main()
