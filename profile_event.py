import xarray
import tempfile
import climtas
import climtas.nci
import time


class Timer:
    def __init__(self):
        self.starts = {}
        self.ends = {}

    def mark(self, name):
        if name not in self.starts:
            self.starts[name] = time.perf_counter()
        else:
            self.ends[name] = time.perf_counter()

    def results(self):
        return {k: v - self.starts[k] for k, v in self.ends.items()}


if __name__ == "__main__":
    t = Timer()
    t.mark("full")

    client = climtas.nci.GadiClient()
    workers = len(client.cluster.workers)
    threads = sum([w.nthreads for w in client.cluster.workers.values()])

    t.mark("load")
    oisst = xarray.open_mfdataset(
        "/g/data/ua8/NOAA_OISST/AVHRR/v2-0_modified/oisst_avhrr_v2_*.nc",
        chunks={"time": 1},
    )
    sst = oisst.sst
    t.mark("load")

    clim_file = "/scratch/w35/saw562/tmp/oisst_clim.nc"

    t.mark("clim")
    # climatology = climtas.blocked_groupby(
    #     sst.sel(time=slice("1985", "1987")), time="monthday"
    # ).percentile(90)
    # climatology.name = "sst_thresh"

    # climtas.io.to_netcdf_throttled(climatology, clim_file)

    climatology = xarray.open_dataarray(clim_file, chunks={"monthday": 1})
    t.mark("clim")

    t.mark("find")
    delta = climtas.blocked_groupby(sst.sel(time="1985"), time="monthday") - climatology
    delta = delta.chunk({"time": 30, "lat": 100, "lon": 100})
    print(delta)
    events = climtas.event.find_events_block(
        delta > 0, min_duration=10, offset=(0, 0, 0)
    )
    t.mark("find")

    t.mark("full")

    print("workers ", workers, " threads ", threads)
    print(t.results())
