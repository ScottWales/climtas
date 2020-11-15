import xarray
import climtas
import climtas.nci
import tempfile
import time
import typing as T
import csv


class Timer:
    def __init__(self):
        self.starts = {}
        self.stops = {}
        self.chunks = {}
        self.client = None

    def mark(self, name: str) -> None:
        if name not in self.starts:
            self.starts[name] = time.perf_counter()
        else:
            self.ends[name] = time.perf_counter()

    def times(self) -> T.Dict[str, float]:
        return {k: self.ends[k] - v for k, v in self.starts.items()}

    def record(self, file) -> None:
        result = {
            "xarray_version": xarray.__version__,
            "climtas_version": climtas.__version__,
            "client_workers": len(self.client.cluster.workers),
            "worker_threads": self.client.cluster.workers[0].nthreads,
        }

        result.update({"chunk_" + k: v for k, v in chunks})

        result.update(self.times())

        exists = os.path.exists(file)

        with open(file, "a") as f:
            writer = csv.DictWriter(f, result.keys())

            if not exists:
                writer.writeheader()

            writer.writerow(result)


def main():
    t = Timer()
    t.mark("total")
    client = climtas.nci.GadiClient()
    t.client = client

    chunks = {"time": 93, "latitude": 91, "longitude": 180}
    t.chunks.update(chunks)

    t.mark("open_mfdataset")
    t2m = xarray.open_mfdataset(
        "/g/data/ub4/era5/netcdf/surface/t2m/*/t2m_era5_global_*.nc",
        chunks=chunks,
        parallel=True,
        combine="override",
        concat_dims="minimal",
    )["t2m"]
    t.mark("open_mfdataset")

    t2m_period = t2m.sel(time=slice("1980", "2019"))

    t.mark("resample")
    t2m_daily = t2m_period.resample(time="D")
    t.mark("resample")

    t.mark("rolling")
    t2m_smooth = t2m.rolling(time=5).mean()
    t.mark("rolling")

    t.mark("smooth")
    t2m_clim = t2m_smooth.groupby("time.dayofyear").quantile(0.9)
    t.mark("smooth")

    t2m_clim.name = "t2m_clim"

    with tempfile.NamedTemporaryFile("wb") as f:
        t.mark("to_netcdf")
        t2m_clim.to_netcdf(f)
        t.mark("to_netcdf")

    t.mark("total")

    t.record("climatology_xarray.csv")
