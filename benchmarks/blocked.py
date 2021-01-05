import pandas
import dask
import xarray
import climtas


def sample_data(years, freq):
    x = range(100)
    y = range(100)
    t = pandas.date_range("2001", str(2001 + years), freq=freq, closed="left")

    t_chunks = pandas.Series(0, index=t).resample("M").count().values

    data = dask.array.concatenate(
        [dask.array.zeros((c, len(y), len(x)), chunks=(-1, 50, 50)) for c in t_chunks]
    )

    da = xarray.DataArray(data, coords=[("time", t), ("y", y), ("x", x)])

    return da


class GroupbySuite:
    def setup(self):
        self.data = sample_data(years=5, freq="D")

    def time_xarray_dayofyear(self):
        self.data.groupby("time.dayofyear").mean().load()

    def time_blocked_dayofyear(self):
        climtas.blocked.blocked_groupby(self.data, time="dayofyear").mean().load()

    def time_blocked_monthday(self):
        climtas.blocked.blocked_groupby(self.data, time="monthday").mean().load()


class ResampleSuite:
    def setup(self):
        self.data = sample_data(years=2, freq="6H")

    def time_xarray(self):
        self.data.resample(time="D").mean().load()

    def time_blocked(self):
        climtas.blocked.blocked_resample(self.data, time=4).mean().load()
