import climtas
from .sample import sample_data


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
