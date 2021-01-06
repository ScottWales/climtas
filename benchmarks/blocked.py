import climtas
import dask
import tempfile
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


class GroupbyDistributedSuite(GroupbySuite):
    def setup(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.client = dask.distributed.Client(local_directory=self.tmpdir.name)
        super().setup()

    def teardown(self):
        self.client.close()


class ResampleDistributedSuite(ResampleSuite):
    def setup(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.client = dask.distributed.Client(local_directory=self.tmpdir.name)
        super().setup()

    def teardown(self):
        self.client.close()
