import climtas
import dask
import tempfile
from .sample import sample_data


class EventSuite:
    def setup(self):
        self.data = sample_data(years=1, freq="D")

    def time_find_event(self):
        events = climtas.event.find_events(self.data > 0.9, min_duration=4)
        dask.compute(events)


class EventDistributedSuite(EventSuite):
    def setup(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.client = dask.distributed.Client(local_directory=self.tmpdir.name)
        super().setup()

    def teardown(self):
        self.client.close()
