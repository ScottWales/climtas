import climtas
import dask
from .sample import sample_data


class EventSuite:
    def setup(self):
        self.data = sample_data(years=1, freq="D")

    def time_find_event(self):
        events = climtas.event.find_events(self.data > 0.9, min_duration=4)
        dask.compute(events)
