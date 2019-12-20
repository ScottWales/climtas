import pytest
import dask

@pytest.fixture
def distributed_client():
    c = dask.distributed.Client(n_workers=1, threads_per_worker=1)
    yield c
    c.close()
