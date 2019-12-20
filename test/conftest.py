import pytest
import dask


@pytest.fixture
def distributed_client(tmpdir):
    c = dask.distributed.Client(
        n_workers=1, threads_per_worker=1, local_directory=tmpdir / "dask-worker-space"
    )
    yield c
    c.close()
