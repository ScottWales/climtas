import dask
import pandas
import numpy
from climtas.helpers import *


def test_blockwise():
    da = dask.array.zeros((10, 10), chunks=(5, 5))

    def func(da):
        return pandas.DataFrame({"mean": da.mean()}, index=[1])

    meta = pandas.DataFrame({"mean": pandas.Series([], dtype=da.dtype)})

    df = map_blocks_array_to_dataframe(func, da, meta)
    df = df.compute()

    numpy.testing.assert_array_equal(df.to_numpy(), [[0], [0], [0], [0]])

    def func(da, block_info=None):
        return pandas.DataFrame.from_records([block_info[0]], index=[1])

    df = map_blocks_array_to_dataframe(func, da, meta)
    df = df.compute()

    numpy.testing.assert_array_equal(
        df["chunk-location"].sort_values().apply(lambda x: x[0]),
        numpy.array(
            [
                0,
                0,
                1,
                1,
            ]
        ),
    )

    numpy.testing.assert_array_equal(
        df["chunk-location"].sort_values().apply(lambda x: x[1]),
        numpy.array(
            [
                0,
                1,
                0,
                1,
            ]
        ),
    )
