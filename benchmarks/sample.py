import pandas
import dask
import xarray


def sample_data(years, freq):
    x = range(100)
    y = range(100)
    t = pandas.date_range("2001", str(2001 + years), freq=freq, closed="left")

    t_chunks = pandas.Series(0, index=t).resample("M").count().values

    data = dask.array.concatenate(
        [
            dask.array.random.random((c, len(y), len(x)), chunks=(-1, 50, 50))
            for c in t_chunks
        ]
    )

    da = xarray.DataArray(data, coords=[("time", t), ("y", y), ("x", x)])

    return da
