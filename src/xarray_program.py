# ~/~ begin <<docs/xarray.md#src/xarray_program.py>>[init]
import dask.config
import xarray as xr
import zarr

def main() -> None:
    ds = xr.open_dataset(
        "tutorial.zarr",
        chunks="auto",
    )
    climatology = ds.groupby("time.month").mean("time")
    with dask.config.set(
            {
                "scheduler": "threads",
                "num_workers": 4,
            },
        ), zarr.config.set(
            {
                # This does seem to impact performance and keeps the profile more readable.
                "threading.max_workers": 2,
            },
        ):
        climatology.air.compute();

if __name__ == "__main__":
    main()
# ~/~ end
