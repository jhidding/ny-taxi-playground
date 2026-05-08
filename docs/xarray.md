# Running computations in parallel on gridded data with Xarray and Dask

> Converted from Jupyter. This notebook gives an example of the use of XArray on large multi-dimensional data sets. This section stands out from the others, which deal with purely tabulated data. Usually there is a bit of a digotomy in data handling between the use of tables and dense datasets. We need to serve both. This episode can fill the gap of explaining how to deal with large 3 or 4 dimensional datasets, while also teaching the use of profilers.

Xarray makes working with labelled multi-dimensional arrays in Python simple, efficient, and fun!

Use Dask and Xarray to churn through terabytes of multi-dimensional array data in formats like HDF, NetCDF, TIFF, or Zarr.  


```python
import os
from pathlib import Path

from distributed import LocalCluster
import dask
import cartopy.crs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr.codecs
```

Helper function for plotting data on a map


```python
def plot_map(da: xr.DataArray, projection=cartopy.crs.Robinson()) -> None:
    """Create a map plot."""
    plt.figure(figsize=[12,8])
    p = da.plot(
        subplot_kws={
            "projection": projection,
        },
        add_colorbar=False,
        transform=cartopy.crs.PlateCarree(),
    )
    plt.colorbar(p, shrink=0.6)
    p.axes.set_global()
    p.axes.coastlines()
```

## Explore the tutorial data

We will use the xarray tutorial data, a small dataset that xarray can automatically download. When loading large data with xarray, the `chunks` keyword argument should be provided to ensure data is not loaded into memory. Specifying `chunks={}` will load the data with chunks as they are on disk. This is useful for inspection, but should generally not be used for computations as the chunks on disk are typically of the order of a few megabytes, while a good chunks size for computations with Dask is about 100 megabyte. The `chunks="auto"` option will automatically choose a good chunk size for running computations with Dask, while aligning the Dask chunks with the chunks on Disk for efficient reading.

```python
ds = xr.tutorial.open_dataset("air_temperature", chunks="auto")
ds
```

We see that it is a three dimensional dataset of air temperature. The data has been interpreted as a single Dask chunk because it is so small.


```python
ds.air
```

Let's plot the first time point of the data to get an idea of what we are working with:


```python
north_america = cartopy.crs.Orthographic(-90, 35)
plot_map(ds.air.isel(time=0), projection=north_america)
```
    
![png](notebook_files/notebook_9_0.png)
  
So, this datasets is air temperature over North America from 2013 and 2014 at four time points a day.

## Prepare the tutorial data

We will increase the spatial resolution of the tutorial dataset to make the computation that we are going to perform more interesting and then save to to zarr format.

```python
scale_factor = 30
ds = xr.tutorial.open_dataset(
    "air_temperature",
    chunks={"time": 30}  # We choose the input chunks here such the resulting high-resolution dataset has reasonably sized (~100 MB) chunks.
)
ds = ds.interp(
    lat=np.linspace(ds.lat.data[0], ds.lat.data[-1], len(ds.lat) * scale_factor),
    lon=np.linspace(ds.lon.data[0], ds.lon.data[-1], len(ds.lon) * scale_factor),
    method="linear",
)
ds["air"] = ds.air.astype("float32")
ds.air
```

Let's plot the data again to see that it is now much higher resolution:

```python
plot_map(ds.air.isel(time=0), projection=north_america)
```
    
![png](notebook_files/notebook_14_0.png)
    
Let's save the data to disk for usage in the remainder of the tutorial. We use chunks of the order of a megabyte for saving to disk.

```python
# If you do not know how the data is going to be read when saving the file,
# a reasonable choice may be to chunk along all dimensions.
n_splits = 20
file_chunks = {
    d: int(s/n_splits)
    for d, s in ds.sizes.items()
}
file_chunks
```

    {'time': 146, 'lat': 37, 'lon': 79}

```python
# In this tutorial we will apply some temporal operations, so we will
# split up the data along the time axis only.
file_chunks = {"time": 1, "lat": len(ds.lat), "lon": len(ds.lon)}
```

```python
ds.chunk(file_chunks).air
```

The following code will write the data to a zarr directory called "tutorial.zarr". A separate file is created for each chunk inside the directory. This may not be desirable on HPC systems that have filesystems, such as Lustre, that do not perform well with many small files. In this case, you may be able to [use a `ZipStore` instead of a directory](https://zarr.readthedocs.io/en/stable/user-guide/storage/#zip-store).

```python
compressor = zarr.codecs.BloscCodec(
    cname="zstd",
    clevel=5,
    shuffle=zarr.codecs.BloscShuffle.shuffle
)
ds.chunk(file_chunks).to_zarr(
    "tutorial.zarr",
    mode="w",
    encoding={v: {"compressors": compressor} for v in ds.data_vars}
);
```

A more commonly used file format is NetCDF. We also write the data to NetCDF (this may take a while):

```python
xr.open_dataset(
    "tutorial.zarr",
    chunks="auto",
).to_netcdf(
    "tutorial.nc",
    engine="netcdf4",
    mode="w",
    encoding={v: {"zlib": True, "complevel": 5, "shuffle": True, "chunksizes": tuple(file_chunks.values())} for v in ds.data_vars},
)
```

## Running computations in parallel

The example computation we will use in this tutorial is a computation that is commonly used in climate science: we compute the average monthly temperature. This is called a "climatology".


```python
ds = xr.open_dataset(
    "tutorial.zarr",
    chunks="auto",
)
ds.air
```

We can also do the same experiments with a NetCDF dataset, uncomment the code below to use that instead:

```python
# ds = xr.open_dataset(
#     "tutorial.nc",
#     chunks="auto",
# )
# ds.air
```

```python
climatology = ds.groupby("time.month").mean("time")
climatology.air
```

### Compute with the threaded scheduler

Recommended if all these conditions are true:
- the computation has a small task graph
- the computation has a simple task graph
- the computation should be run on a single computer


```python
# Run the computation on a single CPU core.
with dask.config.set({"scheduler": "threads", "num_workers": 1}):
    %time climatology.air.compute();
```

    CPU times: user 39.3 s, sys: 12.8 s, total: 52.1 s
    Wall time: 41.4 s



```python
# Run the computation in parallel on two CPU cores and notice it is faster.
with dask.config.set({"scheduler": "threads", "num_workers": 2}):
    %time climatology.air.compute();
```

    CPU times: user 43.1 s, sys: 15.1 s, total: 58.2 s
    Wall time: 24.4 s



```python
# Run the computation in parallel on four CPU cores. It is slightly faster still than two CPU cores, but not nearly two times as fast.
with dask.config.set({"scheduler": "threads", "num_workers": 4}):
    %time climatology.air.compute();
```

    CPU times: user 55.7 s, sys: 25.8 s, total: 1min 21s
    Wall time: 19.2 s


Parallism on the threaded scheduler may be limited by Python's [Global Interpreter Lock](https://docs.python.org/3/glossary.html#term-global-interpreter-lock), unless a free-threaded Python build is used, and depending on which libraries are used for reading the input data, there may be additional locks that limit parallelism.

### Compute with the distributed scheduler

Recommended if any of these conditions is true:
- the computation has a large task graph
- the computation has a complex task graph
- the data is read using a library that does not support multiple threads (HDF5, NetCDF, ..)
- the computation should be run on multiple computers


```python
cluster = LocalCluster(
    n_workers=3,
    threads_per_worker=2,
    memory_limit="2.5GiB",  # per worker
)
client = cluster.get_client()
client
```

```python
%time climatology.air.compute();
```

    2026-05-04 17:41:42,558 - distributed.worker.memory - WARNING - Unmanaged memory use is high. This may indicate a memory leak or the memory may not be released to the OS; see https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os for more information. -- Unmanaged memory: 1.93 GiB -- Worker memory limit: 2.50 GiB
    2026-05-04 17:41:43,378 - distributed.worker.memory - WARNING - Unmanaged memory use is high. This may indicate a memory leak or the memory may not be released to the OS; see https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os for more information. -- Unmanaged memory: 1.84 GiB -- Worker memory limit: 2.50 GiB
    2026-05-04 17:41:45,766 - distributed.worker.memory - WARNING - Unmanaged memory use is high. This may indicate a memory leak or the memory may not be released to the OS; see https://distributed.dask.org/en/latest/worker-memory.html#memory-not-released-back-to-the-os for more information. -- Unmanaged memory: 1.84 GiB -- Worker memory limit: 2.50 GiB


    CPU times: user 1.29 s, sys: 309 ms, total: 1.6 s
    Wall time: 16.2 s



```python
client.shutdown()
```

### Why is the computation slow

Use a profiler to find out why the computation is slow. When opening the profile on www.speedscope.app, the "Left Heavy" button aggregates all the function calls so it easier to see which in which functions the program spends most time. When you move the mouse over a colored box the file and line number of the code represented by that box is shown. When you click the box additional detail is available and double clicking will zoom in on a box. There is a dropdown menu at the top to select the thread to view the profile of. There are multiple threads:
- there is a main thread running the program
- Zarr use the [`asyncio`](https://docs.python.org/3/library/asyncio.html) module to read and write data in parallel and runs that in one or more threads
- Dask uses multiple threads to do the computation

The most interesting threads are those representing dask workers. Note the calls to the function `getter` from the `dask.array.core` module in the dask worker threads: this function loads data from disk. Clicking it and looking under `Total` reveals the amount time is spent waiting for data to come from disk when reading the data file. Reading the netcdf file seems to take considerably more time, most likely because only one thread per process can use the NetCDF library at a time, though this bottleneck can be alleviated by using the Dask distributed scheduler.

Note that not all threads are using a full CPU core all the time. They may be waiting on input data to be loaded from disk or memory, for their turn to run Python code (global interpreter lock) or use the NetCDF library, or if you are running more threads than there are CPU cores in your machine, they may be waiting for a CPU core to become available to run on.


```python
#| file: src/xarray_program.py
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
```

```bash
py-spy record --idle --nonblocking --format speedscope --output profile.json -- python program.py
```

### Bonus material: other Xarray backends for reading NetCDF data

Xarray supports various readers for NetCDF. We saw that NetCDF appears to be slower than zarr, so if your input data comes as NetCDF or HDF5 files, it may be tempting to try these.

#### Virtual zarr

By storing the indexes of where data is in the NetCDF files, virtual zarr files can be created that can be used to read the file with the zarr library, thus bypassing the single-threaded NetCDF and HDF5 libraries. See the [VirtualiZarr](https://virtualizarr.readthedocs.io) package for more information.


```python
# Copied from https://virtualizarr.readthedocs.io/en/stable/usage.html#__tabbed_1_7
import xarray as xr
from obstore.store import LocalStore
from obspec_utils.registry import ObjectStoreRegistry

from virtualizarr import open_virtual_dataset, open_virtual_mfdataset
from virtualizarr.parsers import HDFParser

from pathlib import Path

store_path = Path.cwd()
file_path = str(store_path / "tutorial.nc")
file_url = f"file://{file_path}"

store = LocalStore(prefix=store_path)
registry = ObjectStoreRegistry({file_url: store})
parser = HDFParser()
with open_virtual_dataset(
      url=file_url,
      parser=parser,
      registry=registry,
    ) as vds:
    vds.vz.to_kerchunk('tutorial.json', format='json')
```

    /home/bandela/src/bouweandela/parallel-python-xarray-dask-example/.pixi/envs/default/lib/python3.14/site-packages/zarr/codecs/numcodecs/_codecs.py:141: ZarrUserWarning: Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations.
      super().__init__(**codec_config)


The dataset can be loaded from the virtual zarr file with


```python
ds = xr.open_dataset(
    'tutorial.json',
    engine="kerchunk",
    chunks="auto",
)
ds.air
```

Running the experiments above with this dataset seems to suggest that virtual zarr parallizes better on the threaded scheduler, but does not work so well with the distributed scheduler.

#### h5py

[h5py](https://github.com/h5netcdf/h5netcdf) is a library for reading and writing NetCDF files without using the NetCDF libraries and supported by Xarray as a backend. h5py also supports the [pyfive](https://github.com/NCAS-CMS/pyfive) library as its own backend, which is a pure Python implementation so should not be limited by the single-threaded implementation of the HDF5 library. The h5py backend can be specified using the environment variable `H5NETCDF_READ_BACKEND`.


```python
os.environ["H5NETCDF_READ_BACKEND"] = "h5py"
ds = xr.open_dataset(
    "tutorial.nc",
    engine="h5netcdf",
    chunks="auto",
)
ds.air
```

Running the experiments above with this dataset seems to suggest that just using h5py instead of netCDF4 does not offer any improvement in parallism. This is expected because the HDF5 library does not support multiple threads.

```python
# Disable the lock because we are not using a library that is not thread-safe.
from xarray.backends.locks import DummyLock

os.environ["H5NETCDF_READ_BACKEND"] = "pyfive"
ds = xr.open_dataset(
    "tutorial.nc",
    engine="h5netcdf",
    chunks="auto",
    lock=DummyLock(),
)
ds.air
```

Running the experiments above with this dataset seems to suggest that parallism is better on the threaded scheduler, but the code crashed when trying to use the distributed scheduler.

## Bonus material: ERA5 data

If you would like to run the computation above on some real data, you can take a look at the ERA5 data. ERA5 is a popular reanalysis dataset. A reanalysis dataset is produced by a weather or climate model that is contiously fed with observational data to make it match reality as closely as possible, thus combinging many different observations and filling in gaps. We use a version that has been reformatted by Google into an Analysis Ready Cloud Optimized (ARCO) format because it is much easier to work with.


```python
# Data description at: https://github.com/google-research/arco-era5
era5 = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options={"token": "anon"},
)
era5
```

Note that the dataset contains many different variables. We will load only the temperature at 2 meter height:


```python
variable = "2m_temperature"
da = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    drop_variables=[v for v in era5.data_vars if v != variable],
    chunks={},
    storage_options={"token": "anon"},
)[variable]
da
```

```python
plot_map(da.sel(time="2000-01-01 00:00:00"))
```


    
![png](notebook_files/notebook_53_0.png)
    


If you are not running this notebook in the Google cloud (e.g. https://colab.research.google.com), it is probably most convenient to download a part of the dataset above and save it locally for further analysis, as this avoids downloading the same data over and over again. We use a subset here to limit the runtime of the notebook, but this can be extended:


```python
subset = da.loc["2000-01-01 00:00:00":"2000-03-01 00:00:00"]
subset
```


```python
compressor = zarr.codecs.BloscCodec(
    cname="zstd",
    clevel=5,
    shuffle=zarr.codecs.BloscShuffle.shuffle
)
subset.drop_encoding().to_zarr(
    "era5_2m_temperature.zarr",
    mode="w",
    encoding={subset.name: {"compressors": compressor}}
)
```

    <xarray.backends.zarr.ZarrStore at 0x70d665d91800>


```python
ds = xr.open_dataset(
    "era5_2m_temperature.zarr",
    chunks="auto",
)
```


```python
ds["2m_temperature"]
```

```python
climatology = ds.groupby("time.month").mean("time")
climatology["2m_temperature"]
```

```python
with dask.config.set({"scheduler": "threads", "num_workers": 4}):
    %time climatology["2m_temperature"].compute();
```

    CPU times: user 12 s, sys: 3.74 s, total: 15.8 s
    Wall time: 3.03 s

## Dependencies

The above script used the following `pixi.toml`. Pixi defaults to pulling packages from Conda Forge. As far as I can see, all these packages have wheels on PyPI as well though. The use of cartopy could be nice to replace the NY plotting as well.

```toml
# file: pixi.toml
[workspace]
authors = ["Bouwe Andela"]
channels = ["conda-forge"]
name = "parallel-python-xarray-dask-example"
platforms = ["linux-64", "osx-arm64"]
version = "0.1.0"

[tasks]

[dependencies]
cartopy = "*"
dask = "*"
distributed = "*"
flox = "*"
gcsfs = "*"
h5netcdf = "*"
jupyterlab = "*"
kerchunk = "*"
matplotlib = "*"
netcdf4 = "*"
obspec_utils = "*"
obstore = "*"
pooch = "*"
py-spy = "*"
pyfive = "*"
python-graphviz = "*"
samply = "*"
scipy = "*"
virtualizarr = "*"
xarray = "*"
zarr = "*"
```
