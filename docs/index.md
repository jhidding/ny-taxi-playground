---
icon: lucide/rocket
title: The NY Taxi dataset
---

- What precise data to download?
   - [x] PARQUET on nyc.gov: all of 2025
   - CSV from Kaggle
   - Create fake "bad" datasets directly from PARQUET?

- Interesting queries, both in execution complexity and use case relevance.

## Objectives

- Patterns like map and reduce
- Understand delayed evaluation as a concept? (Use existing material)
- Profiling, reading flame graphs: `py-spy` (sampling), `cprof` (instrument profiling), `speedscope`
- Understand and identify IO bound processes
- Have an overview of several libraries for efficient data handling: `pandas`, `polars`, `duckdb`, `dask` / `xarray` (gridded), `dask` / `dataframe`.
- Polars and duckdb merge files transparently.

## Tasks

- Bouwe: generate large dataset, reduce with `xarray` example.
- Flavio: check out duckdb
- Leon: check out polars
- Johan: whatever, dask dataframe and coordination, download script (use pooch?), work out lesson objectives.

### Question for UU

- Can we host the data locally?

# Retrieving data

The following script downloads the PARQUET data files for all of 2025 and generates a Pooch registry file for subsequent downloads.

```python
#| file: src/download_data.py
import pooch
from pathlib import Path

def taxi_filename(year: int = 2025, month: int = 1):
    return f"yellow_tripdata_{year:04}-{month:02}.parquet"

def bootstrap_taxi_data(path: Path = Path() / "data" / "trip-data", year: int = 2025) -> pooch.Pooch:
    return pooch.create(
        path=path,
        base_url="https://d37ci6vzurychx.cloudfront.net/trip-data/",
        registry={taxi_filename(year=year, month=month): None for month in range(1, 13)})

def download_all(p: pooch.Pooch):
    for filename in p.registry.keys():
        p.fetch(filename)

def make_registry(p: pooch.Pooch):
    registry_name = p.path.name + "-registry.txt"
    pooch.make_registry(p.path, p.path.parent / registry_name)
    
def taxi_data(data_path: Path = Path() / "data", name: str = "taxi-data") -> pooch.Pooch:
    path = data_path / name
    registry = data_path / (name + "-registry.txt")
    
    pooch.create(
        path=path,
        base_url="https://d37ci6vzurychx.cloudfront.net/trip-data/",
        registry=None)
    pooch.load_registry(registry)
 
if __name__ == "__main__":
    registry = Path() / "data" / "taxi-data-registry.txt"
    
    if registry.exists():
        p = taxi_data()
        download_all(p)
    else:
        p = bootstrap_taxi_data()
        download_all(p)
        make_registry(p)
```
