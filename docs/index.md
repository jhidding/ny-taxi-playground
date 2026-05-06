---
icon: lucide/rocket
title: The NY Taxi dataset
---

The initiative is to use the quite famous New York taxi trip data to explain several concepts on the topic of efficient data handling. The dataset is publicly accessible and has a moderate size, which should allow us to demonstrate the efficacy of several different approaches.

The material developed here will replace the latter chapters in the [Parallel Python lesson](https://carpentries-incubator.github.io/lesson-parallel-python/), currently in the [Carpentries](https://carpentries.org/) Incubator.

## Objectives

#### Reason about promises/futures in the context of parallel programming. 

- Explain the danger of shared mutable state in parallel programs.
- Use the concept of delayed evaluation to change how a computation is scheduled.
- Apply `dask.delayed` to visualize a problem.
- Apply `concurrent.futures` to speed-up a computation.

#### Understand the applicability of broad-stroke techniques like `map` and `reduce`.

- Recognize algorithmic mapping and reduction in problems.
- Visualize map/reduce workflows with `dask`.
- Apply map/reduce on a large data set.

#### Identify weak points in a computational process using profiling.

- Explain how the efficiency of a process can be bound by IO.
- Identify an IO bound process using a profiler.
- Read a flame graph, using `speedscope`.
- Explain the difference between a sampling (e.g. `py-spy`) and instrument profiler (e.g. `cProfile`).

#### Apply higher order data handling frameworks to solve problems.

- List several popular libraries for efficient data handling: `pandas`, `polars`, `duckdb`, `dask` / `xarray`, `dask` / `dataframe`.
- Assess the applicability of different data formats: `csv`, `parquet`, `netcdf`, `npy`.
- Discuss the importance of data standards with regards to FAIR principles.
- Use `polars` or `duckdb` to handle heterogeneous and partitioned data sets transparently.
- Evaluate the efficiency of different data handling frameworks

## Tasks

To reach the above lesson objectives we can think of several tasks.

- Summary visualisations:
    - one year histogram showing ride frequency
    - identify traffic hot spots (do we visualize with the given shape files?)
- Identify commute zones, i.e. is there an assymetry between traffic directions during morning and afternoon rush hours?
- Are there significant spikes in traffic that can be attributed to a single target location?

## Planning

- Bouwe: generate large dataset, reduce with `xarray` example.
- Flavio: check out duckdb
- Leon: check out polars
- Johan: whatever, dask dataframe and coordination, download script (use pooch?), work out lesson objectives.


## Discussion with UU

- Can we host the data locally?
    - yes, they will organize this

### Lesson planning
- Consider which part of existing material to keep or not
    - map/reduce yes or no?
    - the part on multithreading for python code should stay for sure, but it 
        can stay as a simple exercise
- schedule half a day for exercises
    - parallel track: let people work on gridded or tabular data
- general ideas on the exercises
    - they should be more substantive than (at least the duckdb example) is now,
    along the lines Johan suggested earlier; visualization would be nice
- comments on xarray example
    - different speedup on different laptop - why? 
    - explain that it's important to profile/optimize on the "production" hardware
- comments on duckdb example
    - add data exploration in the beginning. also highlight why we need parallelization that it's big
    - not everyone familiar with sql -- have polars alongside with ideally the same 
      exercise content
    - explain why speed-up of multithreading in duckdb is sub-linear
    - consider combining the multithreading and the lazy evaluation part?
- Code profiling
    - I experimented with py-spy. For didactical purposes, we think it's better 
    to profile a "pure" python script and not one where the heavy work is done in
    an external library such as duckdb. Reason: it's easier to map between
    components in the flamegraph and compnents in the code.


#### Nexts steps
- Johan will take a look, coordinate and let UU develop further
- What is the intention with entangled for teaching?



Entangled starter
-----------------

Run

```bash
uv run entangled watch
```

Create a new markdown file in the `docs` directory. You can annotate code blocks to create Python files:

~~~markdown
For example, this is "Hello, World" in Python:

```python
#| file: src/hello.py
print("Hello, World!")
```
~~~

You can also give snippets a name using a `#| id: name` tag at the top of a code block, and then `<<name>>` inside another code block to insert its contents there.

Zensical
--------

The contents of `docs` is rendered into HTML pages using the `zensical` engine. To preview locally:

```bash
uv run zensical serve
```

## Retrieving data

The following script downloads the PARQUET data files for all of 2025 and generates a Pooch registry file for subsequent downloads. Run this script using `uv run python src/download_data.py` from a command line in the project root.

```python
#| file: src/download_data.py
import pooch
from pathlib import Path


def taxi_filename(year: int = 2025, month: int = 1):
    return f"yellow_tripdata_{year:04}-{month:02}.parquet"


def bootstrap_taxi_data(
    path: Path = Path() / "data" / "trip-data", year: int = 2025
) -> pooch.Pooch:
    return pooch.create(
        path=path,
        base_url="https://d37ci6vzurychx.cloudfront.net/trip-data/",
        registry={
            taxi_filename(year=year, month=month): None for month in range(1, 13)
        },
    )


def download_all(p: pooch.Pooch):
    for filename in p.registry.keys():
        p.fetch(filename)


def make_registry(p: pooch.Pooch):
    path = Path(p.path)
    registry_name = path.name + "-registry.txt"
    pooch.make_registry(path, path.parent / registry_name)


def taxi_data(
    data_path: Path = Path() / "data", name: str = "trip-data"
) -> pooch.Pooch:
    path = data_path / name
    registry = data_path / (name + "-registry.txt")

    p = pooch.create(
        path=path,
        base_url="https://d37ci6vzurychx.cloudfront.net/trip-data/",
        registry=None,
    )
    p.load_registry(registry)
    return p


if __name__ == "__main__":
    registry = Path() / "data" / "trip-data-registry.txt"

    if registry.exists():
        p = taxi_data()
        download_all(p)
    else:
        p = bootstrap_taxi_data()
        download_all(p)
        make_registry(p)
```

## Utils

```python
#| file: src/utils.py
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


```

License
-------

Copyright 2026 Netherlands eScience Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
