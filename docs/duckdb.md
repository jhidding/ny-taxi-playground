---
icon: lucide/database
title: DuckDB
---

**Entangled questions**

- Can I set the python file globally?
- Can I print things/show output in the rendered html?
- I did not get the `<<...>>` syntax to work

### Overview

- Has different python APIs, among which:
    - [Standard DB API](https://duckdb.org/docs/current/clients/python/dbapi)
    - [Relational API](https://duckdb.org/docs/current/clients/python/relational_api)
- For both, it's useful to be familiar with SQL
- If you prefer a dataframe API, polars may be better



DuckDB works with database connections that are initiated as follows

```python
my_con = duckdb.connect()
my_con.close()
```

The default is an in-memory database. You can also connect to a file 
with an existing duckdb database.


### Setup

Here we'll use the relational API and its *lazy evaluation* feature. 
This allows us to chain queries and only compute them when we need them.

```python
#| file: src/duckdb_and_parquet.py

import duckdb
from pathlib import Path
from timeit import timeit

from download_data import taxi_data, taxi_filename
from utils import get_project_root

project_root = get_project_root()


```


```python
#| file: src/duckdb_and_parquet.py
def get_thread_settings(con):
    settings = con.sql(
        "select name, value, description from duckdb_settings() where name like '%thread%'"
    )
    print(settings)


```


Set up the DuckDB connection and explore the settings for parallelism.

```python
#| file: src/duckdb_and_parquet.py
duckdb_conn = duckdb.connect()
rel = duckdb_conn.from_parquet(str(project_root / "data/trip-data/"))
```


### Task 1: Explore how amounts vary by month and hour of the day

```python
#| file: src/duckdb_and_parquet.py
times_and_amounts = rel.select("""
    hour(tpep_pickup_datetime) as hour,
    monthname(tpep_pickup_datetime) as month, 
    payment_type,
    total_amount
""")

means = times_and_amounts.aggregate("mean(total_amount), month, hour", "month, hour")

if __name__ == "__main__":
    # Only now it's executed
    df = means.pl()
    print(df.head())

```

How long does it take with 8 vs 1 thread?

```python
#| file: src/duckdb_and_parquet.py
if __name__ == "__main__":
    get_thread_settings(duckdb_conn)
    n_repeat = 5

    timing = timeit("means.pl()", "from __main__ import means", number=n_repeat)
    print(f"Timing with default number of threads: {timing / n_repeat}")  # 1.5s

    duckdb_conn.sql("set threads to 1")
    get_thread_settings(duckdb_conn)
    timing = timeit("means.pl()", "from __main__ import means", number=n_repeat)
    print(f"Timing with 1 thread: {timing / n_repeat}")  # 5-7s

# Reset for the future
duckdb_conn.sql("set threads to 8")
```

**Explanation**

DuckDB parallelizes the query across multiple threads, leading to faster
execution.



### Task 2: Query specific rows of the data

Goal: Find all trips starting from location 229.

We double-check that the location appears in all months and thus in all 
parquet files.

```python
#| file: src/duckdb_and_parquet.py
pickups = rel.select(
    "PULocationID, total_amount, monthname(tpep_pickup_datetime) as month"
)

pickups_filtered = pickups.filter("PULocationID = 229")

# this executes b/c of distinct()
months = pickups_filtered.select("month").distinct()

if __name__ == "__main__":
    print(pickups.limit(3))
    print(months)

```

Compare timing when applying filter on the lazy relation vs on the loaded data.


```python
#| file: src/duckdb_and_parquet.py
import polars as pl

if __name__ == "__main__":
    get_thread_settings(duckdb_conn)
    timing_with_lazy = timeit(
        "pickups_filtered.pl()", globals=globals(), number=n_repeat
    )
    print(f"Timing with lazy: {timing_with_lazy / n_repeat}")  # ~0.6s

    timing_without_lazy = timeit(
        "pickups.pl().filter(pl.col('PULocationID') == 229)",
        globals=globals(),
        number=n_repeat,
    )
    print(f"Timing with eager: {timing_without_lazy / n_repeat}")  # ~2.5s

```

**Explanation**

DuckDB can apply the filters *before* loading the parquet data into memory.
This is called predicate pushdown and is a key feature of the parquet file format.
Native polars has similar functionality to DuckDB, but the `.pl()` function in duckdb
loads the entire query result into a polars dataframe.



### Other Observations
- Found this threading in python: https://duckdb.org/docs/current/guides/python/multiple_threads, but probably not useful here
- Show/Explore the use of streaming for batch processing?


