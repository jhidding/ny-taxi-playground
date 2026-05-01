# ~/~ begin <<docs/duckdb.md#src/duckdb_and_parquet.py>>[init]

import duckdb
from pathlib import Path
from timeit import timeit

from download_data import taxi_data, taxi_filename
from utils import get_project_root

project_root = get_project_root()


# ~/~ end
# ~/~ begin <<docs/duckdb.md#src/duckdb_and_parquet.py>>[1]
def get_thread_settings(con):
    settings = con.sql(
        "select name, value, description from duckdb_settings() where name like '%thread%'"
    )
    print(settings)


# ~/~ end
# ~/~ begin <<docs/duckdb.md#src/duckdb_and_parquet.py>>[2]
duckdb_conn = duckdb.connect()
rel = duckdb_conn.from_parquet(str(project_root / "data/trip-data/"))
# ~/~ end
# ~/~ begin <<docs/duckdb.md#src/duckdb_and_parquet.py>>[3]
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

# ~/~ end
# ~/~ begin <<docs/duckdb.md#src/duckdb_and_parquet.py>>[4]
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
# ~/~ end
# ~/~ begin <<docs/duckdb.md#src/duckdb_and_parquet.py>>[5]
pickups = rel.select(
    "PULocationID, total_amount, monthname(tpep_pickup_datetime) as month"
)

pickups_filtered = pickups.filter("PULocationID = 229")

# this executes b/c of distinct()
months = pickups_filtered.select("month").distinct()

if __name__ == "__main__":
    print(pickups.limit(3))
    print(months)

# ~/~ end
# ~/~ begin <<docs/duckdb.md#src/duckdb_and_parquet.py>>[6]
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

# ~/~ end
