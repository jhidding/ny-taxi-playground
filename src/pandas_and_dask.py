# ~/~ begin <<docs/pandas.md#src/pandas_and_dask.py>>[init]
import pandas
from pathlib import Path

from download_data import taxi_data, taxi_filename
from utils import get_project_root

project_root = get_project_root()

data = taxi_data(project_root / "data")
jan = pandas.read_parquet(data.fetch(taxi_filename(year=2025, month=1)))
jan

if __name__ == "__main__":
    pass
# ~/~ end
