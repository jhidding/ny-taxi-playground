# ~/~ begin <<docs/index.md#src/download_data.py>>[init]
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
# ~/~ end
