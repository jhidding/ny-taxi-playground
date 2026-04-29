# ~/~ begin <<docs/index.md#src/utils.py>>[init]
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent
# ~/~ end
