import time
from pathlib import Path
from floan.model import config as mc
mc.DUCKDB_MEMORY_LIMIT = "48GB"
from floan.model import history as H
t0 = time.perf_counter()
out = H.build("full", 2025)   # superset: period_ym < test_bounds(2025)[1]=202512, shards 0..63
print(f"k2025 history built in {time.perf_counter()-t0:.0f}s at {out}")
