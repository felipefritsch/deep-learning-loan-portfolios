"""Build ONE rolling window's sequence cache (full, k=argv[1]) from the panel on the GPU pod.

Reuses sequence.build_cache; runtime knobs only (no config edit). Skips if the cache is already
complete (meta.json present) so a batch is resumable. DuckDB memory is held at 24GB so two of
these can run concurrently without oversubscribing a 124GB box (3x48GB OOM-killed earlier runs).
Concurrency safety of verify_sample()'s round-trip scratch is handled in sequence.py itself
(per-call mkdtemp), so no isolation hack is needed here.

  python scripts/m27a/build_window.py 2016
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

from floan.model import config as mc
mc.DUCKDB_MEMORY_LIMIT = "24GB"          # 2 concurrent builds stay within the box

OUTPUTS = mc.OUTPUTS                       # resolves to the pod cache root (via the ROOT symlink)
# W1 scale-up knob (final_week_plan.md). Default 1_500_000 keeps the banked M27a build
# byte-identical (reproducibility); set SEQ_TRAIN_N to rebuild caches at the elevated sample S.
TRAIN_N = int(os.environ.get("SEQ_TRAIN_N", "1500000"))


def main():
    from floan.model import sequence as S
    k = int(sys.argv[1])
    out = OUTPUTS / "seq_cache" / "full" / f"k{k}"
    meta_p = out / "meta.json"
    if meta_p.exists():
        existing = json.loads(meta_p.read_text()).get("train_n")
        if existing == TRAIN_N:
            print(f"[k{k}] SKIP — cache with train_n={TRAIN_N:,} already complete at {out}", flush=True)
            return
        # A different sample size was requested: clear the stale cache and rebuild cleanly
        # (avoids mixing shards from a previous train_n). Banked numbers live in committed
        # src/floan/model/m27a_results/, so overwriting pod scratch is safe.
        print(f"[k{k}] REBUILD — existing cache train_n={existing:,} != requested {TRAIN_N:,}; "
              f"clearing {out}", flush=True)
        shutil.rmtree(out, ignore_errors=True)
    print(f"[k{k}] BUILD -> {out}  train_n={TRAIN_N:,} mem={mc.DUCKDB_MEMORY_LIMIT} chunk=2,000,000", flush=True)
    t0 = time.perf_counter()
    meta = S.build_cache("full", k, train_n=TRAIN_N, T=12, chunk=2_000_000,
                         out_root=str(out), verify=True)   # verify=True -> flip-test (raises on fail)
    wall = time.perf_counter() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # KB -> GB
    org = meta["points_by_origin"]["train"]; tn = meta["points"]["train"]
    (out / "_pod_build.json").write_text(json.dumps({
        "k": k, "wall_s": wall, "peak_rss_gb": peak, "flip_pass": True,
        "points": meta["points"], "bytes": meta["bytes"], "shards": meta["shards"],
        "train_origin": org, "current_frac": org["current"] / tn,
    }, indent=2))
    print(f"[k{k}] DONE wall={wall:.0f}s peak={peak:.1f}GB current={100*org['current']/tn:.1f}%",
          flush=True)


if __name__ == "__main__":
    main()
