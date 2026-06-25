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
import resource
import sys
import time
from pathlib import Path

from floan.model import config as mc
mc.DUCKDB_MEMORY_LIMIT = "24GB"          # 2 concurrent builds stay within the box

OUTPUTS = mc.OUTPUTS                       # resolves to the pod cache root (via the ROOT symlink)


def main():
    from floan.model import sequence as S
    k = int(sys.argv[1])
    out = OUTPUTS / "seq_cache" / "full" / f"k{k}"
    if (out / "meta.json").exists():
        print(f"[k{k}] SKIP — meta.json already exists at {out}", flush=True)
        return
    print(f"[k{k}] BUILD -> {out}  mem={mc.DUCKDB_MEMORY_LIMIT} chunk=2,000,000", flush=True)
    t0 = time.perf_counter()
    meta = S.build_cache("full", k, train_n=1_500_000, T=12, chunk=2_000_000,
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
