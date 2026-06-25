"""Orchestrate windows k=2017..2025: <=3 concurrent single-window builds, resumable,
memory-guarded (auto-drop to 2 if MemAvailable headroom < one window)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from floan.model import config as mc

KS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
OUT = mc.OUTPUTS / "seq_cache" / "full"
MAXC = 2                  # 2 concurrent: 3-way @48GB DuckDB OOM-killed k2021/22/23; now 24GB/2-up
HEADROOM_GB = 28          # need ~one window's worth of real (anon) headroom to add a slot
PY = sys.executable

def mem_avail_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1e6
    return 0.0

def k2016_inflight() -> bool:
    # the externally-launched k=2016 occupies a slot until its meta.json lands
    return not (OUT / "k2016" / "meta.json").exists()

def main():
    todo = [k for k in KS if not (OUT / f"k{k}" / "meta.json").exists()]
    skipped = [k for k in KS if k not in todo]
    print(f"=== BATCH build windows {KS} ===", flush=True)
    print(f"  already complete (skip): {skipped}", flush=True)
    print(f"  to build: {todo}", flush=True)

    running: dict[int, subprocess.Popen] = {}
    i = 0
    while i < len(todo) or running:
        # total concurrent builds (mine + in-flight k2016) must stay <= MAXC; the 3rd
        # also needs real memory headroom (else hold at 2 -> auto-drop under pressure).
        while True:
            slots = len(running) + (1 if k2016_inflight() else 0)
            if not (i < len(todo) and slots < MAXC and
                    (slots < 2 or mem_avail_gb() > HEADROOM_GB)):
                break
            k = todo[i]; i += 1
            here = Path(__file__).resolve().parent
            log = open(here / f"build_k{k}.log", "w")
            p = subprocess.Popen([PY, "-u", str(here / "build_window.py"), str(k)],
                                 stdout=log, stderr=subprocess.STDOUT)
            p._log = log  # type: ignore[attr-defined]
            running[k] = p
            print(f"  [{time.strftime('%H:%M:%S')}] launched k{k} pid={p.pid}  "
                  f"(running={sorted(running)}, mem_avail={mem_avail_gb():.0f}GB)", flush=True)
        # reap finished
        for k in [k for k, p in running.items() if p.poll() is not None]:
            p = running.pop(k); p._log.close()  # type: ignore[attr-defined]
            tag = "OK" if p.returncode == 0 else f"FAIL rc={p.returncode}"
            print(f"  [{time.strftime('%H:%M:%S')}] k{k} finished: {tag}  "
                  f"(running={sorted(running)})", flush=True)
        time.sleep(10)
    print(f"=== BATCH COMPLETE at {time.strftime('%H:%M:%S')} ===", flush=True)

if __name__ == "__main__":
    main()
