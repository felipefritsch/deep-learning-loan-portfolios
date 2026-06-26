# W1 — Scale up the three-arm sequence comparison (final-week plan)

Re-run the M27a three-arm rolling comparison (`ff_base` / `ff_hist` / `gru`) at an **elevated
per-window training sample `S`**, so the loan-level memory result is a firm headline rather than a
1.5M-point read. Same pipeline as `M27a_NOTES.md §Reproduce`; the only change is the sample knob.

## The knob

`build_window.py` now reads `train_n` from the **`SEQ_TRAIN_N`** env var (default `1500000`,
byte-identical to the banked build). It rebuilds a window's cache automatically if the existing
cache's `train_n` differs from the requested one (clears the stale dir first). `build_windows_batch.py`
shells `build_window.py`, so the env var propagates to the whole batch.

## Choosing `S`

- **Current:** 1.5M points/window (uniform train_pool sample + HT weights).
- **Recommended elevated `S`:** **10M/window** — a ~6.7× increase, a clearly "more meaningful"
  sample, and ≳ the advisor's 20–25% floor for the late windows. Set `SEQ_TRAIN_N=10000000`.
- **Strongest (if pod disk + GPU-days allow):** the **full per-window train pool** — pass a very
  large `SEQ_TRAIN_N` (e.g. `100000000`); `prediction_points` caps at the pool size, so each window
  uses its entire (thinned, HT-weighted) train pool. Late windows (train ≤ Dec 2023) are the largest.
- **Transformer (W2):** use the **same `S`** for a clean 4-arm comparison (your stated preference);
  fall back to 20–25% / key windows only if GPU time runs short.

**Scaling rule of thumb:** cache disk and per-epoch train time are ~**linear in `S`**; the val/test
slices are already FULL (84.5M test) so **scoring time is unchanged**. At 10× the sample expect ~10×
the cache GB/window and ~10× per-epoch seconds (epochs-to-early-stop may fall). Check pod free disk
before launching `S` = full.

## Preserve the banked 1.5M result (do this first)

The banked numbers of record are **committed** in `src/floan/model/m27a_results/` (per-window JSON)
and summarised in `M27a_NOTES.md` — re-running at `S` does **not** touch them. The pod scratch dirs
**are** overwritten, and the trainers are resumable (skip if artifacts exist), so clear scratch before
the `S` run or it will reuse stale 1.5M outputs:

```bash
# on the pod, with $OUTPUTS = the seq cache root (ROOT symlink)
mv  "$OUTPUTS/seq_cache/full"   "$OUTPUTS/seq_cache/full_1p5m"      # keep, or rm -rf
mv  "$OUTPUTS/m27a_gpu_runs"    "$OUTPUTS/m27a_gpu_runs_1p5m"        # GRU artifacts (resumable→skip)
# ff_arms.py outputs likewise — move/clear its results dir before re-running
```

## Launch (pod: panel + pools + macro present, GPU attached)

```bash
export SEQ_TRAIN_N=10000000          # <- set S here (10M shown; or 100000000 for full per-window)

# 1) Build the per-window sequence caches at S (resumable, 2 concurrent, ~25GB RSS/window)
python scripts/m27a/build_window.py 2015
python scripts/m27a/build_window.py 2016
python scripts/m27a/build_windows_batch.py        # k=2017..2025
python scripts/m27a/build_history_superset.py     # ff_hist history (k=2025 superset → symlinks)

# 2) Train + score (frozen config; cuda; resumable per (k,seed))
python scripts/m27a/gpu_train_all.py              # GRU: 11 windows seed 0 + key-window seeds 1,2
python scripts/m27a/ff_arms.py                    # ff_base / ff_hist arms

# 3) Aggregate → tables
python scripts/m27a/summary_three_arm.py          # ff_base / ff_hist / gru + deltas + pooled
python scripts/m27a/summary_gru.py                # GRU rolling + ensemble + pooled
```

**After every GPU session:** `scripts/backup_ssd.sh` + `git push` (standing rule 5 — sync
checkpoints off the rented box before teardown).

## What feeds the draft

`summary_three_arm.py` regenerates the Ch.4 `tab:seqrolling` numbers (and the σ-deltas) at `S`. Swap
them in where the `[PLANNED]` flag sits at the end of `subsec:results-seqrolling`, and replace the
"sample-sensitive" caveat with the elevated-`S` read. The fallback (if the `S` run is not finished in
time) is the committed 1.5M result, which already stands.
