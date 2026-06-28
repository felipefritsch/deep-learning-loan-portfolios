# scripts — runnable drivers

Operational entry points that orchestrate the package: convenience wrappers for the local runs and
the multi-machine pipeline for the sequence models (which trains on a rented GPU and needs per-window
sequence caches built first). The wrappers compute the repo root automatically and invoke
`python -m floan.…`.

### Local helpers
| File | What it does |
|---|---|
| `backup_ssd.sh` | mirrors the small, irreplaceable SSD artifacts (models, scalers, metrics, QA reports, manifests) to a gitignored `ssd_mirror/` on the internal disk — run after every milestone gate and GPU session |
| `run_backtest.sh` | runs the rolling-window backtest across all windows |
| `run_depth_check.sh` | the full-scale depth re-check on the tuning window (depth 3 vs 5) |
| `run_gbt_sweep.sh` | the gradient-boosted-tree hyperparameter sweep |

### `m27a/` — the sequence-model rolling pipeline
Build per-window sequence caches, train + score the GRU / transformer / feed-forward baselines on the
GPU, and aggregate the rolling comparison tables. (Named `m27a` after the task that introduced it; it
is the **sequence-model training + evaluation** pipeline.)

| File | What it does |
|---|---|
| `build_window.py` | builds ONE rolling window's sequence cache (the trailing-T tensors + scaler/vocab) from the panel |
| `build_windows_batch.py` | orchestrates the per-window cache builds, memory-guarded and resumable |
| `build_history_superset.py` | builds the engineered-history feature cache (a superset reused across windows) |
| `train_gru.py` (`gpu_train_all.py`) | trains + scores the **GRU** across all 11 windows (seed 0) plus extra seeds on the key windows |
| `train_transformer.py` (`gpu_train_transformer.py`) | trains + scores the **transformer** arm on the same caches (the GRU trainer with the architecture swapped) |
| `ff_arms.py` | trains + scores the two feed-forward baselines (current-state-only and current-state-plus-engineered-history) on the **same** per-window sample, so the three-arm comparison is confound-free |
| `summary_gru.py` | aggregates the GRU rolling table (per-window NLL + key-transition AUC, ensemble on key windows) |
| `summary_three_arm.py` | aggregates the three-arm comparison (baseline / +history / GRU, with the per-window and pooled deltas) |
| `run_seq_rolling.sh` (`run_w1_all.sh`) | the detached, connection-independent driver: build caches → gate → train+score → summaries, end to end |
| `run_transformer_roll.sh` (`finish_xf.sh`) | the detached driver for the transformer roll: train → harvest results → commit/push |

### `m27b/` — sequence-model pricing
| File | What it does |
|---|---|
| `price_h1.py` | the one-month-horizon GRU pool-pricing driver across anchors, with the H=1 identity guard |

> The filenames in parentheses are the current on-disk names; the plain names are the descriptive
> targets in the planned restructure (`notes/REPO_RESTRUCTURE.md`). Runbooks (`W1_SCALEUP.md`,
> `POST_TRANSFORMER.md`) document specific multi-day runs.
