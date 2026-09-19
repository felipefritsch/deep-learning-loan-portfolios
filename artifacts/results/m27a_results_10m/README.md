# Sequence-model rolling results (10M training sample)

Committed result artifacts for the **memory / path-dependence comparison** across the 11 rolling
windows (2015–2025) — the JSONs the dissertation's Chapter 4 sequence tables are built from. The
sibling folder `../m27a_results/` holds the same comparison at the earlier **1.5M** training sample;
this folder is the scaled-up **10M-per-window** run that became the headline.

All arms are trained on the *same* per-window sample and scored on the *same* full frozen test slice
(84.5M pooled rows), so a difference is attributable to the architecture alone.

## The four model arms
| File pattern | Arm |
|---|---|
| `k{year}_ff_base.json` | feed-forward net, **current-state only** (strictly first-order Markov) |
| `k{year}_ff_hist.json` | feed-forward net **+ engineered one-step history** summaries |
| `k{year}_s{seed}.json` | the **GRU** over the trailing 12-month sequence (seed 0 all windows; seeds 1–2 on the key windows) |
| `k{year}_xf_s{seed}.json` | the **transformer** over the same sequence |

## JSON schema (per run)
```
k, seed, model        # window (test year), seed, architecture
n_test                # frozen test rows scored
test_nll, val_nll     # out-of-sample / validation negative log-likelihood (the loss of record)
best_epoch, eff_epochs
test_auc              # list of one-vs-rest AUCs per (origin → destination) transition
load_s, train_s, score_s, peak_gpu_gb   # timings + GPU footprint
```

## Summary tables (regenerated from the JSONs)
- `summary_three_arm.txt` — baseline / +history / GRU test NLL per window + the `gru−base` and
  `gru−hist` deltas + pooled (the value of memory, and the learned-vs-engineered increment).
- `summary_gru.txt` — the GRU rolling table (single-seed NLL, key-transition AUC, ensemble on key
  windows, 3-seed mean±sd).

Regenerate with `scripts/m27a/summary_three_arm.py` and `summary_gru.py`.
