# M10b — Rolling backtest loop (GPU half of M10)

Runs the frozen NN config over all 11 rolling windows on the **full** export, plus the
8-net ensemble on the 5 key windows, with the per-window logit floor. Executed on a RunPod
**RTX A4500 (20 GB)** — a different card from M8b's RTX 4090. All M10 Accept criteria PASS
(evidence in `logs/m10b/verify_final2.log`; `backtest.py --verify-only --device cuda`).

## 1. Preliminaries

### (2) Throughput re-measure + the `train.py` GPU-resident fast path
M8b sized M10 against the RTX 4090 (~1.55 M rows/s). The first full-scale fit on this A4500
ran at **~0.35 M rows/s (97 s/epoch on the 33.5 M-row k=2015 slice), GPU ~10 %** — the dev-era
per-batch path (`preload_split` → NumPy fancy-index → host→device every minibatch) is CPU/
transfer-bound, not compute-bound. At that rate the ~60-fit loop is ~40 h.

Fix: a **GPU-resident** path in `train.py` (`--gpu-resident`, cuda-only; the dev/CPU path is
untouched). The whole index-encoded split is uploaded once (`to_resident`: `cat` as int32 to
halve its footprint) and minibatches are indexed **on the GPU**; the minibatch order is the
identical `np.random.default_rng([seed, epoch]).permutation` the NumPy path uses, and loss/
eval are the same weight-averaged-CE / float64-NLL (`torch_common`). Memory fits 20 GB at every
window (k=2015 ≈ 8 GB → k=2025 ≈ 16 GB). Result: **~0.80 M rows/s (41 s/epoch), 2.3×.**

**This is a post-M9 modification of train.py, made a methodological non-event by bit-identical
output.** Same config (k=2015, d3/do0.2/wd0), naive vs resident, first two epochs:

| epoch | path | train_nll | val_nll | rows/s |
|---|---|---|---|---|
| 0 | naive    | 0.147757 | 0.097586 | 345,278 |
| 0 | resident | 0.147757 | 0.097586 | 797,930 |
| 1 | naive    | 0.138924 | 0.096596 | 351,903 |
| 1 | resident | 0.138924 | 0.096596 | 808,822 |

The NLLs match to all printed digits — only tensor residence changed, not the loss, order,
RNG, checkpoint, or metrics. (Naive: `logs/m10b/depth_d3_do0.2_wd0.log`; resident:
`logs/m10b/bench_resident_d3.log`.) Per-epoch atomic checkpointing + resume are unchanged, so a
tmux disconnect / pod preemption costs at most the in-progress epoch.

### (1) Full-scale depth re-check → frozen config
M9 selected depth 3 on the 3.15 M-row dev slice and flagged it **scale-sensitive**. Re-fit on
the full k=2015 slice (33.5 M rows, 10.6× dev), depth 3 vs 5 at dropout 0.2, ±L2 1e-5,
selecting on val NLL:

| cell | val_nll | test_nll | params |
|---|---|---|---|
| **d3 do0.2 L2=1e-5** | **0.095723** ◄ val argmin | 0.103125 | 170,821 |
| d3 do0.2 L2=0       | 0.095746 | 0.103091 | 170,821 |
| d5 do0.2 L2=0       | 0.095912 | 0.103142 | 210,301 |
| d5 do0.2 L2=1e-5    | 0.096020 | 0.102970 | 210,301 |

**Depth 3 still beats depth 5 decisively** (val gap +1.66e-4, ~7× the within-depth-3 L2
spread) — the M9 selection is *not* a dev-scale artefact. Per the stated protocol ("take the
val-NLL winner") the frozen config is the strict argmin **d3 / dropout 0.2 / L2 1e-5**
(`config.NN_SELECTED`); the L2 sub-choice over L2=0 is within noise (2.3e-5; test marginally
favours L2=0) and immaterial. Evidence: `models/nn/full/k2015_d{3,5}_do0.2_wd{0,1e-05}/`,
`logs/m10b/depth_check.log`.

## 2. Logit equivalence receipt (chain-of-custody)
`backtest.py`'s per-window logit is a new implementation — a 0-hidden-layer net over
`[cont ‖ one-hot(cat) ‖ bin]` parameterised as `Linear(cont‖bin) + Σ_c Embedding(vocab_c, 7)`
(the embedding-sum **is** the full-one-hot multinomial logit, but avoids the ~1.4k-wide design
so it rides the resident GPU path). Only the M7 path (`logit.py`, drop-first one-hot) carries
the sklearn cross-check, so we reproduced it: re-fit the new logit across M7's full L2 grid on
dev k=2015 (`verify_logit_equiv.py`). After zero-initialising the categorical embeddings (the
faithful one-hot start — categorical weights begin at 0), the new logit matches M7 to **<1e-4
across the whole grid** (wd=0 provable-equivalence anchor **|Δ|=5.0e-5**; selected-config test
**|Δ|=5.8e-5**). Receipt: `models/logit/logit_equiv_receipt.json`.

## 3. The loop + Accept evidence
`backtest.py` over k=2015…2025: per-window logit + frozen single NN everywhere; 8-net ensemble
on {2015, 2019, 2020, 2023, 2025}. Per-window scaler/vocab/incentive re-derived from that
window's train slice; early stopping on that window's val only. Ran in **tmux** with per-epoch
checkpointing (loop idempotent — finished runs skipped, interrupted runs resume). Throughput
held ~0.80 M rows/s; no OOM (peak ~16 GB). k=2025 (the largest window) ensemble dominated wall
clock; total run ~20 h.

**Headline (test NLL, per window):** the NN beats the logit in **11/11 windows** (mean
0.10255 vs 0.10937), holding through the COVID-2020 and 2022–23 rate-spike regimes; the
ensemble edges the single NN further on the key windows.

| k | logit | NN | ensemble |
|---|---|---|---|
| 2015 | 0.108246 | 0.103125 | 0.102662 |
| 2016 | 0.116390 | 0.109378 | — |
| 2017 | 0.105750 | 0.097431 | — |
| 2018 | 0.098172 | 0.088404 | — |
| 2019 | 0.106352 | 0.100600 | 0.100479 |
| 2020 | 0.191949 | 0.184869 | 0.183414 |
| 2021 | 0.149628 | 0.141729 | — |
| 2022 | 0.090422 | 0.085874 | — |
| 2023 | 0.074943 | 0.069296 | 0.069318 |
| 2024 | 0.079053 | 0.072203 | — |
| 2025 | 0.082164 | 0.075157 | 0.075095 |

**M10 Accept (all PASS, `logs/m10b/verify_final2.log`):**
1. 11×{logit, NN} + 5×ensemble run folders exist.
2. Per-window early stopping used that window's val only (best_epoch < n_epochs, val-selected).
3. Scaler/vocab locality: k=2015 vs k=2020 scaler centres differ (max|Δ|=7.35); vocab expands
   with the train slice (Σ 1506→1514) — per-window re-fit, no leakage.
4. **Ensemble ≥ best single net** — gated on ensemble ≤ **mean single-member** test NLL (the
   paper's Fig-7 variance-reduction claim, single-draw luck averaged out): **PASS 5/5**.
   Reported per window vs the deployed single net too: ensemble lower in 4/5; k=2023 is a
   +2.2e-5 (0.03 %) tie within seed noise — that window's seed-0 single draw was unusually
   strong on its test slice (gating on min-member would be selection-on-test, unselectable
   out-of-sample).
5. **Base-rate QA (§7):** ensemble mean predicted base rates ≈ realized (max|Δ| ≤ 0.0086 < 0.01,
   the §6.4 importance-weighting check); structural-impossible-transition mass < 1e-3 in every
   key window, **tracking the realized rate** (e.g. k=2015 model 2.5e-4 vs realized 2.4e-4 — no
   hallucinated mass). Deviation noted per the standing rule: this Fannie monthly panel does
   contain rare nominally-"impossible" jumps (notably `dpd_90plus→REO`, ~1–2 %, where a
   foreclosure+REO completes inside a reporting gap), so the test is calibration to that signal,
   not a strict zero.

## 4. Artefacts
- Code: `backtest.py`, `train.py` (+`--gpu-resident`), `config.py` (`NN_SELECTED` frozen),
  `verify_logit_equiv.py`, `run_backtest.sh`, `run_depth_check.sh`.
- Runs (SSD `models/`, synced via `backup_ssd.sh`): `nn/full/k20{15..25}_d3_do0.2_wd1e-05/`,
  `nn/full/ens_k{2015,2019,2020,2023,2025}_s{0..7}/`, `nn/full/ensemble_k*_summary.json`,
  `nn/full/backtest_summary.json`, `logit/full/k20{15..25}/`, `logit/logit_equiv_receipt.json`.
- Logs (local/SSD, gitignored): `logs/m10b/{depth_check,backtest,verify_final2,logit_equiv2}.log`.
