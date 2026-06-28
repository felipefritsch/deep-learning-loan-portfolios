# W3B — Monte-Carlo H>1 pricing for the sequence models (GRU + transformer)

**Goal.** Price the **GRU and the transformer** at the five regime anchors × **H∈{1,3,6,12}** via
`seq_pricing.simulate_paths` (ADR-002), restate as CPR / WAL / **price** errors through the
**unchanged** M24 cashflow engine, and add a sequence column to Ch.5's horizon exhibit.

**Why this matters / the narrative.** M27b already priced the GRU at **H=1** — its *weakest*
horizon (the value-of-memory edge is multi-month). H>1 was deferred there as "future-work
Monte-Carlo." This is that piece: it carries the GRU and transformer to the longer horizons where
the memory edge concentrated (esp. the **COVID 2020** anchor), so the loan-level Ch.4 result gets
its economic counterpart. Both arms priced ⇒ the pricing exhibit matches the four-arm Ch.4 table.

**Sample = 10M per window**, matching the Ch.4 headline (the W1/W2 runs saved only predictions, and
M27b's saved weights are the older 1.5M — so the anchor models must be **retrained at 10M** to be
the model we report). This is the cost the 10M choice buys; it is the expensive part.

---

## What already exists — do NOT rebuild

| Piece | Where | State |
|---|---|---|
| MC simulator `simulate_paths` / `price_paths` / model-agnostic `SeqPredictor(history=…)` | `src/floan/model/seq_pricing.py` | built + 9 unit tests pass (W3). Works for any fitted seq model. |
| Anchor pop / base construction, `train_or_load` (GRU weight-saving), H=1 identity check, H=1 pricing | `src/floan/model/seq_predict.py` | done (M27b), H=1 only |
| Economic aggregation `smm_paths → economics.per_pool_econ(horizons=(1,3,6,12))` | `src/floan/model/{smm_paths,economics}.py` | the M24 reference, unchanged |
| Anchor set | `economics.ANCHORS = [2015, 2019, 2020, 2023, 2025]` → Dec2014/18/19/22/24; COVID = 2020 | fixed |

## The gap W3b fills

1. **Transformer weight-saving** — `gpu_train_transformer.py` saved only predictions; there is no
   `train_or_load` for it. (The GRU's exists in `seq_predict.py`.)
2. **An H>1 MC pricing driver** — mirrors `seq_predict.price_anchor_h1` but rolls via
   `seq_pricing.simulate_paths` (full `[N,12]` SMM, not just column 0), assembles each loan's
   **trailing-T observed history**, and is **arch-agnostic** (GRU or transformer).
3. **Retrain the anchor GRU + transformer at 10M** and run the driver (GPU pod).
4. **Fold the sequence columns into Ch.5** (done on the Mac after results land).

---

## Part A — Code (Mac Claude Code, hermetic; commit *before* the pod run)

This is non-GPU, synthetic-testable code. Paste the prompt below into Claude Code on the Mac, review
the diff, run `python -m pytest`, commit. **It must land first** — the pod pulls and runs it.

> **Prompt for Claude Code (Mac):**
>
> Read `src/floan/model/seq_pricing.py`, `src/floan/model/seq_predict.py`,
> `src/floan/model/smm_paths.py`, `src/floan/model/economics.py`, `tests/test_seq_pricing.py`, and
> `specs/model/ADR-002-sequence-pricing-seam.md` first. Then implement **W3b: Monte-Carlo H>1
> pricing of the sequence models**, surgically and additively (do not touch the cashflow engine,
> `compose`/`absorb_rows`, or `economics`/`smm_paths` math).
>
> **(1) Transformer weight persistence.** Generalise `seq_predict.train_or_load` so it can fit/save
> **either** the GRU (`seq_model.build`) **or** the transformer (`seq_transformer.build`), without
> breaking the existing GRU call. Add a small `arch` argument (default `"gru"`, backward-compatible)
> that selects the builder, the frozen config, and the weights filename
> (`k{k}_s{seed}.pt` for GRU, `k{k}_xf_s{seed}.pt` for the transformer). Reuse the transformer's
> frozen training config from `scripts/m27a/gpu_train_transformer.py` verbatim (d_model 64, 4 heads,
> 2 layers, and the same Adam/wd/scheduler/patience/batch as the GRU). Load if the `.pt` exists,
> else retrain at the cached 10M split and `torch.save` the best state. **Write these 10M weights to
> a distinct directory:** add a `weights_root` argument (default the existing
> `config.OUTPUTS/"m27b_gpu_runs"`, so existing M27b GRU calls are unchanged) and have the W3b path
> pass `config.OUTPUTS/"m27b_mc"` — the 10M weights are a different sample and must never overwrite
> M27b's 1.5M H=1 weights in `m27b_gpu_runs/`.
>
> **(2) The MC pricing driver.** Add `price_anchor_mc(model, scaler, vocab, device, k, pop, *,
> horizon=12, n_paths, calibrator=None, seed=0)` (new module `seq_price_mc.py`, or alongside
> `price_anchor_h1`). It must:
> - build `base = anchor_base(k, pop)` and assemble the **history**: each pop loan's *observed*
>   trailing rows for the months **before** `t0 = Dec(k−1)` (long format, one row per loan-month,
>   `prepare_raw`'d, with `Loan Identifier` + the model feature columns), enough rows to fill the
>   trailing-T window so the seq_pricing H=1 window equals M27b's trailing-T observed window;
> - build `seq_pricing.SeqPredictor(model, scaler, vocab, device, history=history)` and call
>   `seq_pricing.simulate_paths(pred, base, t0, horizon=horizon, n_paths=n_paths,
>   calibrator=calibrator, seed=seed)`;
> - apply the **same calibrator** the M24/M27b path uses for that anchor (per `calibration.py` /
>   ADR-001 §M23) so the MC conditions identically to the composition path;
> - assemble the `smm_paths`-shaped `tbl` from `sim["smm"]` for the **full 12 columns** (model name
>   `"gru"` or `"xf"`), then run the **unchanged** `smm_paths.pool_smm_paths → paths_to_frame →
>   economics.per_pool_econ(k, smm_frame, horizons=(1,3,6,12))`;
> - return the per-(scheme,pool) econ frame + meta + the guard results.
>
> **(3) Guards (hard gates — abort the anchor on failure).**
> - **H=1 identity:** `sim["h1_dist"]` (MC mean at H=1) must reproduce the GRU/transformer's direct
>   one-step `evaluate` prediction (`seq_predict.gru_eval_probs`, generalised to the arch) on the
>   same loans to Monte-Carlo tolerance `O(1/√N)`. This is the integration check that the history
>   assembly is byte-right.
> - **Paths-convergence:** sweep `n_paths ∈ {500,1000,2000,3200}` at the **COVID anchor (k2020)**;
>   pick the smallest `N` where the pool price moves `< 0.05` per 100 face between successive sweeps.
>   Record `N`; reuse it for the other anchors.
> - **Aggregation identity:** Σ per-loan price == pool price to ~1e-6 (the ADR-001 §M22 check; extend
>   the existing one).
>
> **(4) Hermetic test** in `tests/` (synthetic untrained model, CPU, tiny pop+history): assert the
> H=1 MC mean ≈ direct one-step, the returned econ frame carries all four horizons, and the
> aggregation identity holds. Must pass under `python -m pytest` with no SSD/GPU/network.
>
> Keep it minimal and surgical; match the style of `seq_predict.py`. State any assumption you have
> to make about the history-window length explicitly in a comment.

**Acceptance for Part A:** `python -m pytest` green; the new test exercises the H=1 identity + the
4-horizon econ frame on synthetic data. Commit (no `Co-Authored-By` trailer).

---

## Part B — Pod run (GPU)

1. **GPU.** A **high-RAM** card (the RTX PRO 6000 WK / the 1095 GB host you used for W1) — the 10M
   retrain is the same big-RAM job as a W1 per-window train. Network Volume `/workspace`.
2. **Env hardening.** Source the same cache redirects as `run_w1_all.sh` (TMPDIR, XDG_CACHE_HOME,
   TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR, CUDA_CACHE_PATH, MPLCONFIGDIR → all under `/workspace`)
   so the overlay can't fill. `LOGDIR=/workspace/logs`.
3. **Git.** `git pull` main; **create and check out a branch** `w3b-seq-pricing` and push results to
   **that branch, not main** (the unattended-run rule — the harness blocks direct-to-main). Write the
   PAT via `read -rsp` straight into `~/.git-credentials`; **never paste it in chat**.
4. **Caches.** Ensure each anchor window's `seq_cache/full/k{k}` is the **10M** build
   (`SEQ_TRAIN_N=10000000`; `build_window.py` rebuilds on a train_n mismatch).
5. **Run, COVID-first**, anchors in priority order `2020 → 2023 → 2025 → 2019 → 2015`, **seed 0**,
   both arms. Per anchor: `train_or_load` (GRU, then transformer → 10M weights) → `price_anchor_mc`
   → write the anchor JSON → **commit + push that anchor before starting the next** (a crash then
   never loses a completed anchor). The H=1 identity + convergence guards gate each anchor.

**Output:** `src/floan/model/m27b_mc/k{k}_{gru,xf}_h.json` per anchor (econ at H∈{1,3,6,12}, the
chosen `N`, guard receipts).

---

## Part C — Write-up (Mac, after results land)

Fold the GRU + transformer H∈{1,3,6,12} price / CPR / WAL columns into Ch.5's horizon exhibit
(`tab:horizon`, or a dedicated sequence-pricing table) at the priced anchors, **leading with the
COVID 2020 anchor** — the value-of-memory edge is multi-month, so it should appear economically at
H≥3 where it was invisible at H=1 (M27b). Then resolve the Ch.2 `subsec:meth-rollforward` "priced at
H>1 via MC" note and the abstract / conclusions PLANNED boxes. (I'll do this when you say results
landed.)

---

## Timing & risk ladder

- **Cost driver:** up to **10 retrains** (5 anchors × 2 archs) at 10M — each ≈ a W1 per-window 10M
  train (multi-hour, big-RAM) — plus the MC pricing (minutes–hours/anchor on GPU). Plan a long /
  overnight pod session. Sequencing COVID-first means the headline anchor lands even if you stop early.
- **Ship gate (spec minimum):** **≥3 anchors** — COVID 2020 + rate-shock 2023 + one calm (2015 or
  2019) — for **both** arms. That is a complete, defensible exhibit; the other two anchors are
  upside if time allows.
- **Fallback (if it slips):** the GRU stays the **loan-level (H=1)** headline (M27b, already in Ch.5)
  and Ch.5's H>1 sequence pricing reverts to the existing future-work paragraph (keep it). The thesis
  is complete either way — this is the controlled, droppable extension, exactly as the final-week risk
  ladder and ADR-002 "To revisit" intend. Don't let a stuck retrain hold the submission hostage.
