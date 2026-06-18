# M13 — Roll-forward harness (`03_POOL_LEVEL §3`, Phase-3 engine)

`pool.py`: compose the frozen loan-level model into 12-month horizon probabilities by **matrix
multiplication** (no Monte Carlo — terminal states absorb and we carry the full 7-vector, so the
matrix product is the exact horizon distribution under the model). Two outcomes per loan:
`P(prepaid ≤ 12m)` (normal chain) and `P(60+ dpd ≤ 12m)` (first-passage chain). M14 consumes
`roll_forward` to build pools and predicted counts.

## 1. Design
- **Anchor** for window `k`: `t0 = Dec(k−1)`; the population is the eval-pool test-slice rows at
  `period_ym == t0` (origin already transient by construction of the export). Loaded once,
  rolled in bounded-memory loan chunks (`--chunk`, default 200k).
- **Covariate evolution (`§3`):** only deterministic fields evolve — `Loan Age += h−1`,
  `Remaining Months to Maturity −= h−1`, and `period_ym` rolls forward to drive `month_of_year`
  seasonality. The **entire macro block and every static field (UPB, rates, FICO, LTV, vintage)
  are frozen at t0** — predictions are conditional on the t0 macro environment (no macro
  forecasting; the assumption-light choice). Remaining-term decrement is literal per spec (no
  clip at 0; a loan maturing inside the 12m horizon simply sees a negative standardized
  remaining-term — immaterial at this horizon since remaining ≫ 12 for nearly all loans).
- **Per step:** 4 forward passes (one per transient origin) by overriding **only** the `state`
  vocab index, assembled into per-loan 7×7 matrices (terminal rows = identity). Ensemble = mean
  of the 8 members' transition matrices (= mean of member softmaxes, the paper Fig-7 rule;
  identical to `evaluate`'s mean-of-probs at h=1). 48 forwards/chunk for a single net (12×4),
  416 for an ensemble window (12×4×(2+8)).
- **First-passage variant (`§3`):** a parallel chain with `dpd_60` made absorbing
  (`absorb_rows`). Because delinquency is monotone (a loan must pass through `dpd_60` to reach
  `dpd_90plus`/`foreclosure`/`REO`), all 60+ mass concentrates at `dpd_60`; `P(60+ ≤ 12m) =
  p_fp[dpd_60]`. Verified structurally in the unit test (deeper states unreachable once `dpd_60`
  absorbs).
- **Layering:** the numeric engine (`compose` / `absorb_rows` / `assemble_matrix` /
  `onehot_origin`) is pure NumPy and hermetically unit-tested; model scoring reuses the frozen
  run folders and the exact `evaluate.py` softmax path (float64-softmax → float32 cast).

## 2. Accept criteria — evidence

**(1) Toy-chain unit test matches closed form** — `test_pool.py`, 6 tests, all pass (hermetic,
no SSD/GPU):
- `compose` over H homogeneous steps == `p0 · P^H` (matrix-power closed form), to 1e-12; mass
  conserved.
- hand-computed 2-state/2-step example.
- `assemble_matrix`: terminal rows identity, every row stochastic, mass conserved through 12
  steps.
- first-passage (`absorb_rows` + compose) == independent **brute-force path enumeration** of
  "ever reaches target within H steps", to 1e-12.
- monotone-chain concentration: deeper-than-60 states unreachable once `dpd_60` absorbs.

**(2) h=1 probabilities equal `evaluate.py` outputs exactly** — `pool.py --verify`. At h=1 the
evolution is the identity (age+0, remaining−0, month=t0), so the composed `one-hot(origin)·P_1`
is the row of `P_1` for the loan's true origin = the model's per-row prediction. Two checks:
- **Harness exactness** (identical batching, single chunk) vs a direct same-path scoring of the
  anchor rows: **bit-for-bit (Δ=0) for every model on both k=2018 and k=2020** — proves the
  one-hot init, per-origin `state` override, 7×7 assembly and composition select the right row.
- **Cross-check vs the committed `evaluate.score_window`** on the same anchor rows: empirical and
  logit **bit-identical (max|Δ|=0.00)**; deep nets within `XCHECK_TOL=1e-5` (k=2018 nn 0.00;
  k=2020 nn 2.38e-7, ensemble 7.45e-8) — the residual is cuBLAS GEMM batch-shape nondeterminism
  (the anchor rows sit in different batch positions than in the 7M-row full-slice run), **not** a
  harness discrepancy. The logit matching exactly corroborates that the feature encoding is
  faithful.

**(3) Runs over all t0-alive loans in bounded memory** — full roll-forward, default 200k chunks:
- k=2018: **578,431** alive loans rolled 12m in **27s**, peak RSS **9.4 GB**.
- k=2020: **617,744** alive loans rolled 12m in **55s** (ensemble window), peak RSS **10.4 GB**.
- All outcome probabilities in [0,1]. Sanity means (k=2020): prepaid≤12m — empirical 0.152 /
  logit 0.134 / nn 0.130 / ensemble 0.129; 60+≤12m ≈ 0.017–0.025.

## 3. Artefacts
- Code: `pool.py` (engine + `roll_forward` + `verify_h1` + driver), `test_pool.py` (6 unit tests).
- Run summaries (SSD `models/nn/full/`): `pool_rollforward_k2018.json`,
  `pool_rollforward_k2020.json` (Accept-2/3 evidence, outcome means, timing, peak RSS).
