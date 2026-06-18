# Memo 02a — Benchmarks: the linear baseline

*Phase 2 §4–5 of the modelling plan (`specs/model/02_LOAN_LEVEL.md`); the "linear
baseline" section of the results chapter. Covers the two covariate-free frequency
benchmarks and the multinomial logit (incl. the hand-augmented variant), on the tuning
window k = 2015. Tables in `outputs/tables/loan_level/` (`table_a.*`, `table_b.*`,
`auc.*`); the model runs under `models/{benchmarks,logit,nn}/`. All NLLs are out-of-sample
mean negative log-likelihood on the frozen, unthinned test slices.*

---

## 1. The comparison protocol (binding for 02a and 02b)

Every model predicts the same target — `P(state_next = j | state_t, x_t)`, a single
multinomial over the 7 states conditioned on origin state as a feature (`§1`, the paper's
design) — and is scored on **identical** per-window frozen test rows. Window k trains on
labels ≤ Dec(k−2), early-stops on year k−1, and is tested on label-year k; the test slice
is cut from the never-thinned **eval pool** (a fixed loan-disjoint shard block, w ≡ 1), so
it is byte-identical across models. The headline metric is the out-of-sample NLL, reported
per test year and pooled over all 11 years with each test row predicted exactly once by a
model that never saw it (the M11 Accept checks confirm row-count + key-hash identity and
pooled uniqueness over **84,512,377** test rows).

The tuning window **k = 2015** is where every architecture/hyperparameter choice is made;
the configs are then frozen and looped over all 11 windows (memo 02b). This memo reports
the benchmarks on that window (dev-scale grid for Table A, full-scale for the frozen
fits), and their pooled discrimination.

## 2. Model A — the empirical transition matrix (the floor)

The covariate-free floor is the row-normalised 4×7 matrix of one-month transition
frequencies (`§4`), Laplace-smoothed (α = 0.5) so no test transition has probability 0.
It uses **no covariates** and conditions only on origin state — so it cannot represent any
of the EDA nonlinearities (the seasoning hump F3.1, the refinancing S-curve F3.2, the
FICO×LTV interaction F3.4). It is the line every covariate model must beat to justify its
features.

On k = 2015 the empirical matrix scores **0.112870** test NLL (full export; the dev-grid
Table A value is 0.112661 — the estimator is recomputed here as a *weighted* `GROUP BY` on
the thinned train pool, the `1/p_keep` weight making it an unbiased estimate of the
unthinned-panel matrix). A **bucketed** variant conditioning on state × FICO-tercile ×
incentive-tercile (still nonparametric, no fitting) reaches **0.109247** — cheap
conditioning alone closes part of the gap, a useful yardstick for how much of the logit's
edge is just coarse stratification.

The matrix also fixes the structural skeleton the later QA leans on: from `current` the
realised one-month mass is ≈ 97.6 % stay, 1.55 % → prepaid, 0.81 % → dpd_30, and ≈ 0 on
the structurally impossible cells (the monotone-delinquency rule, `§1`).

## 3. Model B — the multinomial logit

The logit is implemented as a 0-hidden-layer network through the *same* loss/eval/data
path as the nets (`§5`), so any later NN gain is attributable to architecture alone; its
coefficients/NLL were cross-checked against `sklearn` on a 1 M-row subsample at M7, and the
full-scale loop's embedding-sum logit reproduces that sklearn-validated fit to < 1e-4
across the L2 grid (M10b receipt). L2 strength is selected on each window's val NLL.

On k = 2015 the logit scores **0.108246** test NLL (full) — it beats the empirical floor
(0.112870) and the bucketed matrix (0.109247), confirming the covariates carry signal a
frequency table cannot. But the margin is thin: the logit recovers only **≈ 2.9 %** of NLL
over the floor (pooled, §1 below), against the ≈ 9 % the deep net will recover (02b). The
linear-additive form leaves most of the achievable improvement on the table — exactly the
EDA prediction (`01_EDA §7`: a linear model should underfit, most severely on prepayment).

**Discrimination (AUC, one-vs-rest, pooled over all 11 windows, `auc.*`).** The logit's
ranking power is concentrated and uneven:

| origin → destination | logit AUC | reading |
|---|---|---|
| current → dpd_30 | 0.764 | early-delinquency rank is decent (FICO/LTV are near-linear in the log-odds here) |
| current → prepaid | 0.653 | weak — the refinancing S-curve (F3.2) is the nonlinearity the logit cannot bend to |
| dpd_90plus → foreclosure | **0.482** | **below chance** — the linear model cannot rank the rarest, most nonlinear credit event |

The prepayment and deep-delinquency channels are precisely where the linear form fails;
this is the gap memo 02b shows the neural net closing.

## 4. The augmented logit (the hand-crafted-nonlinearity concession)

The paper's concession to nonlinearity — add a squared loan-age term and a binned/spline
incentive — is the fair test of "how much of the NN's edge is recoverable by manual
feature engineering" (`§5`). On k = 2015 the augmented logit scores **0.108843** (dev Table
A), an improvement of only **0.0006 NLL** over the plain logit (0.109430 dev) and still
**well above** the deep net (0.103, dev). Hand-crafting the two best-known nonlinearities
recovers a sliver; the bulk of the structure — the interactions (FICO×LTV F3.4,
incentive×FICO F3.5) and the high-cardinality geography the embeddings compress — is not
reachable by a handful of manual terms. This is the quantitative case for letting the
network learn the functional form.

## 5. Pooled benchmark NLL (all 11 windows, each row once)

Pooled over the full 84.5 M test rows (`table_b.*`):

| model | pooled NLL | vs floor |
|---|---|---|
| Empirical matrix | 0.111477 | — |
| Logit | 0.108249 | −2.90 % |
| *(Best NN — memo 02b)* | *0.101468* | *−8.98 %* |

The logit captures roughly **one third** of the NLL reduction the deep net achieves over
the floor. The remaining two thirds is nonlinearity and interaction — the subject of memo
02b.

## 6. One finding worth flagging now (carried into 02b)

Broken out by origin state (`table_b.*` "by origin"), the logit's covariates help **only**
in the dominant `current` origin (0.0911 vs the floor's 0.0947). In every *delinquent*
origin the linear model is **at or below the covariate-free floor** — dpd_30 1.147 vs
1.117, dpd_60 1.388 vs 1.361, dpd_90plus 0.576 vs 0.575. In the sparse, highly nonlinear
multi-way transitions out of delinquency, forcing a linear log-odds form does *worse* than
just using the marginal frequencies. The deep net (02b) beats the floor in **every** origin
— the clearest single statement that the value of the covariates here is inseparable from
modelling them nonlinearly.

*Deviations from Sirignano et al. recorded per `04_TASKS §4`: the GSE public book is a
cleaner credit box than the paper's private-label universe, so absolute NLLs and the
benchmark-to-NN gaps are smaller in magnitude; the qualitative ordering (floor < logit <
augmented logit ≪ NN) reproduces.*
