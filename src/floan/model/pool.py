"""M13 — Roll-forward harness (``03_POOL_LEVEL §3``): compose the frozen loan-level model
into 12-month horizon probabilities by **matrix multiplication**, the Phase-3 engine.

For a loan *i* alive at the anchor ``t0 = Dec(k−1)`` (origin state non-terminal), the
12-month outcome distribution is the product of monthly transition matrices::

    p ← one-hot(state_{t0})                         # 1×7 state distribution
    for h = 1..12:
        features x_{t0+h-1}(i):  age += h−1, remaining −= h−1, month_of_year follows
                                 the calendar;  macro block + all static fields frozen at t0
        P_h ← model's 7×7 matrix for loan i at month h   (terminal rows = identity)
        p ← p · P_h

Because the three terminal states (foreclosure / REO / prepaid) are absorbing and we carry
the full 7-vector, the matrix product is the **exact** horizon distribution under the model —
no Monte-Carlo error (``§3``). Two outcomes per loan:

  * ``P(prepaid within 12m) = p[prepaid]``                       (normal chain)
  * ``P(60+ dpd within 12m) = p_fp[dpd_60]``                     (first-passage chain)

The **first-passage** variant makes ``dpd_60`` temporarily absorbing so cures cannot leak the
event; because delinquency is monotone (a loan must pass through ``dpd_60`` to reach
``dpd_90plus``/``foreclosure``/``REO``), all 60+ mass concentrates at ``dpd_60`` (``§3``).

Layering: the numeric engine (``compose`` / ``absorb_rows`` / ``assemble_matrix``) is pure
NumPy and hermetically unit-tested against the closed form (``test_pool.py``). The model
scoring reuses the frozen run folders and the exact ``evaluate.py`` softmax path, so the h=1
composition reproduces ``evaluate.py``'s per-row probabilities bit-for-bit (the M13 Accept).
M14 consumes :func:`roll_forward` to build pools and predicted counts.

M21 (ECONOMIC_ENGINE §3 / ADR-001) adds the model-agnostic seam: a :class:`Predictor` protocol
(``origin_scores(frame_h)``) with :class:`TorchPredictor` (a verbatim extraction of the former
``_origin_scores`` body), :class:`EmpiricalPredictor`, :class:`EnsemblePredictor`; the single-
predictor engine :func:`roll_forward_predictor` (horizon a parameter, optional ``calibrator`` /
``absorb``, snapshots H∈{1,3,6,12} of one roll to 12); and a staged + legacy-guarded
:func:`verify_h1`. The legacy multi-model :func:`roll_forward` is unchanged for M14/M15.

M22 (ECONOMIC_ENGINE §3.4) adds loan-level pricing: ``roll_forward_predictor(capture_smm=True)``
records the per-month prepay increment the loop already computes (cumulative-prepaid + alive mass,
no second roll); :func:`per_loan_smm` turns that into a per-loan conditional SMM path; and
:func:`price_loans` prices each loan as a one-loan "pool" through the unchanged :func:`cashflow_engine`
and aggregates to the pool (the §4.3 linear-in-UPB aggregation identity). No new engine math.

Run (GPU box; the full eval pool lives there):
    .venv/bin/python -m floan.model.pool --k 2020 --device cuda            # full window
    .venv/bin/python -m floan.model.pool --k 2020 --device cuda --verify   # + h=1 == evaluate
    .venv/bin/python -m floan.model.pool --k 2020 --max-loans 50000        # quick smoke

Local M21 proof (CPU, SSD mounted; standing rule 3):
    .venv/bin/python -m floan.model.pool --k 2020 --device cpu --max-loans 20000 --verify
    .venv/bin/python -m floan.model.pool --k 2020 --device cpu --smoke --horizons
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
import torch

from floan.model import backtest as B
from floan.model import config
from floan.model import data as D
from floan.model import evaluate as E
from floan.model import features as F
from floan.model import macro_features as mf
from floan.model import net as N  # noqa: F401 — E._load_nn rebuilds via net; kept explicit for clarity
from floan.model import torch_common as tc
from floan.model import train as T

VARIANT = "full"
HORIZON = 12
EVAL_BATCH = E.EVAL_BATCH                 # 32768 — same batching as evaluate (numerically inert)
DEFAULT_CHUNK = 200_000                   # loans per roll-forward chunk (bounded memory)

STATES = F.STATES                         # canonical 7-state order
SI = F.STATE_INDEX                        # state name -> class index 0..6
N_CLASSES = F.N_CLASSES
ORIGIN_STATES = list(config.ORIGIN_STATES)            # the 4 transient origins
ORIGIN_ROWS = [SI[s] for s in ORIGIN_STATES]          # their row indices in the 7×7
TERMINAL_ROWS = [SI[s] for s in STATES if s not in ORIGIN_STATES]  # fc / REO / prepaid
TRANSIENT_COLS = np.array(ORIGIN_ROWS)                # transient (= alive) state columns
PREPAID = SI["prepaid"]
DPD60 = SI["dpd_60"]
STATE_COL = F.CATEGORICAL.index("state")              # the `state` column position in the vocab


# ===========================================================================
# Layer 1 — pure numeric composition engine (hermetic; unit-tested vs closed form)
# ===========================================================================
def compose(p0: np.ndarray, step_mats) -> np.ndarray:
    """Compose a per-loan Markov chain: ``p0`` ``[N, S]`` initial distributions, ``step_mats``
    an iterable of per-step transition tensors ``[N, S, S]`` (row = origin, col = destination).
    Returns the horizon distribution ``[N, S]`` = ``p0 · M_1 · M_2 · …`` (float64)."""
    p = np.ascontiguousarray(p0, dtype=np.float64)
    for M in step_mats:
        p = np.einsum("ni,nij->nj", p, M)
    return p


def absorb_rows(M: np.ndarray, idx: int) -> np.ndarray:
    """Copy of ``M`` ``[..., S, S]`` with state ``idx`` made absorbing (its row → ``e_idx``):
    the first-passage trick — once mass enters ``idx`` it stays, so the horizon mass at ``idx``
    is the probability of *ever* reaching it."""
    out = M.copy()
    out[..., idx, :] = 0.0
    out[..., idx, idx] = 1.0
    return out


def assemble_matrix(origin_scores: list[np.ndarray]) -> np.ndarray:
    """Build per-loan 7×7 transition tensors ``[N, 7, 7]`` from the four transient-origin score
    rows (each ``[N, 7]`` softmax over destinations, in ``ORIGIN_STATES`` order). Terminal rows
    are the identity (foreclosure / REO / prepaid are absorbing, ``§3``)."""
    n = origin_scores[0].shape[0]
    M = np.zeros((n, N_CLASSES, N_CLASSES), dtype=np.float64)
    for oi, row in zip(ORIGIN_ROWS, origin_scores):
        M[:, oi, :] = row
    for ti in TERMINAL_ROWS:
        M[:, ti, ti] = 1.0
    return M


def onehot_origin(origin_idx: np.ndarray) -> np.ndarray:
    """``[N, 7]`` one-hot initial distribution from per-loan origin class indices."""
    p0 = np.zeros((origin_idx.shape[0], N_CLASSES), dtype=np.float64)
    p0[np.arange(origin_idx.shape[0]), origin_idx] = 1.0
    return p0


# M20 (ECONOMIC_ENGINE §3.5): the two mechanically-impossible one-step cells out of `current`.
IMPOSSIBLE_FROM_CURRENT = (SI["foreclosure"], SI["REO"])


def _zero_impossible(M: np.ndarray) -> np.ndarray:
    """Copy of the per-loan transition tensors ``[N, 7, 7]`` with the two mechanically-impossible
    one-step cells out of ``current`` (``current→foreclosure``, ``current→REO``) zeroed and the
    ``current`` row renormalised back to a distribution (M20 / §3.5). A current loan cannot
    foreclose or hit REO in a single month — it must pass through the dpd buckets — so these cells
    carry only phantom model mass (~1e-9, likelihood-neutral) but high LGD, which pricing
    severity-amplifies. Applied in the roll-forward predictor **before** any levels-consuming step;
    the loan-level eval path (``_score_direct`` / ``evaluate``) is left untouched."""
    out = M.copy()
    cur = SI["current"]
    out[:, cur, list(IMPOSSIBLE_FROM_CURRENT)] = 0.0
    out[:, cur, :] /= out[:, cur, :].sum(axis=1, keepdims=True)
    return out


# ===========================================================================
# Layer 3 — level-pay pass-through cashflow engine (03_POOL_LEVEL §5.2; hermetic)
# ===========================================================================
SERVICING_SPREAD = 0.0025      # 25 bp servicing strip — the fixed premium that makes price
                               # prepayment-sensitive; the SAME curve for every model, so the
                               # price spread isolates the prepayment-model component (§5.2).


def cashflow_engine(wac: float, wam: float, upb: float, smm,
                    *, servicing_spread: float = SERVICING_SPREAD) -> dict:
    """Deterministic level-pay pass-through (``§5.2``). Given a pool ``(WAC, WAM, UPB)`` and a
    monthly ``SMM`` vector (``h=1..len``, **constant-extrapolated** past its end), amortise the
    pool month by month — recomputing the level payment on the *surviving* balance and remaining
    term each month — and return the principal/interest cashflow with its **price** (per 100
    face) and **WAL** (years).

    Each month: scheduled principal ``= pmt − interest`` (pmt fully amortises the current balance
    over the remaining term); the ``SMM`` then prepays that fraction of the post-amortisation
    balance. Cashflows carry the gross note coupon ``wac`` and are discounted at
    ``y = wac − servicing_spread``: a positive spread makes a premium pass-through whose price
    *falls* as the pool prepays faster (prepaid principal returns at par, retiring above-coupon
    cashflow). With the curve fixed (``wac`` is a pool property, identical across models) all
    price dispersion isolates the SMM path. Closed forms that pin the engine (``test_pool.py``):

      * ``servicing_spread = 0`` ⇒ ``y = c`` ⇒ price ≡ 100 for **any** SMM (par identity, by
        telescoping ``Σ cf_h/(1+c)^h = B_0 − B_n/(1+c)^n`` with ``B_n = 0``);
      * ``smm ≡ 0`` ⇒ the level-pay **annuity** (constant payment; price = ``PMT·a(y,n)``);
      * ``smm ≡ λ`` ⇒ the **survival schedule** ``B_h = UPB·φ_h·(1−λ)^h`` with ``φ_h`` the
        no-prepay scheduled-balance fraction ``((1+c)^n−(1+c)^h)/((1+c)^n−1)``."""
    n = int(round(wam))
    c = wac / 12.0
    y = (wac - servicing_spread) / 12.0
    smm = np.asarray(smm, dtype=np.float64).reshape(-1)
    bal = np.empty(n); interest = np.empty(n); sched = np.empty(n); prepay = np.empty(n)
    B = float(upb)
    for h in range(n):
        s = smm[h] if h < smm.shape[0] else smm[-1]          # constant extrapolation past h=len
        rem = n - h                                          # remaining months incl. this one
        pmt = B / rem if c == 0.0 else B * c / (1.0 - (1.0 + c) ** (-rem))
        i_h = B * c
        sp = min(pmt - i_h, B)                               # scheduled principal (guard last mo.)
        bal_after_sched = B - sp
        pp = s * bal_after_sched                             # SMM acts on post-amortisation balance
        bal[h] = B; interest[h] = i_h; sched[h] = sp; prepay[h] = pp
        B = bal_after_sched - pp
    total_prin = sched + prepay
    cashflow = interest + total_prin
    months = np.arange(1, n + 1, dtype=np.float64)
    disc = (1.0 + y) ** (-months)
    price = 100.0 * float((cashflow * disc).sum()) / upb
    wal = float((months * total_prin).sum() / (12.0 * total_prin.sum()))
    return {"n": n, "balance": bal, "interest": interest, "sched_prin": sched, "prepay": prepay,
            "total_prin": total_prin, "cashflow": cashflow, "price": price, "wal": wal,
            "total_principal": float(total_prin.sum()), "wac": wac, "wam": n, "upb": float(upb),
            "servicing_spread": servicing_spread, "discount_rate": wac - servicing_spread}


# ===========================================================================
# Layer 3b — loan-level pricing (M22, ECONOMIC_ENGINE §3.4): price each loan as a one-loan
# "pool" through the SAME cashflow_engine, then aggregate. No new engine math — this only
# marshals per-loan inputs and a per-loan SMM path into cashflow_engine and sums the result.
# ===========================================================================
def per_loan_smm(cum_prepaid: np.ndarray, alive_before: np.ndarray) -> np.ndarray:
    """Per-loan **conditional** monthly SMM path ``[N, H]`` from a roll-forward SMM capture
    (:func:`roll_forward_predictor` with ``capture_smm``). This is the loan-level analogue of the
    pool aggregation in ``smm_paths.py``: the unconditional prepay increment ``Δ_h(i) =
    cum_prepaid[i,h]−cum_prepaid[i,h−1]`` (with ``Δ_1 = cum_prepaid[:,0]`` — a transient-origin loan
    carries zero prepaid mass at t0) divided by the loan's surviving (transient) mass at the start of
    the month, ``SMM_i(h) = Δ_h(i) / alive_before(i, h−1)`` — the fraction of survivors that prepay in
    month ``h``, in ``[0, 1]`` (``Δ_h ≤ alive_before`` since prepaid mass cannot exceed the alive mass
    it leaves). Survivor-conditional matches what ``cashflow_engine`` expects (SMM acts on the
    surviving balance, ``§5.2``). Denominators of 0 (loan fully terminated) → SMM 0."""
    inc = np.empty_like(cum_prepaid, dtype=np.float64)
    inc[:, 0] = cum_prepaid[:, 0]
    inc[:, 1:] = cum_prepaid[:, 1:] - cum_prepaid[:, :-1]
    den = alive_before.astype(np.float64)
    return np.divide(inc, den, out=np.zeros_like(inc), where=den > 0)


def price_loans(wac: np.ndarray, wam: np.ndarray, upb: np.ndarray, smm: np.ndarray,
                *, servicing_spread: float = SERVICING_SPREAD) -> dict:
    """Price ``N`` loans one at a time through :func:`cashflow_engine` (each loan a one-loan
    "pool" with its **own** note rate / remaining term / UPB and its own ``[H]`` conditional monthly
    SMM path), then aggregate to the pool (``ECONOMIC_ENGINE §3.4``). No new engine math — the
    per-loan call is the unchanged ``§5.2`` engine; this wrapper only loops it and sums dollars.

    Inputs are row-aligned arrays: ``wac[N]`` (annual **decimal**), ``wam[N]`` (months),
    ``upb[N]`` ($), ``smm[N, H]`` (conditional monthly SMM, e.g. from :func:`per_loan_smm`). Returns
    per-loan ``price[N]`` (per 100 face), ``wal[N]`` (years), ``value[N] = price·upb/100`` ($) and
    the pool aggregate:

      * ``pool_value = Σ_i value_i`` — pool dollar value = Σ loan dollar values (the engine is
        **linear in UPB**, so this is the §4.3 aggregation identity by construction);
      * ``pool_price = 100·pool_value / Σ_i upb_i`` — the UPB-weighted mean per-100 price;
      * ``pool_wal = Σ_i wal_i·total_principal_i / Σ_i total_principal_i`` — principal-$-weighted
        (the exact pool WAL: ``Σ_i Σ_h h·prin_{i,h} / (12·Σ_i Σ_h prin_{i,h})``).

    For a pool whose loans share ``wac``/``wam`` (one discount curve, one schedule length) this pool
    aggregate equals the pool-level ``cashflow_engine`` run on the balance-weighted aggregate SMM to
    ~1e-6 (M22 Accept / §4.3); at the population scale the pool ``wac``/``wam`` are UPB-weighted means
    and the reconciliation is approximate — a modelling choice, stated, not a wiring bug."""
    wac = np.asarray(wac, dtype=np.float64).reshape(-1)
    wam = np.asarray(wam, dtype=np.float64).reshape(-1)
    upb = np.asarray(upb, dtype=np.float64).reshape(-1)
    smm = np.asarray(smm, dtype=np.float64)
    n = wac.shape[0]
    if not (wam.shape[0] == upb.shape[0] == smm.shape[0] == n):
        raise ValueError(f"row-count mismatch: wac={n} wam={wam.shape[0]} upb={upb.shape[0]} "
                         f"smm={smm.shape[0]}")
    price = np.empty(n); wal = np.empty(n); tprin = np.empty(n)
    for i in range(n):
        r = cashflow_engine(float(wac[i]), float(wam[i]), float(upb[i]), smm[i],
                            servicing_spread=servicing_spread)
        price[i] = r["price"]; wal[i] = r["wal"]; tprin[i] = r["total_principal"]
    value = price * upb / 100.0
    pool_upb = float(upb.sum())
    pool_value = float(value.sum())
    pool_price = 100.0 * pool_value / pool_upb if pool_upb > 0 else float("nan")
    tp_sum = float(tprin.sum())
    pool_wal = float((wal * tprin).sum() / tp_sum) if tp_sum > 0 else float("nan")
    return {"n_loans": n, "price": price, "wal": wal, "value": value, "upb": upb,
            "total_principal": tprin, "pool_value": pool_value, "pool_upb": pool_upb,
            "pool_price": pool_price, "pool_wal": pool_wal,
            "servicing_spread": servicing_spread}


def price_loans_vec(wac: np.ndarray, wam: np.ndarray, upb: np.ndarray, smm: np.ndarray,
                    *, servicing_spread: float = SERVICING_SPREAD) -> dict:
    """Vectorised :func:`price_loans` — the **same** §5.2 level-pay math, run over all ``N`` loans
    at once instead of one Python ``cashflow_engine`` call per loan (the M24 loan-level scale guard:
    ~0.7 M loans/anchor would be a Python-loop bottleneck). Identical inputs/outputs to
    :func:`price_loans`; agrees with it to ~1e-9 (``test_pricing_grid``), so the M22 closed-form /
    aggregation-identity guarantees carry over unchanged.

    Amortise month-by-month as a single masked array sweep: at each month ``h`` the **active** loans
    (``h < round(wam_i)``) recompute the level payment on their surviving balance and remaining term,
    accrue interest, take scheduled principal, then prepay ``SMM_i(h)`` of the post-amortisation
    balance (``SMM`` constant-extrapolated past its last column). Per-loan price/WAL are accumulated
    as running scalars (PV of cashflow, principal-$, time-weighted principal-$) so memory is ``O(N)``,
    not ``O(N·max_wam)``. Loans past their term contribute nothing (balance has reached 0)."""
    wac = np.asarray(wac, dtype=np.float64).reshape(-1)
    wam = np.asarray(wam, dtype=np.float64).reshape(-1)
    upb = np.asarray(upb, dtype=np.float64).reshape(-1)
    smm = np.asarray(smm, dtype=np.float64)
    n_loans = wac.shape[0]
    if not (wam.shape[0] == upb.shape[0] == smm.shape[0] == n_loans):
        raise ValueError(f"row-count mismatch: wac={n_loans} wam={wam.shape[0]} "
                         f"upb={upb.shape[0]} smm={smm.shape[0]}")
    H = smm.shape[1] if smm.ndim == 2 else 0
    n = np.rint(np.nan_to_num(wam, nan=0.0)).astype(np.int64)         # int(round(wam)) per loan
    n = np.clip(n, 0, None)
    c = wac / 12.0
    y = (wac - servicing_spread) / 12.0
    B = np.nan_to_num(upb, nan=0.0).astype(np.float64)                # surviving balance; null UPB → 0
    pv = np.zeros(n_loans); tp_sum = np.zeros(n_loans); wal_num = np.zeros(n_loans)
    max_n = int(n.max()) if n_loans else 0
    for h in range(max_n):
        active = h < n                                               # rem ≥ 1 exactly where active
        if not active.any():
            break
        rem_safe = np.where(active, n - h, 1)                        # avoid rem≤0 in the pmt formula
        # level payment on the surviving balance over the remaining term (c==0 ⇒ straight-line)
        pmt = np.where(c == 0.0, B / rem_safe, B * c / (1.0 - (1.0 + c) ** (-rem_safe)))
        i_h = B * c
        sp = np.minimum(pmt - i_h, B)                               # scheduled principal (guard last)
        sp = np.where(active, sp, 0.0)
        i_h = np.where(active, i_h, 0.0)
        bal_after = B - sp
        s = smm[:, h] if h < H else (smm[:, H - 1] if H else np.zeros(n_loans))
        pp = np.where(active, s * bal_after, 0.0)                   # SMM on post-amortisation balance
        total_prin = sp + pp
        disc = (1.0 + y) ** (-(h + 1))
        pv += (i_h + total_prin) * disc
        tp_sum += total_prin
        wal_num += (h + 1) * total_prin
        B = bal_after - pp
    pos = upb > 0
    price = np.where(pos, 100.0 * pv / np.where(pos, upb, 1.0), np.nan)
    value = np.where(pos, price * upb / 100.0, 0.0)                 # null/zero-UPB loan ⇒ 0 value
    wal = np.where(tp_sum > 0, wal_num / (12.0 * np.where(tp_sum > 0, tp_sum, 1.0)), np.nan)
    pool_upb = float(upb[pos].sum())
    pool_value = float(value.sum())
    pool_price = 100.0 * pool_value / pool_upb if pool_upb > 0 else float("nan")
    tps = float(tp_sum.sum())
    pool_wal = float((np.where(tp_sum > 0, wal, 0.0) * tp_sum).sum() / tps) if tps > 0 else float("nan")
    return {"n_loans": n_loans, "price": price, "wal": wal, "value": value, "upb": upb,
            "total_principal": tp_sum, "pool_value": pool_value, "pool_upb": pool_upb,
            "pool_price": pool_price, "pool_wal": pool_wal,
            "servicing_spread": servicing_spread}


# ===========================================================================
# Layer 2 — feature evolution (deterministic fields only; macro + statics frozen at t0)
# ===========================================================================
def add_months(ym: int, k: int) -> int:
    """``YYYYMM`` integer plus ``k`` calendar months (reuses the macro month arithmetic)."""
    return mf.sub_months(ym, -k)


def advance_frame(base: pl.DataFrame, t0: int, h: int) -> pl.DataFrame:
    """The loan-month frame at horizon step ``h`` (month ``t0+h−1``): age advances, remaining
    term decrements, ``period_ym`` rolls forward to drive ``month_of_year`` seasonality. Every
    other column — the entire macro block, UPB, rates, FICO, LTV, vintage — stays frozen at t0
    (``§3`` covariate-evolution assumption). ``state`` is overridden later, per origin pass."""
    dm = h - 1
    return base.with_columns(
        (pl.col("Loan Age") + dm).alias("Loan Age"),
        (pl.col("Remaining Months to Maturity") - dm).alias("Remaining Months to Maturity"),
        pl.lit(add_months(t0, dm), dtype=pl.Int64).alias("period_ym"),
    )


@torch.no_grad()
def _forward_probs(model, cont, cat, binb) -> np.ndarray:
    """Softmax class probabilities ``[N, 7]`` (float32) — the exact ``evaluate._resident_probs``
    numerics (float64 softmax then cast), so h=1 matches ``evaluate.py`` bit-for-bit. ``cont`` /
    ``binb`` float32 device tensors, ``cat`` an int device tensor (cast to long per batch)."""
    model.eval()
    n = int(cont.shape[0])
    out = np.empty((n, N_CLASSES), dtype=np.float32)
    for s in range(0, n, EVAL_BATCH):
        sl = slice(s, s + EVAL_BATCH)
        logits = model(cont[sl], cat[sl].long(), binb[sl])
        out[sl] = torch.softmax(logits.double(), dim=1).float().cpu().numpy()
    return out


# The `state` vocab index is window-specific; cache the per-call resolution in a closure-free
# module global set by roll_forward / the predictors (kept tiny and explicit rather than threading
# vocab through every scoring call).
_STATE_IDX: dict[str, int] = {}


def config_state_index(s: str) -> int:
    return _STATE_IDX[s]


# ===========================================================================
# Predictor seam (ADR-001 / ECONOMIC_ENGINE §3.1) — the ONE model-specific surface.
# `roll_forward_predictor` calls `predictor.origin_scores(frame_h)` and nothing else
# model-specific, so the net / GBT (M19) / sequence model (M27) all flow through the identical
# composition + cashflow math; only the injected Predictor differs (the controlled-comparison
# guarantee the dissertation rests on).
# ===========================================================================
class Predictor(Protocol):
    def origin_scores(self, frame_h: pl.DataFrame) -> list[np.ndarray]:
        """One ``(n_origin, 7)`` probability block per transient origin (``current, dpd_30,
        dpd_60, dpd_90plus`` order) at covariates already advanced to month ``h``. Rows sum to 1."""
        ...


class Calibrator(Protocol):
    def __call__(self, probs: np.ndarray, origin: int) -> np.ndarray: ...   # identity when disabled


def identity_calibrator(probs: np.ndarray, origin: int) -> np.ndarray:
    """The no-op calibrator (§3.2): ``calibrator=None`` and ``calibrator=identity_calibrator`` give
    bit-identical roll-forwards (the M23 disabled-path regression check). M23 replaces it with the
    per-window temperature/isotonic transform fit on the val slice."""
    return probs


class TorchPredictor:
    """Wrap the frozen torch path ``(model, scaler, vocab, device)`` behind the Predictor seam.
    :meth:`origin_scores` encodes the advanced frame exactly as the legacy ``_chunk_matrices`` did,
    then runs the **verbatim** ``_origin_scores`` body — so its output is byte-identical to the
    pre-seam scoring by construction (the M14/M15 banked results are unaffected)."""

    def __init__(self, model, scaler, vocab, device):
        self.model, self.scaler, self.vocab, self.device = model, scaler, vocab, device

    def _encode(self, frame_h: pl.DataFrame):
        """Frame → ``(cont, cat, binb)`` device tensors — moved verbatim from ``_chunk_matrices``."""
        enc = F.encode_frame(F.prepare_raw(frame_h), self.scaler, self.vocab, encode="index")
        cont = torch.as_tensor(enc["cont"], dtype=torch.float32, device=self.device)
        binb = torch.as_tensor(enc["bin"], dtype=torch.float32, device=self.device)
        cat = torch.as_tensor(np.asarray(enc["cat"]), dtype=torch.long, device=self.device)
        return cont, cat, binb

    def _origin_scores_tensors(self, cont, cat, binb) -> list[np.ndarray]:
        """=== verbatim ``_origin_scores`` body (``model`` → ``self.model``; the unused ``device``
        arg dropped). Score under each of the 4 transient origins by overriding *only* the ``state``
        vocab index (every other feature held), returning the four destination rows. ==="""
        scores = []
        for s in ORIGIN_STATES:
            cat_o = cat.clone()
            cat_o[:, STATE_COL] = config_state_index(s)
            scores.append(_forward_probs(self.model, cont, cat_o, binb))
        return scores

    def origin_scores(self, frame_h: pl.DataFrame) -> list[np.ndarray]:
        return self._origin_scores_tensors(*self._encode(frame_h))


class EmpiricalPredictor:
    """The feature-free empirical matrix behind the seam — and the M21 **stub second Predictor**
    (Accept: a second model flows through identical engine math, proving model-agnosticism).
    :meth:`origin_scores` broadcasts the 4 stored rows, exactly as the legacy ``_chunk_matrices``
    assembled ``empirical`` (float32 first, so the h=1 composition matches the stored probs)."""

    def __init__(self, emp: np.ndarray):
        self.emp = emp                                              # [4, 7]

    def origin_scores(self, frame_h: pl.DataFrame) -> list[np.ndarray]:
        c = frame_h.height
        return [np.broadcast_to(self.emp[oi].astype(np.float32), (c, N_CLASSES))
                for oi in range(len(ORIGIN_STATES))]


class EnsemblePredictor:
    """Mean-of-members behind the seam (paper Fig-7 rule) = the legacy ``_chunk_matrices`` ensemble
    averaging. Encodes **once** (members share one window's scaler/vocab) and averages each member's
    per-origin scores, so the mean is byte-identical to the legacy path."""

    def __init__(self, members: list[TorchPredictor]):
        self.members = members

    def origin_scores(self, frame_h: pl.DataFrame) -> list[np.ndarray]:
        cont, cat, binb = self.members[0]._encode(frame_h)
        acc = [np.zeros((cont.shape[0], N_CLASSES), dtype=np.float64) for _ in ORIGIN_STATES]
        for m in self.members:
            for j, row in enumerate(m._origin_scores_tensors(cont, cat, binb)):
                acc[j] += row
        return [a / len(self.members) for a in acc]


# ===========================================================================
# Roll-forward driver
# ===========================================================================
def _load_models(k: int, device, *, which: set | None = None):
    """Frozen window-``k`` models + the shared (scaler, vocab). Mirrors ``evaluate.score_window``
    model loading: empirical matrix, logit, best single NN, and (key windows) the 8-net
    ensemble. Returns ``(scaler, vocab, emp[4×7], torch_models{name:callable_or_list})``.

    ``which`` (a subset of ``{"logit", "nn", "ensemble"}``) restricts which torch models to load
    — used by M15a to (a) score ``{ensemble}`` when the logit/full checkpoints are unavailable and
    (b) later score ``{logit}`` alone to merge its SMM path. ``None`` loads the full set (the
    M13/M14 default). The empirical matrix is always built (panel-derived, no checkpoint)."""
    nn_run = config.MODELS / "nn" / VARIANT / B._nn_tag(k, dict(config.NN_SELECTED))
    logit_run = config.MODELS / "logit" / VARIANT / f"k{k}"
    scaler, vocab = F.load_pipeline(nn_run)        # shared by logit/nn/ensemble (same train slice)

    emp = E.empirical_matrix(k)                                   # [4, 7]
    want = (lambda name: which is None or name in which)
    torch_models: dict[str, object] = {}
    if want("logit"):
        torch_models["logit"] = E._load_logit(logit_run, scaler, vocab, device)
    if want("nn"):
        torch_models["nn"] = E._load_nn(nn_run, device)
    if want("ensemble") and k in E.KEY_WINDOWS:
        torch_models["ensemble"] = [
            E._load_nn(config.MODELS / "nn" / VARIANT / f"ens_k{k}_s{s}", device)
            for s in range(B.E.N_MEMBERS)]
    return scaler, vocab, emp, torch_models


def _chunk_matrices(scaler, vocab, emp, torch_models, base: pl.DataFrame,
                    t0: int, h: int, device) -> dict[str, np.ndarray]:
    """Per-model per-loan transition tensors ``[c, 7, 7]`` for horizon step ``h`` on a loan
    chunk. The empirical matrix is feature-free (constant over h); logit/NN/ensemble are scored
    on the evolved features. Ensemble = mean of the 8 members' transition matrices (= mean of
    member softmaxes, the paper Fig-7 rule; identical to ``evaluate``'s mean-of-probs at h=1)."""
    step_df = F.prepare_raw(advance_frame(base, t0, h))
    enc = F.encode_frame(step_df, scaler, vocab, encode="index")
    cont = torch.as_tensor(enc["cont"], dtype=torch.float32, device=device)
    binb = torch.as_tensor(enc["bin"], dtype=torch.float32, device=device)
    cat = torch.as_tensor(np.asarray(enc["cat"]), dtype=torch.long, device=device)
    c = cont.shape[0]

    mats: dict[str, np.ndarray] = {}
    # Empirical — same 4 rows for every loan, every step. Cast to float32 first so the h=1
    # composition reproduces evaluate.py's stored float32 empirical probabilities exactly.
    mats["empirical"] = assemble_matrix([np.broadcast_to(emp[oi].astype(np.float32), (c, N_CLASSES))
                                         for oi in range(len(ORIGIN_STATES))])
    # Torch scoring goes through the extracted seam (TorchPredictor._origin_scores_tensors == the
    # verbatim former `_origin_scores` body), keeping the single shared encode above. Constructing a
    # predictor per call is a few attribute writes; the scoring is byte-identical to the pre-seam path.
    for name, model in torch_models.items():
        if name == "ensemble":
            acc = [np.zeros((c, N_CLASSES), dtype=np.float64) for _ in ORIGIN_STATES]
            for m in model:
                for j, row in enumerate(
                        TorchPredictor(m, scaler, vocab, device)._origin_scores_tensors(cont, cat, binb)):
                    acc[j] += row
            scores = [a / len(model) for a in acc]
        else:
            scores = TorchPredictor(model, scaler, vocab, device)._origin_scores_tensors(cont, cat, binb)
        mats[name] = assemble_matrix(scores)
    return mats


def roll_forward(k: int, device, *, variant: str = VARIANT, chunk: int = DEFAULT_CHUNK,
                 max_loans: int | None = None, horizon: int = HORIZON,
                 capture_h1: bool = False, capture_smm: bool = False,
                 which: set | None = None, zero_impossible: bool = True) -> dict:
    """Roll every frozen window-``k`` model forward ``horizon`` months over all loans alive at
    the anchor ``t0 = Dec(k−1)`` (the eval-pool rows at ``period_ym == t0``), in bounded-memory
    loan chunks. Returns a per-loan result table (loan id, origin, and per-model
    ``prepaid_12m`` / ``dpd60p_12m`` probabilities), plus — when ``capture_h1`` — the full h=1
    distribution per model for the ``evaluate.py`` equality check.

    With ``capture_smm`` the harness also records, per loan and per model, the **monthly
    prepayment-hazard path** (``03_POOL_LEVEL §5.1``): ``out["smm"][m]`` holds two float32
    arrays ``[N, horizon]`` aligned to the result rows —

      * ``cum_prepaid[:, h-1]`` = composed ``p[prepaid]`` after step ``h`` (cumulative prepaid
        mass through month ``t0+h``); its first difference ``Δ_h`` is the *unconditional*
        probability loan ``i`` prepays during step ``h``;
      * ``alive_before[:, h-1]`` = transient (non-terminated) mass at the *start* of step ``h``.

    M15a (``smm_paths.py``) aggregates these UPB-weighted across a pool to the pool monthly
    ``SMM_P(h) = Σ_i w_i·Δ_h(i) / Σ_i w_i·alive_before_{h-1}(i)`` — a balance-of-survivors-weighted
    conditional monthly prepayment rate (the realized analogue comes from the panel).

    ``zero_impossible`` (default ``True``, the M20/§3.5 production behaviour M14/M15 bank) zeroes the
    two mechanically-impossible ``current→fc/REO`` cells before composition; ``False`` exposes the
    raw, un-zeroed path used only by :func:`verify_h1`'s legacy regression guard (so the raw h=1
    composition equals ``_score_direct`` bit-for-bit again, ADR-001 action item 2)."""
    global _STATE_IDX
    assert variant == VARIANT, "roll-forward consumes the full-export frozen models"
    t0 = config._dec(k - 1)
    scaler, vocab, emp, torch_models = _load_models(k, device, which=which)
    _STATE_IDX = {s: vocab.maps["state"][s] for s in ORIGIN_STATES}
    model_names = ["empirical", *torch_models.keys()]

    # All t0-alive loans = the eval-pool test slice rows at the anchor month (origin is
    # already transient by construction of the export). Load once; chunk for bounded memory.
    pool_dir, bounds = D.window_spec(variant, k, "test")
    df = (D._masked_scan(pool_dir, bounds)
          .filter(pl.col("period_ym") == t0)
          .collect())
    if max_loans is not None:
        df = df.head(max_loans)
    n_total = df.height

    loan_out, origin_out = [], []
    prep = {m: [] for m in model_names}
    dpd60 = {m: [] for m in model_names}
    h1_acc = {m: [] for m in model_names} if capture_h1 else None
    smm_cum = {m: [] for m in model_names} if capture_smm else None   # per-chunk [c, horizon]
    smm_alv = {m: [] for m in model_names} if capture_smm else None
    smm_cumd = {m: [] for m in model_names} if capture_smm else None  # 60+ first-passage (M24)

    for start in range(0, n_total, chunk):
        base = F.prepare_raw(df.slice(start, chunk))
        origin_idx = base.select(pl.col("state").replace_strict(
            list(SI), list(SI.values()), default=-1, return_dtype=pl.Int64)
        ).to_numpy().reshape(-1)
        p0 = onehot_origin(origin_idx)
        p = {m: p0.copy() for m in model_names}
        pfp = {m: p0.copy() for m in model_names}
        nc = base.height
        if capture_smm:
            cum_c = {m: np.empty((nc, horizon), np.float32) for m in model_names}
            alv_c = {m: np.empty((nc, horizon), np.float32) for m in model_names}
            cumd_c = {m: np.empty((nc, horizon), np.float32) for m in model_names}  # 60+ first-passage

        for h in range(1, horizon + 1):
            mats = _chunk_matrices(scaler, vocab, emp, torch_models, base, t0, h, device)
            for m in model_names:
                M = _zero_impossible(mats[m]) if zero_impossible else mats[m]   # M20: drop phantom current→fc/REO mass
                if capture_smm:                                   # alive mass at START of step h
                    alv_c[m][:, h - 1] = p[m][:, TRANSIENT_COLS].sum(axis=1)
                p[m] = np.einsum("ni,nij->nj", p[m], M)
                pfp[m] = np.einsum("ni,nij->nj", pfp[m], absorb_rows(M, DPD60))
                if capture_smm:                                   # cumulative prepaid / 60+ through step h
                    cum_c[m][:, h - 1] = p[m][:, PREPAID]
                    cumd_c[m][:, h - 1] = pfp[m][:, DPD60]        # ever-60+ first-passage mass (M24)
                if capture_h1 and h == 1:
                    h1_acc[m].append(p[m].copy())

        loan_out.append(base.get_column("Loan Identifier").to_numpy())
        origin_out.append(base.get_column("state").to_numpy())
        for m in model_names:
            prep[m].append(p[m][:, PREPAID])
            dpd60[m].append(pfp[m][:, DPD60])
            if capture_smm:
                smm_cum[m].append(cum_c[m])
                smm_alv[m].append(alv_c[m])
                smm_cumd[m].append(cumd_c[m])

    cols = {"Loan Identifier": np.concatenate(loan_out),
            "origin": np.concatenate(origin_out)}
    for m in model_names:
        cols[f"{m}_prepaid_12m"] = np.concatenate(prep[m])
        cols[f"{m}_dpd60p_12m"] = np.concatenate(dpd60[m])
    result = pl.DataFrame(cols)

    out = {"k": k, "t0": t0, "n_loans": n_total, "model_names": model_names,
           "horizon": horizon, "result": result}
    if capture_h1:
        out["h1"] = {m: np.concatenate(h1_acc[m]) for m in model_names}
    if capture_smm:
        out["smm"] = {m: {"cum_prepaid": np.concatenate(smm_cum[m]),
                          "alive_before": np.concatenate(smm_alv[m]),
                          "cum_dpd60p": np.concatenate(smm_cumd[m])} for m in model_names}
    return out


# ===========================================================================
# The agnostic engine (ECONOMIC_ENGINE §3.3 / ADR-001) — ONE injected Predictor, horizon a
# parameter, optional calibrator + first-passage absorb. Drives the SAME untouched primitives
# (assemble_matrix / absorb_rows / _zero_impossible / compose) as the legacy multi-model
# roll_forward; only `predictor.origin_scores` is model-specific.
# ===========================================================================
def roll_forward_predictor(predictor: Predictor, base: pl.DataFrame, t0: int, *,
                           horizon: int = HORIZON, snapshots=None,
                           calibrator: Calibrator | None = None, absorb: int | None = None,
                           zero_impossible: bool = True, chunk: int = DEFAULT_CHUNK,
                           capture_h1: bool = False, capture_smm: bool = False) -> dict:
    """Roll ONE injected ``predictor`` forward ``horizon`` months over the loans in ``base`` (an
    anchor slice at ``t0``, already ``prepare_raw``-ed), in bounded-memory chunks, and snapshot the
    composed distribution at each ``h`` in ``snapshots`` (default ``config.HORIZONS``). Because the
    chain is rolled **once** to ``horizon`` and read off at the snapshot months, the intermediate
    horizons are free (cost is ``horizon × population``, not ``Σ_h``).

    Per step (``ECONOMIC_ENGINE §4``):
      ``scores = predictor.origin_scores(advance_frame(...))`` → ``calibrator`` on the **raw**
      per-origin scores (item 2: calibrate *before* assembly) → :func:`assemble_matrix` →
      :func:`_zero_impossible` (M20, if ``zero_impossible``) → :func:`absorb_rows` (if ``absorb`` is
      a state index — the first-passage chain) → compose.

    With ``capture_smm`` (M22) the harness also **records what the loop already computes** at every
    step — no second roll: ``out["smm"] = {"cum_prepaid": [N, horizon], "alive_before": [N, horizon]}``
    (float32, row-aligned to ``base``), where ``cum_prepaid[:, h−1]`` is the composed prepaid mass
    after step ``h`` (cumulative through ``t0+h``) and ``alive_before[:, h−1]`` is the transient
    (non-terminated) mass at the **start** of step ``h``. :func:`per_loan_smm` turns these into the
    per-loan conditional SMM path that :func:`price_loans` consumes (M22 / §3.4). Capture is the
    normal-chain prepay increment, so ``absorb`` must be ``None`` (the first-passage chain redirects
    mass and would not give the prepayment path).

    Returns ``{"snapshots": {h: P[N, 7]}, "n_loans": N}`` (plus ``"h1"`` when ``capture_h1``, plus
    ``"smm"`` when ``capture_smm``), the arrays aligned to ``base``'s row order. ``horizon=1,
    zero_impossible=False`` is the parameterized M13 Accept-#2 identity (must equal ``evaluate.py``
    exactly)."""
    if snapshots is None:
        snapshots = config.HORIZONS
    if capture_smm and absorb is not None:
        raise ValueError("capture_smm records the normal-chain prepay increment; absorb must be None")
    snaps = sorted({s for s in snapshots if 1 <= s <= horizon})
    n_total = base.height
    acc = {h: [] for h in snaps}
    h1_acc = [] if capture_h1 else None
    smm_cum, smm_alv = ([], []) if capture_smm else (None, None)   # per-chunk [c, horizon]

    for start in range(0, n_total, chunk):
        chunk_df = base.slice(start, chunk)
        origin_idx = chunk_df.select(pl.col("state").replace_strict(
            list(SI), list(SI.values()), default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1)
        p = onehot_origin(origin_idx)
        nc = chunk_df.height
        if capture_smm:
            cum_c = np.empty((nc, horizon), np.float32)
            alv_c = np.empty((nc, horizon), np.float32)
        for h in range(1, horizon + 1):
            scores = predictor.origin_scores(advance_frame(chunk_df, t0, h))
            if calibrator is not None:                              # §4 item 2: calibrate raw scores
                scores = [calibrator(s, o) for o, s in enumerate(scores)]
            P_h = assemble_matrix(scores)
            if zero_impossible:                                     # M20 / §3.5
                P_h = _zero_impossible(P_h)
            if absorb is not None:                                  # first-passage chain (03 §3)
                P_h = absorb_rows(P_h, absorb)
            if capture_smm:                                         # alive mass at START of step h
                alv_c[:, h - 1] = p[:, TRANSIENT_COLS].sum(axis=1)
            p = np.einsum("ni,nij->nj", p, P_h)                     # = compose(p, [P_h]) (one step)
            if capture_smm:                                         # cumulative prepaid through step h
                cum_c[:, h - 1] = p[:, PREPAID]
            if capture_h1 and h == 1:
                h1_acc.append(p.copy())
            if h in acc:
                acc[h].append(p.copy())
        if capture_smm:
            smm_cum.append(cum_c); smm_alv.append(alv_c)

    out = {"n_loans": n_total, "horizon": horizon,
           "snapshots": {h: np.concatenate(acc[h]) for h in snaps}}
    if capture_h1:
        out["h1"] = np.concatenate(h1_acc)
    if capture_smm:
        out["smm"] = {"cum_prepaid": np.concatenate(smm_cum),
                      "alive_before": np.concatenate(smm_alv)}
    return out


# ===========================================================================
# Accept #2 — h=1 composition equals evaluate.py outputs exactly
# ===========================================================================
# cross-check tolerance: cuBLAS picks different GEMM algorithms for different batch shapes, so
# the *committed* full-slice evaluate run and a fresh anchor-only scoring of the deep net agree
# only to GPU floating-point precision (~1e-6), not bit-for-bit. The harness exactness is proven
# separately (identical batching ⇒ Δ=0); this bounds the residual against the committed numbers.
XCHECK_TOL = 1e-5


def _score_direct(base: pl.DataFrame, scaler, vocab, emp, torch_models, origin_idx, device) -> dict:
    """The plain loan-level prediction for each anchor row: score it ONCE with its *true* origin
    state — exactly what ``evaluate.py`` computes per row. Same encode + same float64-softmax
    forward as the roll-forward, so it is the bit-exact reference for the h=1 composition."""
    enc = F.encode_frame(base, scaler, vocab, encode="index")
    cont = torch.as_tensor(enc["cont"], dtype=torch.float32, device=device)
    binb = torch.as_tensor(enc["bin"], dtype=torch.float32, device=device)
    cat = torch.as_tensor(np.asarray(enc["cat"]), dtype=torch.long, device=device)
    direct = {"empirical": emp[origin_idx].astype(np.float32)}      # 4×7 indexed by true origin
    for name, model in torch_models.items():
        if name == "ensemble":
            acc = np.zeros((cont.shape[0], N_CLASSES), dtype=np.float64)
            for m in model:
                acc += _forward_probs(m, cont, cat, binb)
            direct[name] = (acc / len(model))
        else:
            direct[name] = _forward_probs(model, cont, cat, binb)
    return direct


def _build_predictors(scaler, vocab, emp, torch_models, device) -> dict:
    """The four anchor models behind the seam: the empirical-matrix stub plus a TorchPredictor per
    torch model (an EnsemblePredictor over the 8 members for ``ensemble``)."""
    preds: dict[str, Predictor] = {"empirical": EmpiricalPredictor(emp)}
    for name, model in torch_models.items():
        preds[name] = (EnsemblePredictor([TorchPredictor(m, scaler, vocab, device) for m in model])
                       if name == "ensemble" else TorchPredictor(model, scaler, vocab, device))
    return preds


def verify_h1(k: int, device, max_loans: int | None = None, *, xcheck: bool = False) -> dict:
    """At h=1 the evolution is the identity (age+0, remaining−0, month=t0), so the composed 1-step
    distribution ``one-hot(origin)·P_1`` must equal the row of ``P_1`` for the loan's true origin —
    the model's per-row prediction that ``evaluate.py`` scores. All checks compare against
    :func:`_score_direct` (each anchor row scored ONCE with its true origin — evaluate.py's exact
    per-row computation), the independent reference unaffected by the seam refactor.

    **Part A — legacy regression guard** (the pre-seam ``verify_h1`` assertion, kept): the legacy
    multi-model :func:`roll_forward` at h=1 on the **raw, un-zeroed** path
    (``zero_impossible=False``) equals ``_score_direct`` **bit-for-bit**, proving the
    ``_chunk_matrices`` repoint did not alter the scoring M14/M15 banked.

    **Part B — staged seam check** via :func:`roll_forward_predictor`, asserted IN ORDER so a
    divergence localises instead of needing a bisection:
      (a) the h=1 input frame == the eval input frame (``advance_frame(·,1)`` is the identity);
      (b) the predictor's per-origin output (gathered at each loan's true origin) == ``_score_direct``;
      (c) the composed h=1 vector (raw path) == ``_score_direct`` EXACTLY.

    With ``xcheck`` (a GPU/full session — expensive, scores the whole window), also cross-check the
    composed h=1 against the **committed** ``evaluate.score_window`` to ``XCHECK_TOL`` (the deep
    net's residual is cuBLAS GEMM batch-shape nondeterminism; logit/empirical match exactly). The
    tolerance is unchanged; ``xcheck`` only gates whether this expensive comparison runs."""
    global _STATE_IDX
    t0 = config._dec(k - 1)
    scaler, vocab, emp, torch_models = _load_models(k, device)
    _STATE_IDX = {s: vocab.maps["state"][s] for s in ORIGIN_STATES}
    model_names = ["empirical", *torch_models.keys()]

    pool_dir, bounds = D.window_spec(VARIANT, k, "test")
    base = (D._masked_scan(pool_dir, bounds).filter(pl.col("period_ym") == t0).collect())
    if max_loans is not None:
        base = base.head(max_loans)
    base = F.prepare_raw(base)
    n = base.height
    origin_idx = base.select(pl.col("state").replace_strict(
        list(SI), list(SI.values()), default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1)

    # Independent reference — evaluate.py's per-row computation on the capped slice.
    direct = _score_direct(base, scaler, vocab, emp, torch_models, origin_idx, device)
    preds = _build_predictors(scaler, vocab, emp, torch_models, device)

    # ----- Part A — legacy regression guard (raw, un-zeroed legacy path) -----
    legacy = roll_forward(k, device, chunk=n + 1, max_loans=max_loans, horizon=1,
                          capture_h1=True, zero_impossible=False)
    print("  -- Part A: legacy roll_forward (raw path) byte-identical vs _score_direct --")
    legacy_checks, legacy_ok = [], True
    for m in model_names:
        a = legacy["h1"][m]
        eq = a.shape == direct[m].shape and np.array_equal(a, direct[m])
        legacy_ok = legacy_ok and eq
        legacy_checks.append({"model": m, "byte_identical": bool(eq)})
        print(f"     [{'PASS' if eq else 'FAIL'}] legacy h=1 {m}: {a.shape[0]:,} rows | byte-identical={eq}")

    # ----- Part B — staged seam check (a -> b -> c) -----
    block_of_loan = np.array([ORIGIN_ROWS.index(o) for o in origin_idx])   # class idx -> origin block
    frame1 = advance_frame(base, t0, 1)
    print("  -- Part B: staged seam check (a) frame -> (b) predictor -> (c) composed --")
    staged, staged_ok = [], True
    for m in model_names:
        pred = preds[m]
        # (a) input-frame equality: at h=1 advance_frame is the identity, so the encoded inputs match.
        if isinstance(pred, TorchPredictor):
            stage_a = all(torch.equal(x, y) for x, y in zip(pred._encode(frame1), pred._encode(base)))
        else:                                                          # feature-free predictor
            stage_a = (frame1.height == base.height)
        # (b) predictor per-origin output, gathered at each loan's true origin, vs the direct forward.
        scores = pred.origin_scores(frame1)
        gathered = (np.stack(scores, axis=0)[block_of_loan, np.arange(n)] if n
                    else np.empty((0, N_CLASSES)))
        stage_b = gathered.shape == direct[m].shape and np.array_equal(gathered, direct[m])
        # (c) composed h=1 on the raw path == direct, bit-exact (the parameterized M13 Accept #2).
        comp = roll_forward_predictor(pred, base, t0, horizon=1, snapshots=(1,),
                                      zero_impossible=False, chunk=n + 1, capture_h1=True)["h1"]
        stage_c = comp.shape == direct[m].shape and np.array_equal(comp, direct[m])
        first_fail = (None if (stage_a and stage_b and stage_c)
                      else "a" if not stage_a else "b" if not stage_b else "c")
        ok_m = first_fail is None
        staged_ok = staged_ok and ok_m
        staged.append({"model": m, "stage_a_frame": bool(stage_a), "stage_b_predictor": bool(stage_b),
                       "stage_c_composed": bool(stage_c), "first_failing_stage": first_fail})
        tag = "PASS" if ok_m else f"FAIL@stage-{first_fail}"
        print(f"     [{tag}] staged h=1 {m}: (a)frame={stage_a} (b)predictor={stage_b} "
              f"(c)composed={stage_c}")

    out = {"all_pass": legacy_ok and staged_ok, "n_anchor": int(n),
           "legacy_guard": {"all_pass": legacy_ok, "checks": legacy_checks},
           "staged": {"all_pass": staged_ok, "checks": staged}}

    # ----- optional: committed-run cross-check (expensive; GPU/full session, like M20b) -----
    if xcheck:
        ev = E.score_window(k, device)
        period = (D._masked_scan(pool_dir, bounds)
                  .select("period_ym").collect().get_column("period_ym").to_numpy())
        anchor = period == t0
        if max_loans is not None:
            keep = np.where(anchor)[0][:max_loans]
            anchor = np.zeros(anchor.shape[0], dtype=bool)
            anchor[keep] = True
        print(f"  -- xcheck: composed h=1 vs committed evaluate.score_window (≤{XCHECK_TOL:g}) --")
        xchecks, xok = [], True
        for m in model_names:
            a = legacy["h1"][m]
            b = ev["probs"][m][anchor] if m in ev["probs"] else None
            xmax = float(np.abs(a - b).max()) if b is not None and a.shape == b.shape else None
            within = (xmax is None) or (xmax <= XCHECK_TOL)
            xok = xok and within
            xchecks.append({"model": m, "xcheck_max_abs_diff": xmax,
                            "xcheck_bit_identical": (xmax == 0.0) if xmax is not None else None,
                            "xcheck_within_tol": bool(within)})
            xstr = "n/a" if xmax is None else f"{xmax:.2e}"
            print(f"     [{'PASS' if within else 'FAIL'}] xcheck h=1 {m}: vs evaluate max|Δ|={xstr}")
        out["xcheck"] = {"all_pass": xok, "tol": XCHECK_TOL, "checks": xchecks}
        out["all_pass"] = out["all_pass"] and xok

    return out


# ===========================================================================
# Driver
# ===========================================================================
def _peak_rss_mb() -> float:
    import resource
    # ru_maxrss is KB on Linux, bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return rss / 1024 if sys.platform != "darwin" else rss / (1024 * 1024)


def run_horizons(k: int, device, *, max_loans: int | None, chunk: int, snapshots=None) -> dict:
    """M21 multi-horizon plumbing proof (ECONOMIC_ENGINE §5): ONE anchor, a capped slice, a SINGLE
    roll to ``max(snapshots)`` with the composed distribution snapshotted at H ∈ snapshots — the
    intermediates are free. Runs a torch predictor (logit) AND the EmpiricalPredictor **stub**
    through the identical :func:`roll_forward_predictor`, to show the engine is model-agnostic and
    memory stays bounded. NOT the full-population ≥3-anchor grid (that is M24)."""
    global _STATE_IDX
    snapshots = tuple(snapshots or config.HORIZONS)
    horizon = max(snapshots)
    t0 = config._dec(k - 1)
    scaler, vocab, emp, torch_models = _load_models(k, device, which={"logit"})
    _STATE_IDX = {s: vocab.maps["state"][s] for s in ORIGIN_STATES}

    pool_dir, bounds = D.window_spec(VARIANT, k, "test")
    base = D._masked_scan(pool_dir, bounds).filter(pl.col("period_ym") == t0).collect()
    if max_loans is not None:
        base = base.head(max_loans)
    base = F.prepare_raw(base)
    n = base.height

    preds = {"logit": TorchPredictor(torch_models["logit"], scaler, vocab, device),
             "empirical_stub": EmpiricalPredictor(emp)}
    models = {}
    print(f"  rolling {n:,} loans to H={horizon} (single roll), snapshots {list(snapshots)}")
    for name, pred in preds.items():
        tr = time.perf_counter()
        rf = roll_forward_predictor(pred, base, t0, horizon=horizon, snapshots=snapshots, chunk=chunk)
        wall = time.perf_counter() - tr
        snaps = rf["snapshots"]
        shapes_ok = all(snaps[h].shape == (n, N_CLASSES) for h in snaps)
        valid = all(bool((snaps[h] >= 0).all() and (snaps[h] <= 1).all()
                         and np.allclose(snaps[h].sum(1), 1.0)) for h in snaps)
        prepaid = {h: float(snaps[h][:, PREPAID].mean()) for h in sorted(snaps)}
        models[name] = {"snapshots_produced": sorted(snaps), "all_shapes_ok": shapes_ok,
                        "all_valid_dists": valid, "E_prepaid_by_h": prepaid, "roll_sec": wall}
        print(f"  {name:>14}: H={sorted(snaps)} produced  shapes_ok={shapes_ok}  valid={valid}  "
              + "  ".join(f"E[prepaid|h{h}]={prepaid[h]:.4f}" for h in sorted(prepaid))
              + f"  [{wall:.0f}s]")
    peak = _peak_rss_mb()
    print(f"  peak RSS {peak:,.0f} MB  (single roll to {horizon}, {len(snapshots)} free snapshots)")
    return {"k": k, "t0": t0, "n_loans": n, "horizon": horizon, "snapshots": list(snapshots),
            "peak_rss_mb": peak, "models": models}


def run(k: int, device, *, chunk: int, max_loans: int | None, do_verify: bool,
        do_horizons: bool = False, xcheck: bool = False) -> dict:
    t0 = time.perf_counter()
    print(f"=== M13 roll-forward  k={k}  anchor=Dec{k-1}  device={device}"
          f"  chunk={chunk:,}{' max_loans=' + format(max_loans, ',') if max_loans else ''} ===")

    summary: dict = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "variant": VARIANT, "k": k, "git_commit": T._git_commit(),
                     "device": str(device), "chunk": chunk, "max_loans": max_loans}

    if do_verify:
        print("\n[M21 Accept] staged h=1 identity + legacy regression guard (bit-identical)")
        summary["h1_vs_evaluate"] = verify_h1(k, device, max_loans=max_loans, xcheck=xcheck)

    if do_horizons:
        print("\n[M21 Accept] multi-horizon snapshots H∈{1,3,6,12} of a single roll to 12")
        summary["horizons"] = run_horizons(k, device, max_loans=max_loans, chunk=chunk)

    if do_verify or do_horizons:                         # M21 smoke sections only; skip the full roll
        summary["wall_sec"] = time.perf_counter() - t0
        out_dir = config.MODELS / "nn" / VARIANT
        out = out_dir / f"pool_m21_k{k}{'_smoke' if max_loans else ''}.json"
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote {out}  [{summary['wall_sec']:.0f}s]")
        return summary

    print("\n[Accept 3] full roll-forward over all t0-alive loans (bounded-memory chunks)")
    tr = time.perf_counter()
    rf = roll_forward(k, device, chunk=chunk, max_loans=max_loans, horizon=HORIZON)
    wall = time.perf_counter() - tr
    res = rf["result"]
    # Sanity: probabilities are valid and the harness produced one row per alive loan.
    sane = {}
    for m in rf["model_names"]:
        pr = res.get_column(f"{m}_prepaid_12m").to_numpy()
        dq = res.get_column(f"{m}_dpd60p_12m").to_numpy()
        sane[m] = {"prepaid_mean": float(pr.mean()), "dpd60p_mean": float(dq.mean()),
                   "prepaid_in_01": bool((pr >= 0).all() and (pr <= 1).all()),
                   "dpd60p_in_01": bool((dq >= 0).all() and (dq <= 1).all())}
        print(f"  {m:>9}: E[prepaid≤12m]={pr.mean():.4f}  E[60+≤12m]={dq.mean():.4f}  "
              f"in[0,1]={sane[m]['prepaid_in_01'] and sane[m]['dpd60p_in_01']}")
    peak = _peak_rss_mb()
    print(f"  {rf['n_loans']:,} alive loans rolled {HORIZON}m in {wall:.0f}s  "
          f"peak RSS {peak:,.0f} MB  rows_out={res.height:,}")

    summary.update({"n_loans": rf["n_loans"], "rows_out": res.height, "roll_sec": wall,
                    "peak_rss_mb": peak, "sanity": sane,
                    "outcome_means": {m: sane[m] for m in sane}})
    summary["wall_sec"] = time.perf_counter() - t0

    out_dir = config.MODELS / "nn" / VARIANT
    out = out_dir / f"pool_rollforward_k{k}{'_smoke' if max_loans else ''}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}  [{summary['wall_sec']:.0f}s]")
    return summary


SMOKE_CAP = 50_000          # default --smoke loan cap (standing rule 3: ≤100k CPU-testable slice)


def main() -> None:
    ap = argparse.ArgumentParser(description="M13/M21 roll-forward harness (Phase-3 §3 engine + "
                                             "ECONOMIC_ENGINE §3 predictor seam).")
    ap.add_argument("--k", type=int, default=2020, help="test year / window (anchor = Dec(k-1))")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="loans per chunk")
    ap.add_argument("--max-loans", type=int, default=None, help="cap alive loans (smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"cap alive loans at {SMOKE_CAP:,} if --max-loans unset (local CPU proof)")
    ap.add_argument("--verify", action="store_true",
                    help="staged h=1 identity + legacy regression guard (raw, un-zeroed path)")
    ap.add_argument("--horizons", action="store_true",
                    help="multi-horizon snapshot proof: single roll to 12, snapshot H∈{1,3,6,12}")
    ap.add_argument("--xcheck", action="store_true",
                    help="also cross-check h=1 vs committed evaluate.score_window (GPU/full; expensive)")
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda"
              else tc.resolve_device(args.device))
    max_loans = args.max_loans if args.max_loans is not None else (SMOKE_CAP if args.smoke else None)
    run(args.k, device, chunk=args.chunk, max_loans=max_loans, do_verify=args.verify,
        do_horizons=args.horizons, xcheck=args.xcheck)


if __name__ == "__main__":
    main()
